import sys
import os
import json
import pickle
import argparse
import datetime
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from easydict import EasyDict as edict
sys.path.append('../')
from nets.RoarNet_3D_RPN import Net
from datasets.data_RoarNet_3D_RPN import ObjectProvider
from helpers.timer import Timer
from helpers.misc import cprint, mkdir_p, get_configuration
from helpers.paul_geometry import *

slim = tf.contrib.slim

DECAY_STEP = 200000
BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99



def _configure_learning_rate(cfg, num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.
    Returns:
      A `Tensor` representing the learning rate.
    """
    decay_steps = int(num_samples_per_epoch * cfg.solver.num_epochs_per_decay /
                      cfg.solver.batch_size)

    if cfg.solver.learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(cfg.solver.learning_rate,
                                          global_step,
                                          decay_steps,
                                          cfg.solver.learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif cfg.solver.learning_rate_decay_type == 'fixed':
        return tf.constant(cfg.solver.learning_rate, name='fixed_learning_rate')
    elif cfg.solver.learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(cfg.solver.learning_rate,
                                         global_step,
                                         decay_steps,
                                         cfg.solver.end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         cfg.solver.learning_rate_decay_type)

def get_bn_decay(cfg, num_samples_per_epoch, global_step):
    decay_steps = int(num_samples_per_epoch * cfg.solver.num_epochs_per_decay /
                      cfg.solver.batch_size)

    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      global_step,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def _set_filewriters(ckpt_dir, sess):
    train_writer = tf.summary.FileWriter(os.path.join(ckpt_dir, 'train'), sess.graph)
    eval_writer = tf.summary.FileWriter(os.path.join(ckpt_dir, 'eval'), sess.graph)
    return train_writer, eval_writer


def _average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
              is over individual gradients. The inner list is over the gradient
              calculation for each tower.
    Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g is not None:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

        if grads == []: continue
        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def _configure_optimizer(cfg, learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    """
    if cfg.solver.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=cfg.solver.adadelta_rho,
            epsilon=cfg.solver.opt_epsilon)
    elif cfg.solver.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=cfg.solver.adagrad_initial_accumulator_value)
    elif cfg.solver.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=cfg.solver.adam_beta1,
            beta2=cfg.solver.adam_beta2,
            epsilon=cfg.solver.adam_epsilon)
    elif cfg.solver.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=cfg.solver.ftrl_learning_rate_power,
            initial_accumulator_value=cfg.solver.ftrl_initial_accumulator_value,
            l1_regularization_strength=cfg.solver.ftrl_l1,
            l2_regularization_strength=cfg.solver.ftrl_l2)
    elif cfg.solver.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=cfg.solver.momentum,
            name='Momentum')
    elif cfg.solver.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=cfg.solver.rmsprop_decay,
            momentum=cfg.solver.rmsprop_momentum,
            epsilon=cfg.solver.opt_epsilon)
    elif cfg.solver.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', cfg.solver.optimizer)
    return optimizer



def main(cfg):
    cprint ("-- preparing..")
    max_iter = cfg['solver']['max_iter']
    summary_iter = cfg['solver']['summary_iter']
    save_iter = cfg['solver']['save_iter']
    ckpt_dir = os.path.join(cfg['path']['ckpt_dir'], cfg.cfg_name)
    ckpt_file = os.path.join(ckpt_dir, cfg.cfg_name)

    output_path = '../train_inter_results/'
    mkdir_p(output_path)


    tf.logging.info("-- constructing network..")
    with tf.Graph().as_default():

        with tf.device('/cpu:0'):
            with tf.name_scope('data_provider'):
                sample = {'sample_radii': cfg.radii,
                          'cls': cfg.cls,
                          'x_min': cfg.min_x,
                          'y_min': cfg.min_y,
                          'z_min': cfg.min_z,
                          'x_max': cfg.max_x,
                          'y_max': cfg.max_y,
                          'z_max': cfg.max_z,
                          'num_points': cfg.num_points,
                          'CENTER_PERTURB': cfg.CENTER_PERTURB,
                          'CENTER_PERTURB_Z': cfg.CENTER_PERTURB_Z,
                          'SAMPLE_Z_MIN': cfg.sample_z_min,
                          'SAMPLE_Z_MAX': cfg.sample_z_max,
                          'QUANT_POINTS': cfg.QUANT_POINTS,
                          'QUANT_LEVEL': cfg.QUANT_LEVEL}
                dataset = ObjectProvider(edict({'batch_size': cfg.solver.batch_size,
                                        'dataset': 'kitti',
                                        'split': 'train',
                                        'is_training': True,
                                        'num_epochs': None,
                                        'sample': sample}))
                dataset.data_size = 100000   # FIXME. random number

            global_step = tf.get_variable('global_step', [],
                initializer=tf.constant_initializer(0), trainable=False)
            learning_rate = _configure_learning_rate(cfg, dataset.data_size, global_step)
            bn_decay = get_bn_decay(cfg, dataset.data_size, global_step)

            optimizer = _configure_optimizer(cfg, learning_rate)
            tf.summary.scalar('learning_rate', learning_rate)


        # Calculate the gradients for each model tower.
        towers_ph_points = []
        towers_ph_obj = []
        towers_ph_is_training = []

        tower_grads = []
        tower_losses = []
        device_scopes = []
        scope_name = 'rpn'
        with tf.variable_scope(scope_name):
            for gid in range(cfg.num_gpus):
                with tf.name_scope('gpu%d' % gid) as scope:
                    with tf.device('/gpu:%d' % gid):
                        with tf.name_scope("train_input"):
                            ph_points = tf.placeholder(tf.float32, shape=(None, cfg.num_points, 3))
                            ph_obj = tf.placeholder(tf.float32, shape=(None,))

                            ph_is_training = tf.placeholder(tf.bool, shape=())

                            net = Net(ph_points=ph_points, is_training=ph_is_training, bn_decay=bn_decay, cfg=cfg)
                            net.losses(target_objs=ph_obj, cut_off=cfg.iou_cutoff, gl = global_step)

                        all_losses = tf.get_collection(tf.GraphKeys.LOSSES, scope)
                        sum_loss = tf.add_n(all_losses)

                        for loss in all_losses: tf.summary.scalar(loss.op.name, loss)
                        tf.summary.scalar("sum_loss_tower", sum_loss)

                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data
                        grads = optimizer.compute_gradients(sum_loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)
                        tower_losses.append(sum_loss)
                        device_scopes.append(scope)

                        # Collect all placeholders
                        towers_ph_points.append(ph_points)
                        towers_ph_obj.append(ph_obj)
                        towers_ph_is_training.append(ph_is_training)

        total_loss = tf.add_n(tower_losses, name='total_loss')
        grads = _average_gradients(tower_grads)
        apply_gradient_ops = optimizer.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # Track the moving averages of all trainable variables.
        # if cfg.solver.moving_average_decay:
        with tf.name_scope('expMovingAverage'):
            variable_averages = tf.train.ExponentialMovingAverage(
                0.005, global_step)
                # cfg.solver.moving_average_decay, global_step)
            averages_op = variable_averages.apply(tf.trainable_variables())
        # else:
            # averages_op = None

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_ops, averages_op)
        train_tensor = control_flow_ops.with_dependencies([train_op], total_loss,
                                                          name='train_op')

        # Create a saver
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        init = tf.global_variables_initializer()

        # =================================================================== #
        # Kicks off the training.
        # =================================================================== #\
        # GPU configuration
        if cfg.num_gpus == 0:   config = tf.ConfigProto(device_count={'GPU': 0})
        else:                   config = tf.ConfigProto(allow_soft_placement=True,
                                                        log_device_placement=False)

        with tf.Session(config=config) as sess:

            dataset.set_session(sess)

            # initialization / session / writer / saver
            print('initializing a network may take minutes...')
            sess.run(init)

            tf.train.start_queue_runners(sess=sess)

            train_writer, eval_writer = _set_filewriters(ckpt_dir, sess)
            merged = tf.summary.merge_all()

            ckpt_dir = os.path.join(cfg.path.ckpt_dir, cfg.cfg_name)
            weight_file = tf.train.latest_checkpoint(ckpt_dir)
            if weight_file is not None:
                saver.restore(sess, weight_file)
                tf.logging.info('%s loaded' % weight_file)
            else:
                tf.logging.info('Training from the scratch (no pre-trained weight_filets)..')

            train_timer = Timer()
            print ('start training...')
            stat_correct = []

            def inference(feed_dict):
                loc, conf_iou= sess.run([net.pred_locs, net.pred_conf_iou], feed_dict)
                return loc, conf_iou

            for step in range(max_iter):

                train_timer.tic()
                feed_dict = {}
                for i in range(cfg.num_gpus):
                    b_points, b_objs, b_locs = dataset.get_batch()
                    feed_dict[towers_ph_points[i]] = b_points
                    feed_dict[towers_ph_obj[i]] = b_objs
                    feed_dict[towers_ph_is_training[i]] = True

                gl, loss, _ = sess.run([global_step, total_loss, train_tensor], feed_dict=feed_dict)
                
                if gl % 100  == 0:                    
                    cprint("gl: {} loss: {:.3f}".format(gl, loss))

                train_timer.toc()
                if gl % summary_iter == 0:
                    if gl % (summary_iter * 10) == 0:
                        # Summary with run meta data
                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        run_metadata = tf.RunMetadata()
                        summary_str, loss, _ = sess.run([merged, total_loss, train_tensor],
                                                        feed_dict=feed_dict,
                                                        options=run_options,
                                                        run_metadata=run_metadata)
                        train_writer.add_run_metadata(run_metadata, 'step_{}'.format(gl), gl)
                        train_writer.add_summary(summary_str, gl)
                    else:
                        # Summary
                        summary_str = sess.run(merged, feed_dict=feed_dict)
                        train_writer.add_summary(summary_str, gl)

                    log_str = ('{} Epoch: {:3d}, Step: {:4d}, Learning rate: {:.4e}, Loss: {:5.3f}\n'
                        '{:14s} Speed: {:.3f}s/iter, Remain: {}').format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%S'),
                        int(cfg.solver.batch_size * gl / dataset.data_size),
                        int(gl),
                        round(learning_rate.eval(session=sess), 6),
                        loss,
                        '',
                        train_timer.average_time,
                        train_timer.remain(step, max_iter))
                    print(log_str)
                    train_timer.reset()

                if gl % save_iter == 0:
                    print('{} Saving checkpoint file to: {}'.format(
                        datetime.datetime.now().strftime('%m/%d %H:%M:%S'), ckpt_dir))
                    saver.save(sess, ckpt_file, global_step=global_step)

                evaluate_iter = 100000




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hello, deep learning world!')
    parser.add_argument("--conf", required=True, action="store", help="training configuration file")
    parser.add_argument("--gpuid", default="0", action="store", help="specify which gpu to turn on. e.g., 0,1,2")
    parser.add_argument("--iou_cutoff", default="0.73", type=float, action="store", help="specify which gpu to turn on. e.g., 0,1,2")
    parser.add_argument("--class", default="car", type=str, action="store", help="specify class of object to train. e.g., car, ped, cyclist")
    parser.add_argument("--label", default=False, action="store", help="generate result label file for validation")

    args = vars(parser.parse_args())

    ext = os.path.splitext(args['conf'])[-1]
    if ext == '.json':
        with open(args['conf']) as jf: cfg = json.load(jf)
    elif ext == '.py':
        import re
        from importlib import import_module
        spl = re.split('/', args['conf'])[1:]
        spl[-1] = spl[-1][:-3]
        modulename = ('.').join(spl)
        cfg = get_configuration(modulename, args['class'])

    # there shouldn't be common keys. ignore for now.
    cfg = {**cfg, **args}
    cfg = edict(cfg)
    cfg.cfg_name = os.path.splitext(os.path.basename(args['conf']))[0]

    if cfg.gpuid == "-1":
        cfg.num_gpus = 0
        print ("Training must use at least one GPU.")
        quit()
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpuid
        cfg.num_gpus = len(cfg.gpuid.split(','))

    tf.logging.set_verbosity(tf.logging.DEBUG)
    main(cfg)







