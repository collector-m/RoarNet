from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import numpy as np
import tensorflow as tf
sys.path.append('../')
from helpers import tf_util

def pointnet(point_cloud,
            num_outputs,
            is_training=True,
            bn_decay=None,
            dropout_keep_prob=0.7):
  '''
  Ref: from Charles R. Qi's implementation of PointNet(: https://github.com/charlesq34/pointnet)
  '''
  num_point = point_cloud.get_shape()[1].value
  input_image = tf.expand_dims(point_cloud, 2)
  bn = True

  net = tf_util.conv2d(input_image, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=bn, is_training=is_training,
                       scope='conv1', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=bn, is_training=is_training,
                       scope='conv2', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=bn, is_training=is_training,
                       scope='conv3', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=bn, is_training=is_training,
                       scope='conv4', bn_decay=bn_decay)
  net = tf_util.conv2d(net, 1024, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=bn, is_training=is_training,
                       scope='conv5', bn_decay=bn_decay)

  # Symmetric function: max pooling
  net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='maxpool')
  net = tf.squeeze(net, axis=[1,2])

  net = tf_util.fully_connected(net, 512, bn=bn, is_training=is_training, scope='fc1', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='dp1')

  net = tf_util.fully_connected(net, 256, bn=bn, is_training=is_training, scope='fc2', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='dp2')

  net = tf_util.fully_connected(net, 128, bn=bn, is_training=is_training, scope='fc3', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='dp3')

  net = tf_util.fully_connected(net, 128, bn=bn, is_training=is_training, scope='fc4', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=dropout_keep_prob, is_training=is_training, scope='dp4')

  pred = tf_util.fully_connected(net, num_outputs, activation_fn=None, scope='fc5')
  pred = tf.reshape(pred, [-1, num_outputs])

  return pred

class RoarNet_3D_RPN(object):

    def __init__(self, ph_points, is_training, bn_decay=None, cfg=None, scope=''):
      self.is_training = is_training
      self.inputs = ph_points
      self.bn_decay = bn_decay       
      self.cfg = cfg
      self.n_objectness = 1
      self.n_loc = 3
      self.n_outputs = self.n_objectness + self.n_loc

      with tf.variable_scope('RoarNet_3D_RPN') as roarnet_3d_rpn:
        self.preds = pointnet(self.inputs,
                      num_outputs = self.n_outputs,
                      is_training=self.is_training,
                      bn_decay=self.bn_decay,
                      dropout_keep_prob=0.7)

      self.idx_objectness = 0
      self.idx_loc = self.idx_objectness + self.n_objectness
      self.idx_rot = self.idx_loc + self.n_loc

      self.pred_objectness = self._infer_objectness(self.preds[:, 0:self.n_objectness])
      self.pred_locs = self._infer_location(self.preds[:, self.idx_loc:self.idx_loc + self.n_loc])
      self.pred_rots = self._infer_rotation(self.preds[:, self.idx_rot:])

    def _infer_objectness(self, pred_objs):
      p_objectness = tf.sigmoid(pred_objs)
      return p_objectness

    def _infer_location(self, loc_pred):
      x = loc_pred[:, 0]
      x = 2.0 * (tf.sigmoid(x) - 0.5) * self.cfg.CENTER_PERTURB
      y = loc_pred[:, 1]
      y = 2.0 * (tf.sigmoid(y) - 0.5) * self.cfg.CENTER_PERTURB
      z = loc_pred[:, 2]
      z = 2.0 * (tf.sigmoid(z) - 0.5) * self.cfg.CENTER_PERTURB_Z
      pred_locs = tf.stack([x, y, z], axis=-1)
      return pred_locs

    def losses(self, target_objs, target_locs, scope=None, gl = 0):
        pred_objectness = self.preds[:, 0]
        pred_locs = self.preds[:, self.idx_loc:self.idx_loc+self.n_loc]
        small_addon_for_BCE = 1e-6

        with tf.name_scope('losses'):          
          with tf.name_scope('objectness'):
            pred_objectness = tf.sigmoid(pred_objectness)

            self.pred_objectness = pred_objectness
            self.targ_objectness = target_objs

            loss_objectness = -target_objs * tf.log(pred_objectness + small_addon_for_BCE)\
                             -(1.0 - target_objs) * tf.log(1.0 - pred_objectness + small_addon_for_BCE) 
            loss_objectness = tf.reduce_mean(loss_objectness) * 10.0
            tf.losses.add_loss(loss_objectness)

          with tf.name_scope('loc_x'):
            pred_x = (2.0 * tf.sigmoid(pred_locs[:, 0])) - 1.0
            loss_loc = smooth_l1(pred_x, target_locs[:,0], sigma=3.0)
            loss_loc = target_objs * loss_loc
            loss_loc = tf.reduce_mean(loss_loc)
            tf.losses.add_loss(loss_loc)

          with tf.name_scope('loc_y'):
            pred_y = (2.0 * tf.sigmoid(pred_locs[:, 1])) - 1.0
            loss_loc = smooth_l1(pred_y, target_locs[:,1], sigma=3.0)
            loss_loc = target_objs * loss_loc
            loss_loc = tf.reduce_mean(loss_loc)
            tf.losses.add_loss(loss_loc)

          with tf.name_scope('loc_z'):
            pred_y = (2.0 * tf.sigmoid(pred_locs[:, 2])) - 1.0
            loss_loc = smooth_l1(pred_y, target_locs[:,2], sigma=3.0)
            loss_loc = target_objs * loss_loc
            loss_loc = tf.reduce_mean(loss_loc)
            tf.losses.add_loss(loss_loc)

def smooth_l1(deltas, targets, sigma=3.0):
    sigma2 = sigma * sigma
    diffs  = tf.subtract(deltas, targets)
    smooth_l1_signs = tf.cast(tf.less(tf.abs(diffs), 1.0 / sigma2), tf.float32)
    smooth_l1_option1 = tf.multiply(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = tf.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = tf.multiply(smooth_l1_option1, smooth_l1_signs) + tf.multiply(smooth_l1_option2, 1-smooth_l1_signs)
    smooth_l1 = smooth_l1_add
    
    return smooth_l1