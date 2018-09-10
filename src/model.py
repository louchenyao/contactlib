#!/usr/bin/env python3

from __future__ import print_function

import tensorflow as tf

def buildModule(prevlayer, numlayers=2, numnodes=64, training=False, name=None, reuse=None):
  l = prevlayer
  for i in range(numlayers):
    tmp = None if name is None else "%s-d%d" % (name, i)
    l = tf.layers.dense(l, numnodes, activation=tf.nn.relu, name=tmp, reuse=reuse)
    tmp = None if name is None else "%s-bn%d" % (name, i)
    l = tf.layers.batch_normalization(l, training=training, name=tmp, reuse=reuse)
  return l

def buildEncoder(x, dimcode, numlayers=2, numnodes=64, training=False, name=None, reuse=None):
  tmp = None if name is None else "%s-m%d" % (name, 1)
  m1 = buildModule(x, numlayers, numnodes, training=training, name=tmp, reuse=reuse)
  tmp = None if name is None else "%s-out%d" % (name, 2)
  x_code = tf.layers.dense(m1, dimcode, activation=None, name=tmp, reuse=reuse)
  x_code = tf.maximum(tf.minimum(x_code, 8.0), -8.0)

  return x_code

def buildModel(x0, x1, y, w, dimcode, numlayers=2, numnodes=64, training=False, name="encode"):
  tmp = "%s-d%d-l%d-n%d" % (name, dimcode, numlayers, numnodes)
  code0 = buildEncoder(x0, dimcode, numlayers, numnodes, training=training, name=tmp, reuse=None)
  t0 = len(tf.global_variables())
  code1 = buildEncoder(x1, dimcode, numlayers, numnodes, training=training, name=tmp, reuse=True)
  t1 = len(tf.global_variables())
  assert t0 == t1

  y_pred = tf.sqrt(tf.reduce_mean(tf.square(code0 - code1), axis=1, keepdims=True))
  y_max = tf.reduce_max(tf.abs(code0 - code1), axis=1, keepdims=True)
  loss_sum = tf.reduce_sum(w * tf.square(tf.log(y_pred / y)))
  loss_avg = loss_sum / tf.reduce_sum(w)

  scope = tf.get_variable_scope().name
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=scope)
  with tf.control_dependencies(update_ops):
    opt = tf.train.AdamOptimizer()
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    grad = opt.compute_gradients(loss_sum, var_list=train_vars)
    train = opt.apply_gradients(grad)

  return {'name': tmp, 'code': code0, 'y_pred': y_pred, 'y_max': y_max, 'loss': loss_avg, 'train': train}
