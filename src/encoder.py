#!/usr/bin/env python2

import os
import sys
import pickle
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.contrib.slim as slim

from common import *
from sklearn.metrics import *

from glob import glob
from struct import pack
from itertools import combinations
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
from tensorflow.python.client import device_lib



os.environ["CUDA_VISIBLE_DEVICES"] = "0" # used gpu
clone_id = "1"
dimin, dimout = 28, 7

np.random.seed(20180825)


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

def val2idx(value):
  vmin, vmax, nslot = -8.0, 8.0, 160
  index = ((value - vmin) / (vmax - vmin) * nslot + 0.5).astype(np.int32)
  index = np.minimum(np.maximum(index, 0), nslot)
  return index

def saveCode(value, pdbid, seqnum, ofn):
  fraglen = 4

  with open(ofn, "wb") as f:
    index = val2idx(value)
    for i in range(len(value)):
      header = "%s:%d:%d:%d" % (pdbid, seqnum[i, 0], seqnum[i, fraglen], fraglen)
      f.write(pack("%ds" % len(header), header))
      f.write(pack("s", "\t"))
      f.write(pack("fi" * dimout, *np.stack([value[i], index[i]], axis=1).flatten()))
      f.write(pack("s", "\n"))

# Define model
x0 = tf.placeholder(tf.float32, [None, dimin*2], name="x0")
x1 = tf.placeholder(tf.float32, [None, dimin*2], name="x1")
y = tf.placeholder(tf.float32, [None, 1], name="y")
w = tf.placeholder(tf.float32, [None, 1], name="w")
training = tf.placeholder_with_default(False, [], name="training")
with tf.variable_scope("clone%s" % clone_id): # load clone1 to model
  model = buildModel(x0, x1, y, w, dimout, 4, 512, training=training)

# Load model
variables = slim.get_variables_to_restore()
#print variables
sess = tf.Session() 
saver = tf.train.Saver(variables)
saver.restore(sess, "model/encoder")

with open("model/pca-coef.pickle", "rb") as f:
  _ = pickle.load(f)
  _ = pickle.load(f)
  _ = pickle.load(f)
  _ = pickle.load(f)
  _ = pickle.load(f)
  _ = pickle.load(f)
  coef = pickle.load(f)
  pca = pickle.load(f)

sslst = {}
with open("model/filter33.lst", "r") as f:
  for line in f:
    sslst[line.strip()] = 1

start = time.time()

pdbfn = sys.argv[1]
pdbid = sys.argv[2]
ofn = sys.argv[3]

pdb_start = time.time()
dist, coord, res, ss, idx, _, _ = loadPDB(pdbfn, fraglen=4, mingap=0, mincont=2, maxdist=16.0)
pdb_end = time.time()
index = np.array(filter(ss, sslst))
if np.any(index):
  idx = idx[index]
  dist = dist[index]
  distpca = pca.transform(dist)
  data = np.concatenate([dist, distpca], axis=-1)
  nn_start = time.time()
  distdnn = sess.run(model['code'], feed_dict={training: False, x0: data})
  nn_end = time.time()
  saveCode(distdnn, pdbid, idx, ofn)

end = time.time()

print "#done!!!"
print "# ", end - start
print "#pdb ", pdb_end - pdb_start
print "#nn ", nn_end - nn_start
