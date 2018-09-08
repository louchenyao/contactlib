#!/usr/bin/env python2

import os
import sys
import psutil
import pickle
import threading
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from common import *
from sklearn.metrics import *

from tqdm import tqdm
from glob import glob
from struct import pack
from itertools import combinations
from sklearn.decomposition import PCA
from scipy.spatial.distance import squareform
from tensorflow.python.client import device_lib
from pyRMSD.RMSDCalculator import RMSDCalculator
from tensorflow.python.tools import inspect_checkpoint as chkp


dimin, dimout = 28, 7

dev = [d.name for d in device_lib.list_local_devices() if d.device_type == "GPU"]
clonesize, numepochs, batchsize, testsize, eps = len(dev), 1000, 100, 10000, 0.01
ifn = "build/r4-g0-c2-d16-rnd2m.npz"
tfn = "temp/encode-test%d.bin" % testsize
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

  with tf.device("/cpu:0"):
    tf.summary.histogram("x_code", x_code)

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

  with tf.device("/cpu:0"):
    with tf.name_scope("gradients"):
      for g, v in grad:
        mean = tf.reduce_mean(g)
        tf.summary.scalar("%s-mean" % v.name, mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(g - mean)))
        tf.summary.scalar("%s-stdev" % v.name, stddev)
        tf.summary.histogram(v.name, g)
    tf.summary.scalar("loss", loss_avg)

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

def buildCode(pdbfn, pca, sess, sem_cpu, sem_gpu):
  dist, coord, res, ss, idx, _, _ = loadPDB(pdbfn, fraglen=4, mingap=0, mincont=2, maxdist=16.0)
  index = np.array(filter(ss, sslst))
  if not np.any(index):
    sem_cpu.release()
    return

  idx = idx[index]
  dist = dist[index]
  distpca = pca.transform(dist)
  data = np.concatenate([dist, distpca], axis=-1)
  sem_gpu.acquire()
  distdnn = sess.run(model['code'], feed_dict={training: False, x0: data})
  sem_gpu.release()

  pdbid = os.path.basename(pdbfn).replace(".pdb", "")
  saveCode(distdnn, pdbid, idx, pdbfn.replace(".pdb", ".dnnb"))
  saveCode(distpca[:, :dimout]*coef, pdbid, idx, pdbfn.replace(".pdb", ".pcab"))

  sem_cpu.release()

def testset(dist, coord, pca=None):
  try:
    with open(tfn, "rb") as f:
      td0 = pickle.load(f)
      td1 = pickle.load(f)
      trmsd = pickle.load(f)
      tw = pickle.load(f)
      tpred = pickle.load(f)
      tmax = pickle.load(f)
      tcoef = pickle.load(f)
      _ = pickle.load(f)
    print "#testset loaded"
  except:
    with open(tfn, "wb") as f:
      index = np.random.choice(len(dist), testsize, replace=False)
      d, c = dist[index], coord[index]
      rmsd = np.maximum(np.array(RMSDCalculator("QCP_OMP_CALCULATOR", c).pairwiseRMSDMatrix(), dtype=np.float32), eps).reshape((-1, 1))
      index = np.random.choice(len(rmsd), int(testsize*np.log(testsize)), replace=False)
      comb = np.array(list(combinations(range(testsize), 2)))[index]
      td0, td1, trmsd, tw = d[comb[:, 0]], d[comb[:, 1]], rmsd[index], np.ones((len(index), 1), dtype=np.float32)

      tpred = np.sqrt(np.mean(np.square(td0[:, dimin:dimin+dimout] - td1[:, dimin:dimin+dimout]), axis=-1, keepdims=True))
      tcoef = np.mean(trmsd / np.maximum(tpred, eps))
      tpred = tpred * tcoef
      tmax = np.max(np.abs(td0[:, dimin:dimin+dimout] - td1[:, dimin:dimin+dimout]), axis=-1, keepdims=True) * tcoef

      for i in (td0, td1, trmsd, tw, tpred, tmax, tcoef): pickle.dump(i, f)
      if pca != None: pickle.dump(pca, f)
    print "#testset processed"
  return td0, td1, trmsd, tw, tpred, tmax, tcoef

def stat(value, pred, bench, ofp):
  data = pd.DataFrame(zip(value.flatten(), pred.flatten()), columns=["RMSD", "MSE"])
  sns.set(font_scale=1.25, style="white")

  for i in (6, 2):
    data = data[data["RMSD"] <= i]
    data = data[data["MSE"] <= i]
    sns.jointplot(x="RMSD", y="MSE", data=data, kind="hex", color="red", xlim=[0, i], ylim=[0, i])
    plt.savefig("%s-heat%d.png" % (ofp, i), dpi=300, bbox_inches='tight')
    plt.close()

  if bench is not None:
    dpred = pred - value
    dbench = bench - value
    epred = np.abs(dpred)
    ebench = np.abs(dbench)
    df = np.concatenate([dpred, epred, dbench, ebench], axis=-1)[value.flatten() <= 2.0]
    print pd.DataFrame(df, columns=['', 'DNN', '', 'PCA']).describe(percentiles=[.01, .05, .1, .25, .5, .75, .9, .95, .99])

    out = ["cutoff\t", "precision", "recall\t", "f1-score"]
    for cutoff in np.arange(0.6, 1.41, 0.2):
      bvalue = value <= cutoff
      bpred = pred <= cutoff
      bbench = bench <= cutoff
      out[0] += "\t\t%.1f" % cutoff
      out[1] += "\t%.3f\t%.3f" % (precision_score(bvalue, bpred), precision_score(bvalue, bbench))
      out[2] += "\t%.3f\t%.3f" % (recall_score(bvalue, bpred), recall_score(bvalue, bbench))
      out[3] += "\t%.3f\t%.3f" % (f1_score(bvalue, bpred), f1_score(bvalue, bbench))
    for o in out: print o


models = []
x0 = tf.placeholder(tf.float32, [None, dimin*2], name="x0")
x1 = tf.placeholder(tf.float32, [None, dimin*2], name="x1")
y = tf.placeholder(tf.float32, [None, 1], name="y")
w = tf.placeholder(tf.float32, [None, 1], name="w")
training = tf.placeholder_with_default(False, [], name="training")
for i in range(clonesize):
  with tf.variable_scope("clone%d" % i), tf.device("/gpu:%d" % (i % len(dev))):
    models.append(buildModel(x0, x1, y, w, dimout, 4, 512, training=training))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
saver = tf.train.Saver(max_to_keep=numepochs-10)
tblog = tf.summary.FileWriter("./tblog", sess.graph)
sess.run(tf.global_variables_initializer())
train_ops = [m['train'] for m in models]
sum_ops = tf.summary.merge_all()
print "#model built"


if sys.argv[1] == 'train':
  epoch = int(sys.argv[2])

  data = np.load(ifn)
  dist = data['dist'].astype(np.float32)
  coord = data['coord'].astype(np.float64)
  datasize = len(dist)
  data = None
  print "#dataset loaded"

  pca = PCA(n_components=dimin).fit(dist)
  dist = np.concatenate((dist, pca.transform(dist)), axis=-1)
  td0, td1, trmsd, tw, tpred, tmax, tcoef = testset(dist, coord, pca)
  stat(trmsd, tpred, None, "plot/mse")
  print "#dataset preprocessed"

  try:
    saver.restore(sess, "saver/encoder-%d" % epoch)
    print "#resumed epoch %d ..." % epoch
    for model in models:
      print "#model:", model['name']
      l, p, m = sess.run([model['loss'], model['y_pred'], model['y_max']], feed_dict={training: False, x0: td0, x1: td1, y: trmsd, w: tw})
      print "#loss:", l
      stat(trmsd, p, tpred, "plot/%s-pred%d" % (model['name'], epoch))
      stat(trmsd, m, tmax, "plot/%s-max%d" % (model['name'], epoch))
  except:
    epoch = -1
  sys.exit()

  batchindex = np.array(list(combinations(range(batchsize), 2)))
  batchweight = np.maximum(-np.log(np.arange(1, len(batchindex)+1, dtype=np.float32) / (len(batchindex)+1)), eps).reshape([-1, 1])
  for epoch in range(epoch+1, numepochs):
    print "#training epoch %d ..." % epoch
    index = np.random.permutation(datasize)

    for batch in tqdm(range(0, datasize, batchsize)):
      d0 = dist[index[batch : batch+batchsize]][batchindex[:, 0]]
      d1 = dist[index[batch : batch+batchsize]][batchindex[:, 1]]
      c = coord[index[batch : batch+batchsize]]
      cal = RMSDCalculator("QCP_OMP_CALCULATOR", c)
      rmsd = np.maximum(np.array(cal.pairwiseRMSDMatrix(), dtype=np.float32), eps).reshape((-1, 1))
      weight = np.empty(rmsd.shape, dtype=np.float32)
      weight[np.argsort(rmsd.flatten())] = batchweight

      sess.run(train_ops, feed_dict={training: True, x0: d0, x1: d1, y: rmsd, w: weight})
    saver.save(sess, "./saver/encoder", global_step=epoch)

    for model in models:
      print "#model:", model['name']
      l, p, m = sess.run([model['loss'], model['y_pred'], model['y_max']], feed_dict={training: False, x0: td0, x1: td1, y: trmsd, w: tw})
      print "#loss:", l
      stat(trmsd, p, tpred, "plot/%s-pred%d" % (model['name'], epoch))
      stat(trmsd, m, tmax, "plot/%s-maxe%d" % (model['name'], epoch))
    tblog.add_summary(sess.run(sum_ops, feed_dict={training: False, x0: td0, x1: td1, y: trmsd, w: tw}), epoch)
  tblog.close()
elif sys.argv[1] == 'build':
  epoch, clone = 109, 1
  saver.restore(sess, "saver/encoder-%d" % epoch)
  model = models[clone]

  with open(tfn, "rb") as f:
    _ = pickle.load(f)
    _ = pickle.load(f)
    _ = pickle.load(f)
    _ = pickle.load(f)
    _ = pickle.load(f)
    _ = pickle.load(f)
    coef = pickle.load(f)
    pca = pickle.load(f)

  sslst = {}
  ssfn="align.all8k/greedy33.lst"
  with open(ssfn, 'r') as f:
    for line in f:
      sslst[line.strip()] = 1

  sem_cpu = threading.Semaphore(4)
  sem_gpu = threading.Semaphore(1)
  for pdbfn in glob("pdb25/*/*.pdb"):
    sem_cpu.acquire()
    print "#working on %s ..." % pdbfn
    threading.Thread(target=buildCode, args=(pdbfn, pca, sess, sem_cpu, sem_gpu)).start()
  main_thread = threading.currentThread()
  for t in threading.enumerate():
    if t is not main_thread:
      t.join()

