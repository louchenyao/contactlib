from __future__ import absolute_import, print_function

import os
import pickle
import string
import sys
from struct import pack

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from contactlib.common import *
from contactlib.data_manger import asset_path
from contactlib.model import buildModel
from contactlib.timer import TimeIt


def val2idx(value):
  vmin, vmax, nslot = -8.0, 8.0, 160
  index = ((value - vmin) / (vmax - vmin) * nslot + 0.5).astype(np.int32)
  index = np.minimum(np.maximum(index, 0), nslot)
  return index

def saveCode(value, pdb_id, seqnum, output_fn):
  fraglen = 4

  with open(output_fn, "wb") as f:
    index = val2idx(value)
    for i in range(len(value)):
      header = "%s:%d:%d:%d" % (pdb_id, seqnum[i, 0], seqnum[i, fraglen], fraglen)
      f.write(pack("%ds" % len(header), header.encode()))
      f.write(pack("s", b"\t"))
      for j in range(len(index[i])):
          f.write(pack("fi", value[i][j], index[i][j]))
      f.write(pack("s", b"\n"))

np.random.seed(20180825)
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # use gpu 0
clone_id = "1"
dimin, dimout = 28, 7

graph = tf.Graph()

with graph.as_default():
    x0 = tf.placeholder(tf.float32, [None, dimin*2], name="x0")
    x1 = tf.placeholder(tf.float32, [None, dimin*2], name="x1")
    y = tf.placeholder(tf.float32, [None, 1], name="y")
    w = tf.placeholder(tf.float32, [None, 1], name="w")
    training = tf.placeholder_with_default(False, [], name="training")
    with tf.variable_scope("clone%s" % clone_id): # load clone1 to model
        model = buildModel(x0, x1, y, w, dimout, 4, 512, training=training)


class Encoder(object):
    def __init__(self, tf_sess_config=None):
        with graph.as_default():
            asset_path("encoder.index")
            asset_path("encoder.meta")
            p = asset_path("encoder.data-00000-of-00001")
            p = os.path.join(os.path.dirname(p), "encoder")

            self.sess = tf.Session(config=tf_sess_config)
            variables = slim.get_variables_to_restore()
            saver = tf.train.Saver(variables)
            saver.restore(self.sess, p)

        with open(asset_path("pca-coef.pickle"), "rb") as f:
        # encoding='latin1' fixs the incompatibility of numpy between Python 2 and 3
        # see more: https://stackoverflow.com/a/41366785
            def load(f):
                if sys.version_info.major == 2:
                    return pickle.load(f)
                else:
                    return pickle.load(f, encoding='latin1')
            _ = load(f)
            _ = load(f)
            _ = load(f)
            _ = load(f)
            _ = load(f)
            _ = load(f)
            self.coef = load(f)
            self.pca = load(f)

        self.sslst = {}
        with open(asset_path("filter33.lst"), "r") as f:
            for line in f:
                self.sslst[line.strip()] = 1

    def __del__(self):
        self.sess.close()

    def encode(self, pdb_fn, output_fn, pdb_id=None):
        if not pdb_id:
            pdb_id = os.path.basename(pdb_fn).split(".")[0]
            pdb_id = "".join(c for c in pdb_id if c in string.ascii_letters or c in string.digits)
            if not pdb_id:
                pdb_id = "target"
        with TimeIt("loadPDB"):
            dist, coord, res, ss, idx, _, _ = loadPDB(pdb_fn, fraglen=4, mingap=0, mincont=2, maxdist=16.0)
        index = np.array(filter(ss, self.sslst))
        if np.any(index):
            idx = idx[index]
            dist = dist[index]
            distpca = self.pca.transform(dist)
            data = np.concatenate([dist, distpca], axis=-1)
            with TimeIt("tf_run"):
                distdnn = self.sess.run(model['code'], feed_dict={training: False, x0: data})
            saveCode(distdnn, pdb_id, idx, output_fn)
        else:
            raise Exception("Not exists any indexable contact groups!")
