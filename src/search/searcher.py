from __future__ import print_function

import os
from ctypes import CDLL, c_uint64

class Searcher(object):
    def __init__(self, db_fn, cutoff):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        lib_fn = os.path.join(cur_dir, "libsearch.so")

        self.lib = CDLL(lib_fn)
        self.lib.newDB.restype = c_uint64 # void * is 64 bit, but the defualt type is unint32

        self.db = self.lib.newDB(db_fn.encode(), cutoff)
    
    def __del__(self):
        self.lib.deleteDB(c_uint64(self.db))

    def search(self, target_fn, result_fn):
        self.lib.search(c_uint64(self.db), target_fn.encode(), result_fn.encode())
