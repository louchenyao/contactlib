from __future__ import print_function, absolute_import

from contactlib.data_manger import asset_path

import os
import platform
from ctypes import CDLL, c_uint64

from contactlib.data_manger import asset_path

class Searcher(object):
    def __init__(self, db_fn=None, cutoff=12):
        """ Init the Searcher Instance. If db_fn is not indicated, it automatically download
        contactlib-l4-g0-c2-d7.db as default database.
        """
        if db_fn is None:
            db_fn = asset_path("contactlib-l4-g0-c2-d7.db")
        
        if os.getenv("ENV") == "Development":
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            lib_fn = os.path.join(cur_dir, "libsearch.so")
        else:
            if platform.system() == "Darwin":
                libsearch = "libsearch-2.1-macOS.so"
            elif platform.system() == "Linux":
                libsearch = "libsearch-2.1-linux-amd64.so"
            else:
                raise Exception("Unsupported platform! Please try it under Linux.")
            lib_fn = asset_path(libsearch)

        self.lib = CDLL(lib_fn)
        self.lib.newDB.restype = c_uint64 # void * is 64 bit, but the defualt type is unint32

        self.db = self.lib.newDB(db_fn.encode(), cutoff)
    
    def __del__(self):
        if hasattr(self, "db"):
            self.lib.deleteDB(c_uint64(self.db))

    def search(self, target_fn, result_fn):
        self.lib.search(c_uint64(self.db), target_fn.encode(), result_fn.encode())
