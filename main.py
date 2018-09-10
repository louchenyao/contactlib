#! /usr/bin/env python3

from __future__ import print_function, absolute_import

from src.search import Searcher
from src.encoder import Encoder

import sys
import time

def main():
    pdb_fn = sys.argv[1]
    pdb_id = sys.argv[2]
    result_fn = sys.argv[3]

    e = Encoder()
    s = Searcher()

    encode_start = time.time()
    e.encode(pdb_fn, pdb_id, "test.dnn")
    encode_end = time.time()    
    
    search_start = time.time()
    s.search("test.dnn", result_fn)
    search_end = time.time()

    print("encode\t%.3f s" % (encode_end - encode_start,) )
    print("search\t%.3f s" % (search_end - search_start,) )


if __name__ == '__main__':
    main()