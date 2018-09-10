#! /usr/bin/env python3

from __future__ import print_function, absolute_import

from src.search import Searcher

import sys

def main():
    target_fn = sys.argv[1]
    db_fn = sys.argv[2]
    cutoff = int(sys.argv[3])
    result_fn = sys.argv[4]

    s = Searcher()
    s.search(target_fn, result_fn)


if __name__ == '__main__':
    main()