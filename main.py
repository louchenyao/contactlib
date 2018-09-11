#! /usr/bin/env python

from __future__ import print_function, absolute_import

import sys

from contactlib import Searcher, Encoder
from contactlib.data_manger import asset_path

e = Encoder()
s = Searcher()

# asset_path returns the path of "3aa0a.pdb", it is an example
e.encode(asset_path("3aa0a.pdb"), "3aa0a.fingerprint")

# the result be stored in "result.txt"
s.search("3aa0a.fingerprint", "result.txt")
