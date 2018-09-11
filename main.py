#! /usr/bin/env python3

from __future__ import print_function, absolute_import

import sys

from contactlib import Searcher, Encoder
from contactlib.data_manger import asset_path

e = Encoder()
s = Searcher()

e.encode(asset_path("3aa0a.pdb"), "3aa0a.fingerprint")
s.search("3aa0a.fingerprint", "result.txt")
