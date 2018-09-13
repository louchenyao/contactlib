#! /usr/bin/env python

from __future__ import print_function, absolute_import

import sys

from contactlib import Searcher, Encoder
from contactlib.common import convertPDB
from contactlib.data_manger import asset_path

e = Encoder()
s = Searcher()

# generate FASTA and DSSP files, given a PDB file
convertPDB(asset_path("3aa0a.pdb"))

# build ContactLib fingerprint file
e.encode(asset_path("3aa0a.pdb"), "3aa0a.cl")

# scan ContactLib for homologous proteins, and store the similarity scores
s.search("3aa0a.cl", "output.txt")

