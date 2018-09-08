#!/bin/bash

rm -f test.*

blastpgp -i cullpdb25/aa/3aa0a.fa -d database/cullpdb2 -o test.log
# if high-quality hits found, no need to continue

time src/encoder.py cullpdb25/aa/3aa0a.pdb 3aa0a test.dnn
ls test.dnn >test.lst
time src/search test.lst database/contactlib-l4-g0-c2-d7.db 7 12 >test.log

