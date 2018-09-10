#!/bin/bash

export PYTHONPATH=$(pwd)

cd src/search
make
cd ../..

rm -f test.*

#blastpgp -i cullpdb25/aa/3aa0a.fa -d database/cullpdb2 -o test.log
# if high-quality hits found, no need to continue

time ./main.py cullpdb25/aa/3aa0a.pdb 3aa0a test.log
diff test.log test_log_std
