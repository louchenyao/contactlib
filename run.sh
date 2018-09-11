#!/bin/bash

export PYTHONPATH=$(pwd)

cd src/search
make
cd ../..

rm -f test.* *.fingerprint result.txt

#blastpgp -i cullpdb25/aa/3aa0a.fa -d database/cullpdb2 -o test.log
# if high-quality hits found, no need to continue

time python ./main.py
diff result.txt test_log_std
