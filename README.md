# ContactLib

Tool for searching proteins through structural fingerprints.

The method is based on [Learning Protein Structural Fingerprints under the Label-Free Supervision of Domain Knowledge](https://www.biorxiv.org/content/early/2018/09/03/407106).

If it makes any help for you, please cite:

~~~
@article {Min407106,
	author = {Min, Yaosen and Liu, Shang and Cui, Xuefeng},
	title = {Learning Protein Structural Fingerprints under the Label-Free Supervision of Domain Knowledge},
	year = {2018},
	doi = {10.1101/407106},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2018/09/03/407106},
	eprint = {https://www.biorxiv.org/content/early/2018/09/03/407106.full.pdf},
	journal = {bioRxiv}
}
~~~

## Install

~~~
pip install contactlib
~~~


## Usage

Try to run the following Python script:

~~~
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
~~~

(If it's your first time run ContactLib, it takes a while to download the necessary model and database.)

And you will find "result.txt" in your current directory, it looks like:

~~~
3aa0a	2a0ba	0.178717
3aa0a	3a02a	0.218567
3aa0a	3a07a	0.249448
3aa0a	3a09a	0.301373
3aa0a	3a0sa	0.345046
3aa0a	3a0yb	0.341316
3aa0a	4a02a	0.266509
3aa0a	4a0pa	0.354293
3aa0a	5a07b	0.293344
3aa0a	5a0lb	0.280329
3aa0a	5a0na	0.288656
3aa0a	5a0ra	0.211348
3aa0a	5a0ya	0.324709
...
~~~

### Explanation

`e.encode(asset_path("3aa0a.pdb"), "3aa0a.fingerprint")` encodes the pdb file to fingerprint by ContactLib-DNN.

`s.search("3aa0a.fingerprint", "result.txt")` searchs the proteins with similar struct in our database.