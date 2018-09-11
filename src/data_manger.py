#! /usr/bin/python3

from src import logger

import os
import requests
import random
import string
import shutil

HOME = os.getenv("HOME")
DATA = os.path.join(HOME, ".cache", "contactlib")
TMP = os.path.join(DATA, "tmp")

# For Python2 compatibility
def mkdirp(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdirp(DATA)
shutil.rmtree(TMP, ignore_errors=True)
mkdirp(TMP)

ASSETS = {
    "contactlib-l4-g0-c2-d7.db": "https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/contactlib-l4-g0-c2-d7.db",

    "encoder.data-00000-of-00001": "https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/encoder.data-00000-of-00001",
    "encoder.index": "https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/encoder.index",
    "encoder.meta": "https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/encoder.meta",

    "filter33.lst": "https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/filter33.lst",
    "pca-coef.pickle": "https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/pca-coef.pickle",

    "dssp": "https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/dssp-2.0.4-linux-amd64",

    "3aa0a.pdb": "https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/test_assests/3aa0a.pdb"
}

# copyright: https://stackoverflow.com/a/2030081
def randomword(len=10):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(len))

# TODO: check integrity
# copyright: https://stackoverflow.com/a/16696317
def download(p, url):
    tmp_p = os.path.join(os.path.join(TMP), randomword(10))
    r = requests.get(url, stream=True)
    with open(tmp_p, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    shutil.move(tmp_p, p)

def asset_path(name, auto_download=True):
    if name not in ASSETS:
        raise KeyError()

    p = os.path.join(DATA, name)
    if os.path.isfile(p):
        return p
    elif auto_download:
        logger.info("%s does not exist, downloading it..." % name)
        download(p, ASSETS[name])
        return p
    else:
        return None
