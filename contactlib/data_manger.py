#! /usr/bin/python3

import hashlib
import os
import random
import requests
import shutil
import stat
import string

from contactlib import logger

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

class Assest(object):
    def __init__(self, url, executable=False, md5=None):
        self.url = url
        self.executable = executable
        self.md5 = md5

ASSETS = {
    "contactlib-l4-g0-c2-d7.db": Assest("https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/contactlib-l4-g0-c2-d7.db", md5="1fa6ab39a3869e9b4f4969ae192c940b"),

    "encoder.data-00000-of-00001": Assest("https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/encoder.data-00000-of-00001", md5="6262b90c210c148985cbe5adf8aee440"),
    "encoder.index": Assest("https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/encoder.index", md5="b93d2df4196facec2dd4aa96c159dc0a"),
    "encoder.meta": Assest("https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/encoder.meta", md5="c9f415d5fd6c4cc76a1a890a1b82483d"),

    "filter33.lst": Assest("https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/filter33.lst", md5="ade14361f61c8038398c9e69318c7a7a"),
    "pca-coef.pickle": Assest("https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/pca-coef.pickle", md5="3ef8ac406edfead3ff2e5d01b211e611"),

    "dssp-2.0.4-linux-amd64": Assest("https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/dssp-2.0.4-linux-amd64", executable=True, md5="b8dcc06305a64ab4d0265f03de7bb718"),
    "dssp-2.0.4-macOS": Assest("https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/dssp-2.0.4-macOS", executable=True, md5="3a02711e103a669cdcec6f63d498cd4f"),
    "libsearch-2.1-macOS.so": Assest("https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/libsearch-2.1-macOS.so", md5="9a5b5ad990b8ba8f5f62d9edc70d9a5e"),
    "libsearch-2.1-linux-amd64.so": Assest("https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/libsearch-2.1-linux-amd64.so", md5="5a587342e56112a1260d6c1bda7ab274"),

    "3aa0a.pdb": Assest("https://pppublic.oss-cn-beijing.aliyuncs.com/tmp-server/test_assests/3aa0a.pdb", md5="387265f649aad08fadb0591647bdcbc1")
}

# copyright: https://stackoverflow.com/a/2030081
def randomword(len=10):
   letters = string.ascii_lowercase
   return ''.join(random.choice(letters) for i in range(len))

# copyright: https://stackoverflow.com/a/3431838
def calc_md5(p):
    hash_md5 = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

# copyright: https://stackoverflow.com/a/16696317
def download(p, url, md5=None):
    tmp_p = os.path.join(os.path.join(TMP), randomword(10))
    r = requests.get(url, stream=True)
    with open(tmp_p, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
    if md5 != None:
        if calc_md5(tmp_p) != md5:
            raise Exception("%s md5 mismatch" % p)
    shutil.move(tmp_p, p)

def asset_path(name):
    if name not in ASSETS:
        raise KeyError()
    a = ASSETS[name]

    p = os.path.join(DATA, name)
    if not os.path.isfile(p):
        logger.info("%s does not exist, downloading it..." % name)
        download(p, a.url, a.md5)
    if a.executable:
        st = os.stat(p)
        os.chmod(p, st.st_mode | stat.S_IEXEC)
    
    return p