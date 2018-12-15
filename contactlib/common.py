import os
import platform
import sys
import numpy as np

from Bio.PDB.DSSP import *
from Bio.PDB.PDBParser import *
from Bio.PDB.Polypeptide import *
from scipy.spatial.distance import *

from contactlib.data_manger import asset_path
from contactlib.timer import TimeIt


def convertPDB(pdbfn):
  if platform.system() == "Darwin":
    cmd = asset_path("dssp-2.0.4-macOS")
  elif platform.system() == "Linux":
    cmd = asset_path("dssp-2.0.4-linux-amd64")
  else:
    raise Exception("Unsupported platform! Please try it under Linux.")

  pdbid = os.path.basename(pdbfn).replace(".pdb", "").upper()
  model = PDBParser(PERMISSIVE=1).get_structure(pdbid, pdbfn)[0]

  dsspfn = pdbfn.replace(".pdb", ".dssp")
  subprocess.check_call([cmd, '-i', pdbfn, '-o', dsspfn])

  dssp, keys = make_dssp_dict(dsspfn)
  idx, res, ss = [], [], []
  for k in keys:
    try:
      i, r, s, c = k[1][1], dssp[k][0], dssp[k][1], model[k[0]][k[1]]["CA"].get_coord()
      idx.append(i)
      res.append(r)
      ss.append(s)
    except KeyError: pass

  fafn = pdbfn.replace(".pdb", ".fa")
  with open(fafn, "w") as f:
    f.write(">%s\n" % pdbid)
    tmp = "".join(res)
    while tmp:
      if len(tmp) > 120:
        f.write("%s\n" % tmp[:120])
        tmp = tmp[120:]
      else:
        f.write("%s\n" % tmp)
        tmp = ""

def loadPDB(pdbfn, fraglen=4, mingap=0, mincont=2, maxdist=16.0):
  pdbid = os.path.basename(pdbfn).replace(".pdb", "").upper()
  model = PDBParser(PERMISSIVE=1).get_structure(pdbid, pdbfn)[0]

  dsspfn = pdbfn.replace(".pdb", ".dssp")
  if not os.path.isfile(dsspfn):
    convertPDB(pdbfn)

  dssp, keys = make_dssp_dict(dsspfn)

  idx, res, ss, coord = [], [], [], []
  for k in keys:
    try:
      i, r, s, c = k[1][1], dssp[k][0], dssp[k][1], model[k[0]][k[1]]["CA"].get_coord()
      idx.append(i)
      res.append(r)
      ss.append(s)
      coord.append(c)
    except KeyError: pass
  idx = np.array(idx)
  res = np.array(res)
  ss = np.array(ss)
  coord = np.array(coord)
  dist = squareform(pdist(coord))

  with TimeIt("gen data"):
    data = []
    possible_frag_j_idx = []
    possible_frag_i_idx = []
    for j in range(len(dist)-fraglen + 1):
      if not np.any(dist[j:j+fraglen, j:j+fraglen] >= maxdist):
        if(j >= fraglen+mingap):
          possible_frag_j_idx.append(j)
        possible_frag_i_idx.append(j)


    for i in possible_frag_i_idx:
      for j in possible_frag_j_idx:
        if i >= j-fraglen-mingap + 1:
          continue

        if np.any(dist[i:i+fraglen, j:j+fraglen] >= maxdist): continue
        if np.sum(dist[i:i+fraglen, j:j+fraglen] <= 8.0) < mincont: continue
        
        k = np.concatenate((idx[i:i+fraglen], idx[j:j+fraglen]), axis=0)
        r = np.concatenate((res[i:i+fraglen], res[j:j+fraglen]), axis=0)
        s = np.concatenate((ss[i:i+fraglen], ss[j:j+fraglen]), axis=0)
        c = np.concatenate((coord[i:i+fraglen], coord[j:j+fraglen]), axis=0)
        d0 = np.concatenate((dist[i:i+fraglen, i:i+fraglen], dist[i:i+fraglen, j:j+fraglen]), axis=1)
        d1 = np.concatenate((dist[j:j+fraglen, i:i+fraglen], dist[j:j+fraglen, j:j+fraglen]), axis=1)
        d = squareform(np.concatenate((d0, d1), axis=0))

        data.append([d, c, r, s, k])
  data = np.array(data)

  if len(data) > 0: return np.stack(data[:, 0]), np.stack(data[:, 1]), np.stack(data[:, 2]), np.stack(data[:, 3]), np.stack(data[:, 4]), len(res), len(data)
  else: return None, None, None, None, None, len(res), len(data)

def encode(ss):
  a, b, c = 0, 0, 0
  for i in ss:
    if i in "H": a += 1
    elif i in "E": b += 1
    else: c += 1
  if a >= 2 and a+c >=3: return "A%d" % a
  elif b >= 2 and b+c >=3: return "B%d" % b
  elif c >= 3: return "C%d" % c
  else: return "D0";

def filter(ss, lst):
  index = []
  fraglen = int(ss.shape[-1] / 2) # In Python3, fraglen is float due to division.
  for i in ss:
    code0 = encode(i[:fraglen])
    code1 = encode(i[-fraglen:])
    code = "".join(sorted([code0, code1]))
    index.append(code in lst)
  return index

