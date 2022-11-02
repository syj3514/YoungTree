from rur import uri, uhmi
import numpy as np
import sys
import os
import time
import gc
from collections.abc import Iterable
import inspect
import psutil
import logging
import copy
import traceback

if not "/home/jeon/YoungTree" in sys.path:
    sys.path.append("home/jeon/YoungTree")
from tree_utool import *

mode = input("mode=? ")

repo, rurmode, dp = mode2repo(mode)
nout = load_nout(mode=mode, galaxy=True)
nstep = load_nstep(mode=mode, galaxy=True, nout=nout)
iout = nout[0]

uri.timer.verbose=0
snap = uri.RamsesSnapshot(repo, iout, mode=rurmode)
gals = uhmi.HaloMaker.load(snap, galaxy=True, double_precision=dp)


path = f"/home/jeon/YoungTree/result/{mode}"
yess = ["Y", "y", "Yes", "yes"]
yes = input(f"Results files are in here? `{path}` [Y/N]")
if yes in yess:
    pass
else:
    path = input("Write absolute path: ")
file_list = os.listdir(path)
file_list = [file for file in file_list if (file.startswith(f"Progenitor_Branch") & file.endswith(".pickle"))]

if len(gals) != len(file_list):
    raise ValueError(f"{len(file_list)} results are found, but you have {len(gals)} galaxies!")



dtype = [("rootid", int), ("rootiout", int), ("id", int), ("timestep", int), ("score", float), ("elapsed", float)]
print(f"New file dtype: {dtype}")
first = True
for gal in gals:
    readme, root, elap, branch, scores = pklload(f'/home/jeon/YoungTree/result/{mode}/Progenitor_Branch_ID{gal["id"]:05d}_iout{iout:05d}.pickle')

    outs = list( branch.keys() )
    ntree = len(outs)
    rootids = np.full(ntree, root['id'], dtype=int)
    rootiouts = np.full(ntree, root['timestep'], dtype=int)
    elapseds = np.full(ntree, elap, dtype=float)
    
    ids = np.array( [branch[ith]['id'] for ith in outs] )
    timesteps = np.array( [branch[ith]['timestep'] for ith in outs] )
    scores = np.array( [scores[ith] for ith in outs] )
    if first:
        result = np.rec.fromarrays((rootids, rootiouts, ids, timesteps, scores, elapseds), dtype=dtype)
        first = False
    else:
        result = np.hstack((result, np.rec.fromarrays((rootids, rootiouts, ids, timesteps, scores, elapseds), dtype=dtype)))

print(f"Final array has {result.shape} shape")


yes = input(f"This file will be saved in `YoungTree` directory of `{repo}`, ok? [Y/N]")
if yes in yess:
    pass
else:
    repo = input("Write new repo: ")

if not os.path.isdir(f"{repo}/YoungTree"):
    os.mkdir(f"{repo}/YoungTree")
readme = "README:\n    1) Root galaxy ID\n    2) Root galaxy iout\n    3) Galaxy ID\n    4) Galaxy iout\n    5) Score\n    6) Elapsed computing time"
pklsave((readme, result), f"{repo}/YoungTree/ytree.pickle")
print(f"`{f'{repo}/YoungTree/ytree.pickle'}` save done.")
print(readme)