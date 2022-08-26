import gc
import psutil
import time
import numpy as np
from rur import uri, uhmi

from tree_utool import *
from tree_root import Treebase
import params as p

#########################################################
###############         Targets                ##########
#########################################################
# p.mode = input("\n>>> p.mode=? (hagn, yxxxxx, nh..)")
modenames = {"hagn": "Horizon-AGN", 
            "y01605": "YZiCS-01605",
            "y04466": "YZiCS-04466",
            "y05420": "YZiCS-05420",
            "y05427": "YZiCS-05427",
            "y06098": "YZiCS-06098",
            "y07206": "YZiCS-07206",
            "y10002": "YZiCS-10002",
            "y17891": "YZiCS-17891",
            "y24954": "YZiCS-24954",
            "y29172": "YZiCS-29172",
            "y29176": "YZiCS-29176",
            "y35663": "YZiCS-35663",
            "y36413": "YZiCS-36413",
            "y36415": "YZiCS-36415",
            "y39990": "YZiCS-39990",
            "y49096": "YZiCS-49096",
            "nh": "NewHorizon"}
if not p.mode in modenames.keys():
    raise ValueError(f"mode={p.mode} is not supported!")
modename = modenames[p.mode]
repo, rurmode = mode2repo(p.mode)

# ans = input("\n>>> Use galaxy? ")
galstr = "Halo"
galstrs = "Halos"
if p.galaxy:
    galstr = "Galaxy"
    galstrs = "Galaxies"

nout = load_nout(p.mode, galaxy=p.galaxy)

# print(f"{prefix} {nout[-1]} ~ {nout[0]}")
if p.iout == -1:
    p.iout = nout[0]

progstr = "Descendant"
if p.prog:
    progstr = "Progenitor"
message = f"< YoungTree >\nfinding {progstr}s\nUsing {modename} {galstr}\n{len(nout)} outputs are found! ({nout[-1]}~{nout[0]})\n"
#########################################################
###############         Debugger                #########
#########################################################
debugger = None
fname = f"./{p.logname}.log"
if os.path.isfile(fname):
    num = 1
    while os.path.isfile(fname):
        fname = f"./{p.logname}({num}).log"
        num += 1
debugger = logging.getLogger(f"YoungTree_{p.mode}")
debugger.handlers = []
if p.detail:
    debugger.setLevel(logging.DEBUG)
else:
    debugger.setLevel(logging.INFO)
formatter = logging.Formatter(u'%(asctime)s [%(levelname)8s] %(message)s')
file_handler = logging.FileHandler(fname, mode='a')
file_handler.setFormatter(formatter)
debugger.addHandler(file_handler)
debugger.propagate = False
debugger.info("Debug Start")

debugger.info(message)
print(message)

if not p.iout in nout:
    raise ValueError(f"iout={p.iout} is not in nout!")

uri.timer.verbose = 0
snap_now = uri.RamsesSnapshot(repo, p.iout, path_in_repo='snapshots', mode=rurmode )
gals_now = uhmi.HaloMaker.load(snap_now, galaxy=True)
snap_now.clear()

if p.usegals == 'all':
    targets = gals_now
    message = f"All {galstrs}\n>>> {len(targets)} {galstrs} are loaded"
    loadall = True
else:
    loadall = False
    if isinstance(p.usegals, int):
        message = f"Single {galstr}\n>>> {galstr} (ID={p.usegals}) is loaded"
        targets = gals_now[p.usegals - 1]
    elif isinstance(p.usegals, list):
        message = f"Multiple {galstrs}\n>>> {len(p.usegals)} {galstrs} are loaded\n\t(IDs={''.join([f'{p.usegals[i]}, ' if i<min(3,len(p.usegals)) else f', {p.usegals[i]}' if (i>max(3,len(p.usegals))-3) else '...' if i==min(3,len(p.usegals)) else '' for i in range(len(p.usegals))])})"
        ind = np.array(p.usegals, dtype=int)
        targets = gals_now[ind - 1]
    elif isinstance(p.usegals, tuple):
        mmin, mmax = p.usegals
        targets = gals_now[(gals_now['m'] >= mmin) & (gals_now['m'] < mmax)]
        ids = targets['id']
        message = f"Mass range: {np.log10(mmin):.2f} ~ {np.log10(mmax):.2f}\n>>> {len(targets)} {galstrs} are loaded\n\t(ID={''.join([f'{ids[i]}, ' if i<min(3,len(ids)) else f', {ids[i]}' if (i>max(3,len(ids))-3) else '...' if i==min(3,len(ids)) else '' for i in range(len(ids))])})"
    else:
        raise TypeError(f"Couldn't understand type of `usegals` ({type(p.usegals)}) in params.py!")
    targets = np.atleast_1d(targets)

debugger.info(message)
print(message)

if not os.path.isdir(f"./result"):
    os.mkdir(f"./result")
if not os.path.isdir(f"./result/{p.mode}"):
    os.mkdir(f"./result/{p.mode}")

if p.overwrite:
    fnames = [f"./result/{p.mode}/{progstr}_Branch_ID{target['id']:07d}_iout{p.iout:05d}.pickle" for target in targets]
    ind = np.isin(fnames, os.listdir("./result"))    
    if howmany(ind,True)>0:
        debugger.info(f"I find {howmany(ind, True)} of {len(targets)} saved files! --> Use {howmany(ind, False)} {galstrs}\n")
        print(f"I find {howmany(ind, True)} of {len(targets)} saved files! --> Use {howmany(ind, False)} {galstrs}\n")
        targets = targets[~ind]
else:
    debugger.info(" ")
    print()
        



#########################################################
###############         Tree Making                ######
#########################################################
debugger.info(f"Allow {p.flush_GB:.2f} GB Memory")
print(f"Allow {p.flush_GB:.2f} GB Memory")

MyTree = Treebase(simmode=p.mode, debugger=debugger, verbose=0, flush_GB=p.flush_GB, loadall=loadall, prog=p.prog)

print(f"\n\nSee {fname} (detail debugging = {p.detail})\n\nRunning...\n\n")

#########################################################
###############         Execution                ########
#########################################################
MyTree.queue(p.iout, targets)
print("\nDone\n")