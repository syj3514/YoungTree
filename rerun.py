import gc
import psutil
import time
import numpy as np
from rur import uri, uhmi
import warnings
import logging

from tree_utool import *
from tree_root import Treebase
import importlib

# module = input("params? (____.py)")
# params = importlib.import_module(module)


mode = input("Mode: (hagn, nh, y39990, ...)")
fname = input(f"`./log/[ fname ]_{mode}_ini.log.params` file: ") #########################
logprefix = f"re_{fname}"
fname = f"./log/{mode}/{fname}_{mode}_ini.log.params"
p = pklload(fname) #########################
p.logprefix = logprefix
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
            "nh": "NewHorizon",
            "nh2": "NewHorizon2",
            "nc": "NewCluster",
            "fornax": "FORNAX"
            }



if not p.mode in modenames.keys():
    raise ValueError(f"mode={p.mode} is not supported!")
modename = modenames[p.mode]
repo, rurmode, dp = mode2repo(p.mode)
if dp:
    if not p.galaxy:
        dp = False

# ans = input("\n>>> Use galaxy? ")
galstr = "Halo"
galstrs = "Halos"
if p.galaxy:
    galstr = "Galaxy"
    galstrs = "Galaxies"
progstr = "Descendant"
if p.prog:
    progstr = "Progenitor"

nout = load_nout(p.mode, galaxy=p.galaxy)
if p.iout == -1:
    p.iout = nout[0]
if not p.iout in nout:
    raise ValueError(f"iout={p.iout} is not in nout!")

message = f"< YoungTree >\n[Re-Run] finding {progstr}s\n[Re-Run] Using {modename} {galstr}\n[Re-Run] {len(nout)} outputs are found! ({nout[-1]}~{nout[0]})\n"
#########################################################
###############         Debugger                #########
#########################################################
debugger = None
fname = make_logname(p.mode, -1, logprefix=p.logprefix)


debugger = custom_debugger(fname, detail=p.detail)
debugger.info(message)
print(message)


uri.timer.verbose = 0
snap_now = uri.RamsesSnapshot(repo, p.iout, path_in_repo='snapshots', mode=rurmode )
gals_now = uhmi.HaloMaker.load(snap_now, galaxy=p.galaxy, double_precision=dp)
snap_now.clear()

if p.usegals == 'all':
    targets = gals_now
    message = f"[Re-Run] All {galstrs}\n>>> {len(targets)} {galstrs} are loaded"
    loadall = True
else:
    loadall = False
    if isinstance(p.usegals, int):
        message = f"[Re-Run] Single {galstr}\n>>> {galstr} (ID={p.usegals}) is loaded"
        targets = gals_now[p.usegals - 1]
    elif isinstance(p.usegals, list):
        message = f"[Re-Run] Multiple {galstrs}\n>>> {len(p.usegals)} {galstrs} are loaded\n\t(IDs={''.join([f'{p.usegals[i]}, ' if i<min(3,len(p.usegals)) else f', {p.usegals[i]}' if (i>max(3,len(p.usegals))-3) else '...' if i==min(3,len(p.usegals)) else '' for i in range(len(p.usegals))])})"
        ind = np.array(p.usegals, dtype=int)
        targets = gals_now[ind - 1]
    elif isinstance(p.usegals, tuple):
        mmin, mmax = p.usegals
        targets = gals_now[(gals_now['m'] >= mmin) & (gals_now['m'] < mmax)]
        ids = targets['id']
        message = f"Mass range: {np.log10(mmin):.2f} ~ {np.log10(mmax):.2f}\n>>> {len(targets)} {galstrs} are loaded\n\t(ID={''.join([f'{ids[i]}, ' if i<min(3,len(ids)) else f', {ids[i]}' if (i>max(3,len(ids))-3) else '...' if i==min(3,len(ids)) else '' for i in range(len(ids))])})"
    elif isinstance(p.usegals, dict):
        center = p.usegals['center']
        radii = p.usegals['radii']
        targets = cut_sphere(gals_now, *center, radii, both_sphere=True)
        ids = targets['id']
        message = f"Sphere cut: {radii:.3f} from [{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]\n>>> {len(targets)} {galstrs} are loaded\n\t(ID={''.join([f'{ids[i]}, ' if i<min(3,len(ids)) else f', {ids[i]}' if (i>max(3,len(ids))-3) else '...' if i==min(3,len(ids)) else '' for i in range(len(ids))])})"
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
    if p["saved_ind"] is not None:
        ind = p["saved_ind"]
    else:
        ind = np.isin(fnames, os.listdir("./result"))    
        if howmany(ind,True)>0:
            debugger.info(f"I find {howmany(ind, True)} of {len(targets)} saved files! --> Use {howmany(ind, False)} {galstrs}\n")
            print(f"I find {howmany(ind, True)} of {len(targets)} saved files! --> Use {howmany(ind, False)} {galstrs}\n")
            targets = targets[~ind]
else:
    debugger.info(" ")
    print()
        
pklsave(p, f"{fname}.params", overwrite=True)

#########################################################
###############         Tree Making                ######
#########################################################
debugger.info(f"\nAllow {p.flush_GB:.2f} GB Memory")
print(f"[Re-Run] Allow {p.flush_GB:.2f} GB Memory")

jout = input("[Re-Run] When do you want to go back? (jout): ")
backupfname = make_logname(p.mode, int(jout), logprefix=p.logprefix[3:], overwrite=True)
print(f"Load backup data from `{backupfname}.pickle`")
backup_dict = pklload(f"{backupfname}.pickle")


MyTree = Treebase(simmode=p.mode, galaxy=p.galaxy, debugger=debugger, verbose=0, flush_GB=p.flush_GB, loadall=loadall, prog=p.prog, detail=p.detail, logprefix=p.logprefix, dp=dp)
print(backup_dict['Root']["snapkeys"])
MyTree.import_backup(backup_dict["Root"])

destination = np.min(nout) if p.prog else np.max(nout)
print(f"\nStart making {progstr} from {p.iout} to {destination}\nSee {fname} (detail debugging = {p.detail})\nRunning...\n\n")

#########################################################
###############         Execution                ########
#########################################################
MyTree.queue(p.iout, targets, backup_dict=backup_dict)
print("\nDone\n")