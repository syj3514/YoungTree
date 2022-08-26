import gc
import psutil
import time
import numpy as np
from rur import uri, uhmi

from tree_utool import *
from tree_root import Treebase

#########################################################
###############         Targets                ##########
#########################################################
Data = None
gc.collect()
print(f"\n>>> Current Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB")
mode = input("\n>>> Mode=? (hagn, yxxxxx, nh..)")
repo, rurmode = mode2repo(mode)

ans = input("\n>>> Use galaxy? ")
galaxy=False
galstr = "Halo"
trues = ["True", "T", "true", "yes", "y", "Y", "Yes"]
if ans in trues:
    galaxy=True
    galstr = "Galaxy"

nout = load_nout(mode, galaxy=galaxy)

prefix = f"[{mode} ({galstr})]"
print(f"{prefix} {nout[-1]} ~ {nout[0]}")
iout = 0
while not iout in nout:
    iout = int( input(">>> Choose iout ") )

print(f"\n{prefix} targets load..."); ref = time.time()
# iout = np.max(nout)
uri.timer.verbose = 0
snap_now = uri.RamsesSnapshot(repo, iout, path_in_repo='snapshots', mode=rurmode )
gals_now = uhmi.HaloMaker.load(snap_now, galaxy=True)
printtime(ref, f"{prefix} {len(gals_now)} gals (at iout={iout}) load done")
snap_now.clear()
print(f"\n>>> Current Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB")

ans = input("\n>>> Use all? (`all`, or specific ID)")
if ans == 'all':
    print("All galaxies")
    targets = gals_now
    loadall = True
else:
    print(f"Galaxy ID={ans}")
    targets = gals_now[int(ans)-1]
    printgal(targets, mode=mode)
    loadall = False
targets = np.atleast_1d(targets)

fnames = [f"Branch_{mode}_ID{target['id']:07d}_iout{iout:05d}.pickle" for target in targets]
ind = np.isin(fnames, os.listdir("./data"))

if howmany(ind,True)>0:
    ans = input(f"\n>>> I find {howmany(ind, True)} of {len(targets)} saved files! Do you want to exclude them?")
    if ans in trues:
        targets = targets[~ind]
        print(f"Ok, I'll calculate {len(targets)} gals")


#########################################################
###############         Debugger                #########
#########################################################
# import warnings
# warnings.simplefilter('error')

debugger = None
fname = f"./output_{mode}.log"
ans = input(f"\n>>> Log file will be saved in `{fname}`. Agree? ")
if not ans in trues:
    fname = input(f"\n>>> Type new name (ex: ./output_dev.log) ")
if os.path.isfile(fname):
    os.remove(fname)
debugger = logging.getLogger(f"YoungTree_{mode}")
debugger.handlers = []
ans = input("\n>>> Detail debugging? ")
if ans in trues:
    print("DEBUG level")
    debugger.setLevel(logging.DEBUG)
else:
    print("INFO level")
    debugger.setLevel(logging.INFO)
formatter = logging.Formatter(u'%(asctime)s [%(levelname)8s] %(message)s')
file_handler = logging.FileHandler(fname, mode='a')
file_handler.setFormatter(formatter)
debugger.addHandler(file_handler)
debugger.propagate = False
debugger.debug("Debug Start")
print(f"\n>>> Current Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB")

#########################################################
###############         Tree Making                ######
#########################################################
flush_GB = 10 + 0.3*len(targets)

print(f"\n>>> Current Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB")

if mode == 'nh':
    flush_GB *= 10
print(f"Allow {flush_GB:.2f} GB Memory")
MyTree = Treebase(simmode=mode, debugger=debugger, verbose=0, flush_GB=flush_GB, loadall=loadall)
print(f"\n\nSee {fname}\n\nRunning...\n")

#########################################################
###############         Execution                ########
#########################################################
MyTree.queue(iout, targets)
print("\nDone\n")