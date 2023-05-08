import sys
import importlib
from ytool import *
from yrun import make_log, do_onestep, connect, gather, DebugDecorator
import yroot
import time
from numba import set_num_threads
import gc

# Read command
print("$ python3 YoungTree.py params.py <ncpu>")
args = sys.argv
assert len(args)>=2

if __name__=='__main__':
    # Read params
    params = make_params_dict(args[1])
    if(params.nice>0): os.nice(params.nice)
    mainlog, resultdir,_ = make_log(params.repo, "main", detail=params.detail, prefix=params.logprefix)
    params.resultdir = resultdir
    mainlog.info(f"\nAllow {params.flushGB:.2f} GB Memory\n"); print(f"\nAllow {params.flushGB:.2f} GB Memory\n")
    mainlog.info(f"\nSee `{params.resultdir}`\n"); print(f"\nSee `{params.resultdir}`\n")
    if len(args)>2:
        params.ncpu = int(args[2])
    set_num_threads(params.ncpu)

    if not os.path.exists(f"{params.resultdir}/by-product"):
        os.mkdir(f"{params.resultdir}/by-product")
    if not os.path.exists(f"{params.resultdir}/log"):
        os.mkdir(f"{params.resultdir}/log")

    if not os.path.isfile(f"{params.resultdir}/{params.logprefix}stable.pickle"):
        if not os.path.isfile(f"{params.resultdir}/{params.logprefix}all.pickle"):
            if not os.path.isfile(f"{params.resultdir}/by-product/{params.logprefix}checkpoint.pickle"):
                treebase = yroot.TreeBase(params, logger=mainlog)
                
                reftime = time.time()
                for iout in params.nout:
                    do_onestep(treebase, iout, reftot=reftime)
                    if treebase.memory > treebase.p.flushGB:
                        treebase.flush(iout)

                outs = list(treebase.dict_leaves.keys())
                for iout in outs:
                    treebase.flush(iout, leafclear="True")
                    treebase.finalize(iout)
                treebase.mainlog.info(f"\n{treebase.summary()}\n")
                
                pklsave(np.array([]), f"{params.resultdir}/by-product/{params.logprefix}checkpoint.pickle")
                treebase.mainlog.info("\nLeaf save Done\n"); print("\nLeaf save Done\n")
                treebase = None
                gc.collect()

            func = DebugDecorator(gather, params=params, logger=mainlog)
            func(params, mainlog)
            mainlog.info("\nGather Done\n"); print("\nGather Done\n")
            
        
        mainlog.info("\nConnect Start\n"); print("\nConnect Start\n")
        connectlog, resultdir,_ = make_log(params.repo, "connect", detail=params.detail, prefix=params.logprefix)
        
        func = DebugDecorator(connect, params=params, logger=connectlog)
        func(params, connectlog)
        mainlog.info("\nConnect Done\n"); print("\nConnect Done\n")
    
    mainlog.info(f"\nYoungTree Done\nSee `{params.resultdir}/{params.logprefix}stable.pickle`"); print(f"\nYoungTree Done\nSee `{params.resultdir}/{params.logprefix}stable.pickle`")