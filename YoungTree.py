import sys
import importlib
from ytool import *
from yrun import make_log, do_onestep, connect, gather, DebugDecorator, build_branch, follow_log
import yroot
import time
from numba import set_num_threads
import gc
import argparse, subprocess

# Current Issue:
# - When rerun, file name is written as "logprefix_"
# - However, let's fix it for fixed prefix: ytree_, but only for log file follows logprefix
# - Also, we have lot's of trash files for `ytree_re`. Let's extract nice info from that:

# Read command
print("ex: $ python3 YoungTree.py params.py [--ncpu 32] [--mode y07206]")
# args = sys.argv
# assert len(args)>=2

parser = argparse.ArgumentParser(description='YoungTree (syj3514@yonsei.ac.kr)')
parser.add_argument("params", metavar='params', help='params.py', type=str)
parser.add_argument("-n", "--ncpu", required=False, help='The number of threads', type=int)
parser.add_argument("-m", "--mode", required=False, help='Simulation mode', type=str)
args = parser.parse_args()

if __name__=='__main__':
    # Read params
    params = make_params_dict(args.params, mode=args.mode)
    if(params.nice>0): os.nice(params.nice)
    mainlog, resultdir,fname = make_log(params.repo, "main", detail=params.detail, prefix=params.logprefix, path_in_repo=params.path_in_repo)
    params.resultdir = resultdir
    mainlog.info(f"\nAllow {params.flushGB:.2f} GB Memory\n"); print(f"\nAllow {params.flushGB:.2f} GB Memory\n")
    mainlog.info(f"\nSee `{fname}`\n"); print(f"\nSee `{fname}`\n")
    if(args.ncpu is not None): params.ncpu = args.ncpu
    set_num_threads(params.ncpu)

    if not os.path.exists(f"{params.resultdir}/by-product"):
        os.mkdir(f"{params.resultdir}/by-product")
    if not os.path.exists(f"{params.resultdir}/log"):
        os.mkdir(f"{params.resultdir}/log")
    fout = np.max(params.nout)
    if os.path.exists(f"{params.resultdir}/{params.fileprefix}stable.pickle"):
        mainlog.info(f"`{params.resultdir}/{params.fileprefix}stable.pickle` found... check status...")
        stable = pklload(f"{params.resultdir}/{params.fileprefix}stable.pickle")
        maxout = np.max(stable['timestep'])
        further = maxout < fout
    try:
        if (not os.path.exists(f"{params.resultdir}/{params.fileprefix}stable.pickle"))or(further):
            if (not os.path.exists(f"{params.resultdir}/{params.fileprefix}fatson.pickle"))or(further):
                if (not os.path.exists(f"{params.resultdir}/{params.fileprefix}all.pickle"))or(further):
                    if (not os.path.exists(f"{params.resultdir}/by-product/{params.fileprefix}checkpoint.pickle"))or(further):
                        if(not os.path.exists(f"{params.resultdir}/{params.logprefix}treebase.temp.pickle"))or(not params.takeover):
                            treebase = yroot.TreeBase(params, logger=mainlog)
                            pklsave(treebase, f"{params.resultdir}/{params.logprefix}treebase.temp.pickle", overwrite=True)
                            del treebase
                        reftime = time.time()
                        for iout in params.nout:
                            if os.path.exists(f"{params.resultdir}/by-product/{params.fileprefix}{iout:05d}.pickle"):
                                if(params.takeover):
                                    if(iout == np.max(params.nout))and(params.takeover)and(not params.default):
                                        mainlog.info("Calculate last pids...")
                                        treebase = pklload(f"{params.resultdir}/{params.logprefix}treebase.temp.pickle")
                                        treebase.load_gals(iout)
                                        pklsave(treebase, f"{params.resultdir}/{params.logprefix}treebase.temp.pickle", overwrite=True)
                                    fout = np.max(params.nout[params.nout<iout])
                                    continue
                                else:
                                    mainlog.warning(f"! No takeover ! Remove `{resultdir}/by-product/{params.fileprefix}{iout:05d}.pickle`")
                                    os.remove(f"{params.resultdir}/by-product/{params.fileprefix}{iout:05d}.pickle")
                            subdir = os.getcwd()
                            if(not 'YoungTree' in subdir): subdir = f"{subdir}/YoungTree"

                            
                            #For tardis07, /gem_home/jeon/.conda/envs/py310/bin/python3
                            subprocess.run(["python3", f"{subdir}/ysub.py", str(iout), str(fout), str(reftime), params.resultdir, params.logprefix, mainlog.name], check=True)


                            if(os.path.exists(f"{params.resultdir}/{params.logprefix}success.tmp")):
                                os.remove(f"{params.resultdir}/{params.logprefix}success.tmp")
                            else:
                                if(os.path.exists(f"{params.resultdir}/by-product/{params.fileprefix}{iout:05d}.pickle")):
                                    pass
                                else:
                                    if(os.path.exists(f"{params.resultdir}/by-product/{params.fileprefix}{iout:05d}_temp")):
                                        pass
                                    else:
                                        raise RuntimeError("No success.tmp")
                            time.sleep(1)
                        # for iout in params.nout:
                        #     do_onestep(treebase, iout, reftot=reftime)
                        #     if treebase.memory > treebase.p.flushGB:
                        #         treebase.flush(iout)
                        treebase = pklload(f"{params.resultdir}/{params.logprefix}treebase.temp.pickle")
                        if(treebase.mainlog.name != mainlog.name):
                            treebase.mainlog = follow_log(mainlog.name, detail=treebase.p.detail)
                        treebase.logger = mainlog
                        # treebase.p = DotDict(treebase.p)
                        outs = treebase.out_on_table
                        for iout in outs:
                            treebase.finalize(iout, level='info')
                            treebase.flush(iout)
                        treebase.out_on_table=[]
                        treebase.mainlog.info(f"\n{treebase.summary()}\n")
                        
                        pklsave(np.array([]), f"{params.resultdir}/by-product/{params.fileprefix}checkpoint.pickle")
                        treebase.mainlog.info("\nLeaf save Done\n"); print("\nLeaf save Done\n")
                        treebase = None
                        del treebase
                        os.remove(f"{params.resultdir}/{params.logprefix}treebase.temp.pickle")
                        gc.collect()

                    func = DebugDecorator(gather, params=params, logger=mainlog)
                    func(params, mainlog)
                    mainlog.info("\nGather Done\n"); print("\nGather Done\n")
                    
                mainlog.info("\nConnect Start\n"); print("\nConnect Start\n")
                connectlog, resultdir,_ = make_log(params.repo, "connect", detail=params.detail, prefix=params.logprefix, path_in_repo=params.path_in_repo)
                func = DebugDecorator(connect, params=params, logger=connectlog)
                func(params, connectlog)
                mainlog.info("\nConnect Done\n"); print("\nConnect Done\n")

            mainlog.info("\nBuild Start\n"); print("\nBuild Start\n")
            buildlog, resultdir,_ = make_log(params.repo, "build", detail=params.detail, prefix=params.logprefix, path_in_repo=params.path_in_repo)
            func = DebugDecorator(build_branch, params=params, logger=buildlog)
            func(params, buildlog)
            mainlog.info("\nBuild Done\n"); print("\nBuild Done\n")

        mainlog.info(f"\nYoungTree Done\nSee `{params.resultdir}/{params.fileprefix}stable.pickle`"); print(f"\nYoungTree Done\nSee `{params.resultdir}/{params.fileprefix}stable.pickle`")
    except Exception as e:
        print(); mainlog.error("")
        print(traceback.format_exc()); mainlog.error(traceback.format_exc())
        print(e); mainlog.error(e)
        print("\nIteration is terminated (`__main__`)\n"); mainlog.error("\nIteration is terminated (`__main__`)\n")
        sys.exit(1)