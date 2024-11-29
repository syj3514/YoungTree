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

    FILE_YTREE_STABLE = f"{params.resultdir}/{params.fileprefix}stable.pickle"
    FILE_CHECKPOINT = f"{params.resultdir}/by-product/{params.fileprefix}checkpoint.pickle"
    FILE_GATHER = f"{params.resultdir}/by-product/{params.fileprefix}all.pickle"
    FILE_FATSON = f"{params.resultdir}/{params.fileprefix}fatson.pickle"
    FILE_TREEBASE = f"{params.resultdir}/{params.logprefix}treebase.temp.pickle"
    DIR_BY_PRODUCT = f"{params.resultdir}/by-product"
    DIR_LOG = f"{params.resultdir}/log"

    if not os.path.exists(DIR_BY_PRODUCT):
        os.mkdir(DIR_BY_PRODUCT)
    if not os.path.exists(DIR_LOG):
        os.mkdir(DIR_LOG)
    fout = np.max(params.nout)
    print(params.nout)
    if os.path.exists(FILE_YTREE_STABLE):
        print(f"`{FILE_YTREE_STABLE}` found... check status...")
        mainlog.info(f"`{FILE_YTREE_STABLE}` found... check status...")
        stable = pklload(FILE_YTREE_STABLE)
        maxout = np.max(stable['timestep'])
        further = maxout < fout
        print(f"Further: {further} ({maxout}<{fout})")
        if(os.path.exists(FILE_CHECKPOINT))and(further):
            os.remove(FILE_CHECKPOINT)
    try:
        if (not os.path.exists(FILE_YTREE_STABLE))or(further):
            if (not os.path.exists(FILE_FATSON))or(further):
                if (not os.path.exists(FILE_GATHER))or(further):
                    if (not os.path.exists(FILE_CHECKPOINT))or(further): # <- All pickle saved
                        #-------------------------------------------------------------------------------------
                        # ysub done, but not gathered
                        #-------------------------------------------------------------------------------------
                        check_filelist = True
                        for iout in params.nout:
                            FILE_BRICK = f"{params.resultdir}/by-product/{params.fileprefix}{iout:05d}.pickle"
                            if not os.path.exists(FILE_BRICK):
                                check_filelist = False
                                break
                        if check_filelist:
                            mainlog.info("All pickle files are saved! It's time to `gather`!")
                            pklsave(["All pickle files are saved! It's time to `gather`!"], FILE_CHECKPOINT)
                        #-------------------------------------------------------------------------------------
                        # ysub not completed
                        #-------------------------------------------------------------------------------------
                        else:
                            if(not os.path.exists(FILE_TREEBASE))or(not params.takeover):
                                treebase = yroot.TreeBase(params, logger=mainlog)
                                pklsave(treebase, FILE_TREEBASE, overwrite=True)
                                del treebase
                            reftime = time.time()
                            for iout in params.nout:
                                FILE_BRICK = f"{params.resultdir}/by-product/{params.fileprefix}{iout:05d}.pickle"
                                if os.path.exists(FILE_BRICK):
                                    if(params.takeover):
                                        if(iout == np.max(params.nout))and(params.takeover)and(not params.default):
                                            mainlog.info("Calculate last pids...")
                                            treebase = pklload(FILE_TREEBASE)
                                            treebase.load_gals(iout)
                                            pklsave(treebase, FILE_TREEBASE, overwrite=True)
                                        if(iout == np.min(params.nout)):
                                            break
                                        fout = np.max(params.nout[params.nout<iout])
                                        continue
                                    else:
                                        mainlog.warning(f"! No takeover ! Remove `{FILE_BRICK}`")
                                        os.remove(FILE_BRICK)
                                subdir = os.getcwd()
                                if(not 'YoungTree' in subdir): subdir = f"{subdir}/YoungTree"

                                
                                #For tardis07, /gem_home/jeon/.conda/envs/py310/bin/python3
                                subprocess.run(["python3", f"{subdir}/ysub.py", str(iout), str(fout), str(reftime), params.resultdir, params.logprefix, mainlog.name], check=True)

                                FILE_CURRENT = f"{params.resultdir}/{params.logprefix}current_{iout:05d}.tmp"
                                if(os.path.exists(FILE_CURRENT)):
                                    os.remove(FILE_CURRENT)
                                else:
                                    if(os.path.exists(FILE_BRICK)):
                                        pass
                                    else:
                                        if(os.path.exists(f"{params.resultdir}/by-product/{params.fileprefix}{iout:05d}_temp")):
                                            pass
                                        else:
                                            raise RuntimeError("No current file")
                                time.sleep(1)

                            treebase = pklload(FILE_TREEBASE)
                            if(treebase.mainlog.name != mainlog.name):
                                treebase.mainlog = follow_log(mainlog.name, detail=treebase.p.detail)
                            treebase.logger = mainlog

                            outs = treebase.out_on_table
                            for iout in outs:
                                treebase.finalize(iout, level='info')
                                treebase.flush(iout)
                            treebase.out_on_table=[]
                            treebase.mainlog.info(f"\n{treebase.summary()}\n")
                            
                            pklsave(["All pickle files are saved! It's time to `gather`!"], FILE_CHECKPOINT)
                            treebase.mainlog.info("\nLeaf save Done\n"); print("\nLeaf save Done\n")
                            treebase = None
                            del treebase
                            os.remove(FILE_TREEBASE)
                            gc.collect()
                        #-------------------------------------------------------------------------------------

                    func = DebugDecorator(gather, params=params, logger=mainlog)
                    func(params, mainlog)
                    mainlog.info("\nGather Done\n"); print("\nGather Done\n")
                    
                mainlog.info("\nConnect Start\n"); print("\nConnect Start\n")
                connectlog, resultdir, logname = make_log(params.repo, "connect", detail=params.detail, prefix=params.logprefix, path_in_repo=params.path_in_repo)
                mainlog.info(f"\nSee `{logname}`\n"); print(f"\nSee `{logname}`\n")
                func = DebugDecorator(connect, params=params, logger=connectlog)
                func(params, connectlog)
                mainlog.info("\nConnect Done\n"); print("\nConnect Done\n")

            mainlog.info("\nBuild Start\n"); print("\nBuild Start\n")
            buildlog, resultdir, logname = make_log(params.repo, "build", detail=params.detail, prefix=params.logprefix, path_in_repo=params.path_in_repo)
            mainlog.info(f"\nSee `{logname}`\n"); print(f"\nSee `{logname}`\n")
            func = DebugDecorator(build_branch, params=params, logger=buildlog)
            func(params, buildlog)
            mainlog.info("\nBuild Done\n"); print("\nBuild Done\n")

        mainlog.info(f"\nYoungTree Done\nSee `{FILE_YTREE_STABLE}`"); print(f"\nYoungTree Done\nSee `{FILE_YTREE_STABLE}`")
    except Exception as e:
        print(); mainlog.error("")
        print(traceback.format_exc()); mainlog.error(traceback.format_exc())
        print(e); mainlog.error(e)
        print("\nIteration is terminated (`__main__`)\n"); mainlog.error("\nIteration is terminated (`__main__`)\n")
        sys.exit(1)