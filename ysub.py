from ytool import *
from yrun import do_onestep
import argparse


try:
    parser = argparse.ArgumentParser(description='YoungTree subroutine')
    parser.add_argument("iout", metavar='iout', help='iout', type=int)
    parser.add_argument("reftot", metavar='reftot', help='reference time for total computation time', type=float)
    parser.add_argument("resultdir", metavar='resultdir', help='result_directory', type=str)
    parser.add_argument("logprefix", metavar='logprefix', help='prefix of logfile', type=str)
    args = parser.parse_args()
    iout = args.iout
    reftot = args.reftot
    resultdir = args.resultdir
    logprefix = args.logprefix
    with(open(f"{resultdir}/{logprefix}success.tmp", "wb")) as f:
        f.write(b"success")

    if(os.path.exists(f"{resultdir}/{logprefix}treebase.temp.pickle")):
        treebase = pklload(f"{resultdir}/{logprefix}treebase.temp.pickle")
        treebase.mainlog.info(f"`treebase.temp.pickle` is found")
    else:
        raise FileExistsError("No treebase.temp.pickle")

    do_onestep(treebase, iout, reftot=reftot)
    if treebase.memory > treebase.p.flushGB:
        treebase.print(f"\nMemory exceed\n")
        treebase.flush(iout)
    pklsave(treebase, f"{resultdir}/{logprefix}treebase.temp.pickle", overwrite=True)
    treebase.print(f"\niout={iout} done\n")
except Exception as e:
    print()
    print(e)
    print(traceback.format_exc())
    os.remove(f"{resultdir}/{logprefix}success.tmp")
    os._exit(1)

