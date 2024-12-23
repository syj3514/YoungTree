from ytool import *
from yrun import do_onestep, follow_log
import argparse
import sys

try:
    parser = argparse.ArgumentParser(description='YoungTree subroutine')
    parser.add_argument("iout", metavar='iout', help='iout', type=int)
    parser.add_argument("fout", metavar='fout', help='fout', type=int)
    parser.add_argument("reftot", metavar='reftot', help='reference time for total computation time', type=float)
    parser.add_argument("resultdir", metavar='resultdir', help='result_directory', type=str)
    parser.add_argument("logprefix", metavar='logprefix', help='prefix of logfile', type=str)
    parser.add_argument("mainlogname", metavar='mainlogname', help='name of mainlog', type=str)
    parser.add_argument("maxout", metavar='maxout', help='name of mainlog', type=int)
    args = parser.parse_args()
    iout = args.iout
    fout = args.fout
    reftot = args.reftot
    resultdir = args.resultdir
    logprefix = args.logprefix
    mainlogname = args.mainlogname
    maxout = args.maxout
    with(open(f"{resultdir}/{logprefix}current_{iout:05d}.tmp", "wb")) as f:
        f.write(b"current")

    if(os.path.exists(f"{resultdir}/{logprefix}treebase.temp.pickle")):
        treebase = pklload(f"{resultdir}/{logprefix}treebase.temp.pickle")
        if(treebase.mainlog.name != mainlogname):
            treebase.mainlog = follow_log(mainlogname, detail=treebase.p.detail)
        treebase.mainlog.info(f"`treebase.temp.pickle` is found")
    else:
        raise FileExistsError("No `treebase.temp.pickle`")




    time_record = do_onestep(treebase, iout, fout, maxout, reftot=reftot)




    if treebase.memory > treebase.p.flushGB:
        treebase.print(f"\nMemory exceed\n", level='info')
        treebase.flush(iout)
    pklsave(treebase, f"{resultdir}/{logprefix}treebase.temp.pickle", overwrite=True)
    treebase.print(f"\niout={iout} done\n", level='info')
    msgs = [msg[0] for msg in time_record]
    times = [msg[1] for msg in time_record]
    total_time = np.sum(times)
    treebase.print(f"Total time: {total_time:.2f} sec", level='info')
    maxmsg = 'none'
    maxtime = 0
    for imsg, itime in zip(msgs, times):
        if(itime > maxtime):
            maxmsg = imsg
            maxtime = itime
        treebase.print(f"\t[{itime/total_time*100:05.2f} %] {imsg}: {itime:.2f} sec", level='info')
    treebase.print(f"\n\t`{maxmsg}` is bottleneck ({maxtime:.2f} sec)", level='info')
    del treebase
except Exception as e:
    print("[Error in `ysub.py`]")
    print()
    print(e)
    print(traceback.format_exc())
    files = glob.glob(f"{resultdir}/{logprefix}current_*.tmp")
    for file in files: os.remove(file)
    sys.exit(1)
except KeyboardInterrupt:
    print("[Keyboard interrupt in `ysub.py`]")
    print()
    print(traceback.format_exc())
    files = glob.glob(f"{resultdir}/{logprefix}current_*.tmp")
    for file in files: os.remove(file)
    sys.exit(1)

