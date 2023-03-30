from ytool import *
import os
import logging
from logging.handlers import RotatingFileHandler
import psutil
import time
import traceback
import sys
from numba import set_num_threads
from numpy.lib.recfunctions import append_fields

####################################################################################################################
# Main run
####################################################################################################################
def treerecord(iout:int, nout:int, elapse_s:float, total_elapse_s:float, logger:logging.Logger):
    a = f"{iout} done ({elapse_s/60:.2f} min elapsed)"
    logger.info(a)
    aver = total_elapse_s/60/len(nout[nout>=iout])
    a = f"{len(nout[nout>=iout])}/{len(nout)} done ({aver:.2f} min/snap)"
    logger.info(a)
    a = f"{aver*len(nout[nout<iout]):.2f} min forecast"
    logger.info(a)
    a = f"{psutil.Process().memory_info().rss / 2 ** 30:.4f} GB used\n" # memory used
    logger.info(a)

def do_onestep(Tree, iout:int, reftot=time.time()):
    nout = Tree.p.nout
    nstep = Tree.p.nstep
    resultdir = Tree.p.resultdir
    set_num_threads(Tree.p.ncpu)

    try:
        ref = time.time()
        skip = False
        
        # Fully saved
        if os.path.isfile(f"{resultdir}/{Tree.p.logprefix}{iout:05d}.pickle"):
            Tree.mainlog.info(f"[Queue] {iout} is done --> Skip\n")
            skip=True
        
        # Temp exists
        if os.path.isfile(f"{resultdir}/{Tree.p.logprefix}{iout:05d}_temp.pickle"):
            Tree.mainlog.info(f"[Queue] `{resultdir}/{Tree.p.logprefix}{iout:05d}_temp.pickle` is found")
            istep = Tree.out2step(iout)
            cutstep = istep+5
            if cutstep <= np.max(nstep):
                cutout = Tree.step2out(cutstep)
                if os.path.isfile(f"{resultdir}/{Tree.p.logprefix}{cutout:05d}_temp.pickle"):
                    Tree.mainlog.info(f"[Queue] `{resultdir}/{Tree.p.logprefix}{cutout:05d}_temp.pickle` is found --> Do\n")
                    skip=False
                else:
                    Tree.mainlog.info(f"[Queue] `{resultdir}/{Tree.p.logprefix}{cutout:05d}_temp.pickle` is not found --> Skip\n")
                    skip=True
            else:
                skip=False
    
        # Main process
        if not skip:
            # New log file
            Tree.mainlog.info(f"[Queue] {iout} start\n")
            Tree.logger, _ = make_log(Tree.p.repo, f"{iout:05d}", detail=Tree.p.detail, prefix=Tree.p.logprefix)
            Tree.update_debugger()
            Tree.logger.info(f"\n{Tree.summary()}\n")
            # Load snap gal part
            Tree.logger.info(f"\n\nStart at iout={iout}\n")
            Tree.load_from_backup(iout, level='info')
            Tree.logger.info("\n")
            istep = Tree.out2step(iout)
            # Find progenitors
            for j in range(Tree.p.nsnap):
                jstep = istep-j-1
                if jstep > 0:
                    jout = Tree.step2out(jstep)
                    Tree.logger.info(f"\n\nProgenitor at jout={jout}\n")
                    Tree.load_from_backup(jout, level='info')
                    Tree.logger.info("\n")
                    Tree.find_cands(iout, jout, level='info')
            # Find descendants
            for j in range(Tree.p.nsnap):
                jstep = istep+j+1
                if jstep <= np.max(nstep):
                    jout = Tree.step2out(jstep)
                    if not jout in Tree.dict_snap.keys():
                        Tree.logger.info(f"\n\nDescendant at jout={jout}\n")
                        Tree.load_from_backup(jout, level='info')
                        Tree.logger.info("\n")
                        Tree.find_cands(iout, jout, level='info')
            Tree.logger.info(f"\n\n")
            # Flush redundant snapshots
            cutstep = istep+5
            if cutstep<=np.max(nstep):
                cutout = Tree.step2out(cutstep)
                outs = list(Tree.dict_leaves.keys())
                for out in outs:
                    if out > cutout:
                        Tree.flush(out, leafclear=True, level='info')        
                        # reducebackup(Tree, out, resultdir=resultdir)
                        Tree.reducebackup(out, level='info')
            # Backup files
            Tree.leaf_backup(level='info')
            Tree.logger.info(f"\n{Tree.summary()}\n")
            treerecord(iout, nout, time.time()-ref, time.time()-reftot, Tree.mainlog)
    except Exception as e:
        print(traceback.format_exc())
        print(e)
        Tree.logger.error(traceback.format_exc())
        Tree.logger.error(e)
        Tree.logger.error(Tree.summary())
        print("Iteration is terminated")
        os._exit(1)

def gather(p:DotDict, logger:logging.Logger):
    go=True
    if os.path.isfile(f"{p.resultdir}/{p.logprefix}all.pickle"):
        ans=input(f"You already have `{p.resultdir}/{p.logprefix}all.pickle`. Ovewrite? [Y/N]")
        go = ans in yess
    if go:
        logger.info("Gather all files...")
        for i, iout in enumerate(p.nout):
            file = pklload(f"{p.resultdir}/{p.logprefix}{iout:05d}.pickle")
            if i==0:
                gals = file
            else:
                gals = np.hstack((gals, file))
        logger.info("Add column `last`...")
        temp = np.zeros(len(gals), dtype=np.int32)
        gals = append_fields(gals, "last", temp, usemask=False, asrecarray=False)
        logger.info("Convert `prog` and `desc` to easier format...")
        for gal in gals:
            if gal['prog'] is None:
                gal['prog'] = np.array([], dtype=np.int32)
                gal['prog_score'] = np.array([], dtype=np.float64)
            else:
                arg = gal['prog'][:,1]>0
                prog = gal['prog'][:,0]*100000 + gal['prog'][:,1]
                progscore = gal['prog_score'][:,0]
                gal['prog'] = prog[arg].astype(np.int32)
                gal['prog_score'] = progscore[arg].astype(np.float64)
        
            if gal['desc'] is None:
                gal['desc'] = np.array([], dtype=np.int32)
                gal['desc_score'] = np.array([], dtype=np.float64)
            else:
                arg = gal['desc'][:,1]>0
                desc = gal['desc'][:,0]*100000 + gal['desc'][:,1]
                descscore = gal['desc_score'][:,0]
                gal['desc'] = desc[arg].astype(np.int32)
                gal['desc_score'] = descscore[arg].astype(np.float64)
        pklsave(gals, f"{p.resultdir}/{p.logprefix}all.pickle", overwrite=True)
        logger.info(f"`{p.resultdir}/{p.logprefix}all.pickle` saved\n")
    # gals = pklload(f"{p.resultdir}/{p.logprefix}all.pickle")

def connect(p:DotDict, logger:logging.Logger):
    gals = pklload(f"{p.resultdir}/{p.logprefix}all.pickle")
    go=True
    if os.path.isfile(f"{p.resultdir}/{p.logprefix}stable.pickle"):
        ans=input(f"You already have `{p.resultdir}/{p.logprefix}stable.pickle`. Ovewrite? [Y/N]")
        go = ans in yess
    if go:
        logger.info("Make dictionary...")
        gals = append_fields(gals, "from", np.zeros(len(gals), dtype=np.int32), usemask=False)
        gals = append_fields(gals, "fat", np.zeros(len(gals), dtype=np.int32), usemask=False)
        gals = append_fields(gals, "son", np.zeros(len(gals), dtype=np.int32), usemask=False)
        gals = append_fields(gals, "fat_score", np.zeros(len(gals), dtype=np.float64), usemask=False)
        gals = append_fields(gals, "son_score", np.zeros(len(gals), dtype=np.float64), usemask=False)
        inst = {}
        for iout in p.nout:
            inst[iout] = gals[gals['timestep']==iout]
        gals = None

        logger.info("Find son & father...")
        offsets = np.array([1,2,3,4,5])
        for offset in offsets:
            logger.info(f"\n{offset}\n")
            iterobj = p.nout
            for iout in iterobj:
                temp = inst[iout]
                do = np.ones(len(temp), dtype=bool)
                for ihalo in temp:
                    if ihalo['son']<=0:
                        iid = gal2id(ihalo)
                        do[ihalo['id']-1] = False
                        idesc, idscore = maxdesc(ihalo, all=False, offset=offset)
                        if idesc==0:
                            logger.debug(f"\t{iid} No desc")
                            pass
                        else:
                            dhalo = gethalo(idesc//100000, idesc%100000, halos=inst)
                            prog, pscore = maxprog(dhalo, all=False, offset=offset)
                            if prog==iid: # each other
                                nrival = 0
                                for jhalo, ido in zip(temp, do):
                                    if (idesc in jhalo['desc'])&(jhalo['son']<=0)&(ido):
                                        jid = gal2id(jhalo)
                                        if jid != iid:
                                            jdesc, jdscore = maxdesc(jhalo, all=False, offset=offset)
                                            if jdesc==idesc:
                                                do[jhalo['id']-1] = False
                                                nrival += 1
                                                if jhalo['son'] == 0:
                                                    logger.debug(f"\t\trival {jid} newly have son {-idesc} ({jdscore:.4f})")
                                                    jhalo['son'] = -idesc
                                                    jhalo['son_score'] = jdscore
                                                elif jhalo['son'] < 0:
                                                    if jhalo['son_score'] > jdscore:
                                                        logger.debug(f"\t\trival {jid} keep original son {jhalo['son']} ({jhalo['son_score']:.4f}) rather than {-idesc} ({jdscore:.4f})")
                                                    else:
                                                        logger.debug(f"\t\trival {jid} change original son {jhalo['son']} ({jhalo['son_score']:.4f}) to {-idesc} ({jdscore:.4f})")
                                                        jhalo['son'] = -jdesc
                                                        jhalo['son_score'] = jdscore
                                                else:
                                                    logger.debug(f"\t\trival {jid} keep original son {jhalo['son']} ({jhalo['son_score']:.4f}) rather than {-idesc} ({jdscore:.4f})")
                                # if nrival==0:
                                if ihalo['son'] == 0:
                                    logger.debug(f"\t{iid} change original son {ihalo['son']} ({ihalo['son_score']:.4f}) to {idesc} ({idscore:.4f})")
                                    ihalo['son'] = idesc
                                    ihalo['son_score'] = idscore
                                else:
                                    if ihalo['son_score'] > idscore:
                                        logger.debug(f"\t {iid} keep original son {ihalo['son']} ({ihalo['son_score']:.4f}) rather than {idesc} ({idscore:.4f})")
                                    else:
                                        logger.debug(f"\t{iid} change original son {ihalo['son']} ({ihalo['son_score']:.4f}) to {idesc} ({idscore:.4f})")
                                        ihalo['son'] = idesc
                                        ihalo['son_score'] = idscore
                                if dhalo['fat'] == 0:
                                    logger.debug(f"\tAlso, son {idesc} have father {iid} with {pscore:.4f}")
                                    dhalo['fat'] = iid
                                    dhalo['fat_score'] = pscore
                                else:
                                    pass
                            else:
                                logger.debug(f"\thave desc {idesc}, but his prog is {prog}")
                    else:
                        do[ihalo['id']-1] = False

        logger.info("Connect same Last...")
        for iout in p.nout:
            gals = inst[iout]
            for gal in gals:
                if gal['last'] == 0:
                    last = gal2id(gal)
                else:
                    last = gal['last']

                if (gal['son'] != 0):
                    desc = inst[np.abs(gal['son'])//100000][np.abs(gal['son'])%100000 - 1]
                    last = desc['last']
                    # gal['last'] = last

                if (gal['fat']>0):
                    prog = inst[gal['fat']//100000][gal['fat']%100000 - 1]
                    if np.abs(prog['son']) == gal2id(gal):
                        prog['last'] = last
                gal['last'] = last
        logger.info("Connect same From...")
        for iout in p.nout[::-1]:
            gals = inst[iout]
            for gal in gals:
                if gal['from'] == 0:
                    From = gal2id(gal)
                else:
                    From = gal['from']

                if (gal['fat'] != 0):
                    prog = inst[np.abs(gal['fat'])//100000][np.abs(gal['fat'])%100000 - 1]
                    From = prog['from']
                    # gal['from'] = From

                if (gal['son']>0):
                    desc = inst[gal['son']//100000][gal['son']%100000 - 1]
                    if np.abs(desc['fat']) == gal2id(gal):
                        desc['from'] = From
                gal['from'] = From
    
        logger.info("Recover catalogue...")
        gals = None
        for iout in p.nout:
            iinst = inst[iout]
            gals = iinst if gals is None else np.hstack((gals, iinst))
    
        logger.info("Find fragmentation...")
        uniqs = np.unique(gals['from'])
        feedback = []
        for uniq in uniqs:
            first = gals[gals['from'] == uniq][-1]
            if len(first['prog'])>0:
                prog, score = maxprog(first)
                pgal = gethalo(prog, halos=inst)
                pfirst = gals[gals['from'] == pgal['from']][-1]
                if pgal['last'] == first['last']:
                    feedback.append( (uniq, first['last'], -pfirst['from'], pfirst['last']) )
                else:
                    if pfirst['last']//100000 < first['timestep']:
                        feedback.append( (uniq, first['last'], pfirst['from'], first['last']) )
                        feedback.append( (pfirst['from'], pfirst['last'], pfirst['from'], first['last']) )
                    else:
                        feedback.append( (uniq, first['last'], -pfirst['from'], pfirst['last']) )
        From = np.copy(gals['from'])
        Last = np.copy(gals['last'])
        for feed in feedback:
            From[(From==feed[0])&(Last==feed[1])] = feed[2]
            Last[(From==feed[0])&(Last==feed[1])] = feed[3]
        gals['from'] = From
        gals['last'] = Last
        pklsave(gals, f"{p.resultdir}/{p.logprefix}stable.pickle", overwrite=True)
        logger.info(f"`{p.resultdir}/{p.logprefix}stable.pickle` saved\n")



####################################################################################################################
# Debug and Log
####################################################################################################################
def name_log(repo, name, prefix=None):
    if prefix is None: prefix='ytree_'
    count = 0
    fname = f"{repo}/{prefix}{name}_{count}.log"
    while os.path.isfile(fname):
        count+=1
        fname = f"{repo}/{prefix}{name}_{count}.log"
    return fname

def make_log(repo:str, name:str, path_in_repo:str='YoungTree', prefix:str=None, detail:bool=False) -> tuple[logging.Logger, str]:
    resultdir = f"{repo}/{path_in_repo}"
    if not os.path.isdir(resultdir): os.mkdir(resultdir)
    fname = name_log(resultdir, name, prefix=prefix)

    logger_file_handler = RotatingFileHandler(fname, mode='a')
    if detail: logger_file_handler.setLevel(logging.DEBUG)
    else: logger_file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(u'%(asctime)s [%(levelname)8s] %(message)s')
    logger_file_handler.setFormatter(formatter)

    logging.captureWarnings(True)

    root_logger = logging.getLogger(fname)
    warnings_logger = logging.getLogger("py.warnings")
    root_logger.handlers = []
    warnings_logger.handlers = []
    root_logger.addHandler(logger_file_handler)
    warnings_logger.addHandler(logger_file_handler)
    root_logger.setLevel(logging.DEBUG)
    root_logger.info("Debug Start")
    root_logger.propagate=False
    return root_logger, resultdir

class memory_tracker():
    __slots__ = ['ref', 'prefix', 'logger']
    def __init__(self, prefix:str, logger:logging.Logger):
        self.ref = MB()
        self.prefix = prefix
        self.logger = logger
    
    def done(self, cut=50):
        new = MB() - self.ref
        if self.logger is not None:
            if new > cut: self.logger.debug(f"{self.prefix} Gain {new:.2f} MB")
            elif new < -cut: self.logger.debug(f"{self.prefix} Loss {new:.2f} MB")
        else:
            if new > cut: print(f"{self.prefix} Gain {new:.2f} MB")
            elif new < -cut: print(f"{self.prefix} Loss {new:.2f} MB")
        self.ref = MB()

class timer():
    __slots__ = ['ref', 'units', 'corr', 'unit', 'text', 'verbose', 'logger', 'level']
    def __init__(self, unit:str="sec",text:str="", verbose:int=2, logger:logging.Logger=None, level:str='info'):
        self.ref = time.time()
        self.units = {"ms":1/1000, "sec":1, "min":60, "hr":3600}
        self.corr = self.units[unit]
        self.unit = unit
        self.text = text
        self.verbose=verbose
        self.logger=logger
        self.level = level
        
        if self.logger is not None:
            if self.level == 'info': self.logger.info(f"{text} START")
            else: self.logger.debug(f"{text} START")
        else: print(f"{text} START")
    
    def done(self, add=None):
        elapse = time.time()-self.ref
        if add is not None: self.text = f"{self.text} {add}"
        if self.logger is not None:
            if self.level == 'info': self.logger.info(f"{self.text} Done ({elapse/self.corr:.3f} {self.unit})")
            else: self.logger.debug(f"{self.text} Done ({elapse/self.corr:.3f} {self.unit})")
        else: print(f"{self.text} Done ({elapse/self.corr:.3f} {self.unit})")

import threading
class DisplayCPU(threading.Thread):
    def run(self):
        self.running = True
        currentProcess = psutil.Process()
        while self.running and self.iq<1000:
            if self.iq < 100:
                self.queue[self.iq] = currentProcess.cpu_percent(interval=0.5)
            else:
                self.queue[np.argmin(self.queue)] = max(currentProcess.cpu_percent(interval=1), self.queue[np.argmin(self.queue)])
            self.iq += 1

    def stop(self):
        self.running = False


class DebugDecorator(object):
    def __init__(self, f, params:DotDict=None, logger:logging.Logger=None, ontime=True, onmem=True, oncpu=True):
        self.func = f
        self.ontime=ontime
        self.onmem=onmem
        self.oncpu=oncpu
        self.logger = logger
        self.verbose = params.verbose

    def __call__(self, *args, **kwargs):
        prefix = kwargs.pop("prefix","")
        prefix = f"{prefix}[{self.func.__name__}] "
        if self.ontime:
            self.clock=timer(text=prefix, logger=self.logger, verbose=self.verbose)
        if self.onmem:
            self.mem = MB()
        if self.oncpu:
            self.cpu = DisplayCPU()
            self.cpu.iq = 0
            self.cpu.queue = np.zeros(100)-1
            self.cpu.start()
        self.func(*args, **kwargs)
        if self.ontime:
            self.clock.done()
        if self.onmem:
            self.logger.info(f"{prefix}  mem ({MB() - self.mem:.2f} MB) [{self.mem:.2f} -> {MB():.2f}]")
        if self.oncpu:
            self.cpu.stop()
            icpu = self.cpu.queue[self.cpu.queue >= 0]
            if len(icpu)>0:
                q16, q50, q84 = np.percentile(icpu, q=[16,50,84])
            else:
                q16 = q50 = q84 = 0
            self.logger.info(f"{prefix}  cpu ({q50:.2f} %) [{q16:.1f} ~ {q84:.1f}]")

def debugf(logger:logging.Logger, params:DotDict, ontime=True, onmem=True, oncpu=True):
    def _debug(function):
        return DebugDecorator(function, params=params, logger=logger, ontime=ontime, onmem=onmem, oncpu=oncpu)
    return _debug