from ytool import *
import os
import logging
from logging.handlers import RotatingFileHandler
import psutil
import time
import traceback
from tqdm import tqdm
import sys
from numba import set_num_threads
from numpy.lib.recfunctions import append_fields, drop_fields

####################################################################################################################
# Main run
####################################################################################################################
def treerecord(iout:int, nout:int, elapse_s:float, total_elapse_s:float, logger:logging.Logger):
    '''
    Current status report on logger  
    
    Example
    -------
    180 done (4.84 min elapsed)\n
    8/187 done (5.64 min/snap)\n
    1.24 hr forecast\n
    43.53 GB used\n
    '''
    key, conv = timeconv(elapse_s)
    a = f"{iout} done ({conv:.2f} {key} elapsed)"
    logger.info(a)
    aver = total_elapse_s/len(nout[nout>=iout])
    key, conv = timeconv(aver)
    a = f"{len(nout[nout>=iout])}/{len(nout)} done ({conv:.2f} {key}/snap)"
    logger.info(a)
    key, conv = timeconv(aver*len(nout[nout<iout]))
    a = f"{conv:.2f} {key} forecast"
    logger.info(a)
    a = f"{psutil.Process().memory_info().rss / 2 ** 30:.4f} GB used\n" # memory used
    logger.info(a)
    
# from yroot import TreeBase # Remove this when running!
def do_onestep(Tree:'TreeBase', iout:int, fout:int, reftot:float=time.time()):
    nout = Tree.p.nout
    nstep = Tree.p.nstep
    resultdir = Tree.p.resultdir
    set_num_threads(Tree.p.ncpu)
    time_record = []
    try:
        ref = time.time()
        skip = False
        istep = Tree.out2step(iout)
        logname = Tree.mainlog.name
        Tree.mainlog = follow_log(logname, detail=Tree.p.detail)
        # Fully saved
        if os.path.exists(f"{resultdir}/by-product/{Tree.p.logprefix}{iout:05d}.pickle"):
            if(not Tree.p.takeover):
                Tree.mainlog.warning(f"! No takeover ! Remove `{resultdir}/by-product/{Tree.p.logprefix}{iout:05d}.pickle`")
                os.remove(f"{resultdir}/by-product/{Tree.p.logprefix}{iout:05d}.pickle")
            else:
                Tree.mainlog.info(f"[Queue] {iout} is done --> Skip\n")
                skip=True
        
    
        # Main process
        if not skip:
            # New log file
            t0 = time.time()
            Tree.mainlog.info(f"[Queue] {iout} start")
            Tree.logger, _, newlog = make_log(Tree.p.repo, f"{iout:05d}", detail=Tree.p.detail, prefix=Tree.p.logprefix, path_in_repo=f"{Tree.p.path_in_repo}/log")
            Tree.mainlog.info(f"See `{newlog}`\n")
            time_record.append(["Making New Log", time.time()-t0]); t0 = time.time()
            
            # Load snap gal part
            Tree.logger.info(f"\n\nStart at iout={iout}\n")
            if(Tree.leaves['i']!={}):
                Tree.logger.info(f"[slot i] {Tree.outs['i']} -> {iout}")
                Tree.write_leaves('i', level='info')
                Tree.leaves['i']={}
                time_record.append([f"Write leaves at {Tree.outs['i']}", time.time()-t0]); t0 = time.time()
            Tree.outs['i'] = iout
            Tree.read_leaves('i', level='info')
            time_record.append([f"Read leaves at {Tree.outs['i']}", time.time()-t0]); t0 = time.time()
            Tree.update_debugger('i')
            Tree.logger.info(f"\n{Tree.summary()}\n")
            Tree.logger.info("\n----------------\nFind progenitors\n----------------\n")
            
            # Find progenitors
            for j in range(Tree.p.nsnap):
                jstep = istep-j-1
                if jstep > 0:
                    jout = Tree.step2out(jstep)
                    Tree.logger.info(f"\n\nProgenitor at jout={jout}\n")
                    if(Tree.leaves['j']!={}):
                        Tree.logger.info(f"[slot j] {Tree.outs['j']} -> {jout}")
                        Tree.write_leaves('j', level='info')
                        Tree.leaves['j']={}
                        time_record.append([f"Write leaves at {Tree.outs['j']}", time.time()-t0]); t0 = time.time()
                    Tree.outs['j'] = jout
                    Tree.read_leaves('j', level='info')
                    time_record.append([f"Read leaves at {Tree.outs['j']}", time.time()-t0]); t0 = time.time()
                    Tree.update_debugger('j')
                    Tree.logger.info(f"\n{Tree.summary()}\n")
                    Tree.logger.info("\n")
                    Tree.find_cands(level='info')
                    time_record.append([f"Find candidates {Tree.outs['i']}<->{Tree.outs['j']}", time.time()-t0]); t0 = time.time()
                    Tree.flush(jout, level='info')
            Tree.write_leaves('j', level='info')
            time_record.append([f"Write leaves at {Tree.outs['j']}", time.time()-t0]); t0 = time.time()
            Tree.leaves['j']={}
            
            Tree.logger.info("\n----------------\nFind descendants\n----------------\n")
            # Find descendants
            for j in range(Tree.p.nsnap):
                jstep = istep+j+1
                if jstep <= np.max(nstep):
                    jout = Tree.step2out(jstep)
                    Tree.logger.info(f"\n\nDescendant at jout={jout}\n")
                    if(Tree.leaves['j']!={}):
                        Tree.logger.info(f"[slot j] {Tree.outs['j']} -> {jout}")
                        Tree.write_leaves('j', level='info')
                        Tree.leaves['j']={}
                        time_record.append([f"Write leaves at {Tree.outs['j']}", time.time()-t0]); t0 = time.time()
                    Tree.outs['j'] = jout
                    Tree.read_leaves('j', level='info')
                    time_record.append([f"Read leaves at {Tree.outs['j']}", time.time()-t0]); t0 = time.time()
                    Tree.update_debugger('j')
                    Tree.logger.info(f"\n{Tree.summary()}\n")
                    Tree.logger.info("\n")
                    Tree.find_cands(level='info')
                    time_record.append([f"Find candidates {Tree.outs['i']}<->{Tree.outs['j']}", time.time()-t0]); t0 = time.time()
                    Tree.flush(jout, level='info')
            Tree.logger.info("\n----------------\nFlush Redundants\n----------------\n")
            Tree.logger.info(f"[slot j] Dump {Tree.outs['j']}")
            Tree.write_leaves('j', level='info')
            time_record.append([f"Write leaves at {Tree.outs['j']}", time.time()-t0]); t0 = time.time()
            Tree.leaves['j']={}
            
            # Flush redundant snapshots
            cutstep = istep+Tree.p.nsnap
            if cutstep<=np.max(nstep):
                cutout = Tree.step2out(cutstep)
                outs = Tree.out_on_table
                for out in outs:
                    if out > cutout:
                        if(os.path.exists(f"{resultdir}/by-product/{Tree.p.logprefix}{out:05d}.pickle")):
                            Tree.out_of_use.append(out)
                        else:
                            Tree.finalize(out, level='info')
                            time_record.append([f"Finalize {out}", time.time()-t0]); t0 = time.time()
                        Tree.out_on_table.remove(out)
                        Tree.flush(out, level='info')        
                        Tree.logger.info(f"\n{Tree.summary()}\n")
            Tree.logger.info("\n----------------\nBackup leaf files\n----------------\n")
            # Backup files
            Tree.logger.info(f"[slot i] Dump {Tree.outs['i']}")
            Tree.write_leaves('i', level='info')
            Tree.leaves['i']={}
            time_record.append([f"Write leaves at {Tree.outs['i']}", time.time()-t0]); t0 = time.time()
            Tree.logger.info(f"\n{Tree.summary()}\n")
            treerecord(iout, nout[nout<=fout], time.time()-ref, time.time()-reftot, Tree.mainlog)
    except Exception as e:
        print("\n\n"); Tree.logger.error("\n\n")
        print(traceback.format_exc()); Tree.logger.error(traceback.format_exc())
        print("\n\n"); Tree.logger.error("\n\n")
        print(e); Tree.logger.error(e)
        Tree.logger.error(Tree.summary())
        print("\nIteration is terminated (`do_onestep`)\n"); Tree.logger.error("\nIteration is terminated (`do_onestep`)\n")
        os.remove(f"{Tree.p.resultdir}/{Tree.p.logprefix}success.tmp")
        sys.exit(1)
    
    return time_record
        








# @debugf(ontime=True, onmem=True, oncpu=True)
def gather(p:DotDict, logger:logging.Logger):
    go=True
    if os.path.exists(f"{p.resultdir}/{p.logprefix}all.pickle"):
        ans=input(f"You already have `{p.resultdir}/{p.logprefix}all.pickle`. Ovewrite? [Y/N]")
        go = ans in yess
    if go:
        logger.info("Gather all files...")
        print("Gather all files...")
        for i, iout in enumerate(p.nout):
            brick = pklload(f"{p.resultdir}/by-product/{p.logprefix}{iout:05d}.pickle")
            if(not 'host' in brick.dtype.names):
                field_names = brick.dtype.names
                dtypes = brick.dtype.descr
                insert_index = 3
                new_dtypes = dtypes[:insert_index] + [('host', '<i4')] + dtypes[insert_index:]
                new_brick = np.empty(brick.shape, dtype=new_dtypes)
                for name in field_names:
                    new_brick[name] = brick[name]
                new_brick['host'] = np.zeros(len(brick), dtype=np.int32)
                brick = new_brick
            if i==0:
                gals = brick
            else:
                gals = np.hstack((gals, brick))
        # logger.info("Add column `last`...")
        # temp = np.zeros(len(gals), dtype=np.int32)
        
        logger.info("Convert `prog` and `desc` to easier format...")
        print("Convert `prog` and `desc` to easier format...")
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
        print(f"`{p.resultdir}/{p.logprefix}all.pickle` saved\n")
    # gals = pklload(f"{p.resultdir}/{p.logprefix}all.pickle")


def connect(p:DotDict, logger:logging.Logger):
    gals = pklload(f"{p.resultdir}/{p.logprefix}all.pickle")
    complete = True
    for iout in p.nout:
        temp = gals[gals['timestep']==iout]
        if(len(temp)!=np.max(temp['id'])):
            complete = False
    go=True
    if os.path.exists(f"{p.resultdir}/{p.logprefix}stable.pickle"):
        ans=input(f"You already have `{p.resultdir}/{p.logprefix}stable.pickle`. Ovewrite? [Y/N]")
        go = ans in yess
    if go:
        logger.info("Make dictionary from catalogue...")
        if('last' in gals.dtype.names): gals = drop_fields(gals, 'last')
        if('from' in gals.dtype.names): gals = drop_fields(gals, 'from')
        if('root' in gals.dtype.names): gals = drop_fields(gals, 'root')
        if('final' in gals.dtype.names): gals = drop_fields(gals, 'final')
        ngal = len(gals)
        gals = append_fields(gals, "fat", np.zeros(ngal, dtype=np.int32), usemask=False, asrecarray=False)
        gals = append_fields(gals, "son", np.zeros(ngal, dtype=np.int32), usemask=False, asrecarray=False)
        gals = append_fields(gals, "fat_score", np.zeros(ngal, dtype=np.float64), usemask=False, asrecarray=False)
        gals = append_fields(gals, "son_score", np.zeros(ngal, dtype=np.float64), usemask=False, asrecarray=False)
        gals = append_fields(gals, "first", np.zeros(ngal, dtype=np.int32), usemask=False, asrecarray=False)
        gals = append_fields(gals, "from", np.zeros(ngal, dtype=np.int32), usemask=False, asrecarray=False)
        gals = append_fields(gals, "last", np.zeros(ngal, dtype=np.int32), usemask=False, asrecarray=False)
        gals = append_fields(gals, "final", np.zeros(ngal, dtype=np.int32), usemask=False, asrecarray=False)
        inst = {}
        for iout in p.nout:
            inst[iout] = gals[gals['timestep']==iout]
        gals = None

        logger.info("Find son & father (direct match)...")
        print("Find son & father (direct match)...")
        nfindson = 0
        nfindfat = 0
        offsets = np.arange(1, 1+p.nsnap)
        for offset in offsets:
            logger.info(f"\n\n  OFFSET={offset}\n")
            iterobj = p.nout
            iterobj = tqdm(iterobj, desc=f"offset={offset}")
            for iout in iterobj:
                dstep = out2step(iout, p.nout, p.nstep)+offset
                if(dstep in p.nstep):
                    igals = inst[iout]
                    for ihalo_iout in igals:
                        # For halos who don't have confirmed son
                        if(ihalo_iout['son']<=0):
                            iid = gal2id(ihalo_iout)
                            idesc, idscore = maxdesc(ihalo_iout, all=False, offset=offset, nout=p.nout, nstep=p.nstep)
                            if(idesc==0):
                                logger.debug(f"\t{iid} No desc")
                                pass
                            else:
                                dhalo = gethalo(idesc//100000, idesc%100000, halos=inst, complete=complete) # desc of ihalo_iout
                                prog, pscore = maxprog(dhalo, all=False, offset=offset, nout=p.nout, nstep=p.nstep)  # prog of desc of ihalo_iout
                                # Choose each other (prog <-> desc)
                                if(prog==iid)&(dhalo['fat']<=0):
                                    ihalo_iout['son'] = idesc
                                    ihalo_iout['son_score'] = idscore
                                    logger.debug(f"\t{iid} newly has son {idesc}({idscore:.2f})")
                                    dhalo['fat'] = prog
                                    dhalo['fat_score'] = pscore
                                    logger.debug(f"\tAlso, {idesc} newly has father {prog}({pscore:.2f})")
                                    nfindson += 1
                                    nfindfat += 1
        logger.info(f" > {nfindson} gals found son & {nfindfat} gals found fat (of {ngal:.2f})\n")
        print(f" > {nfindson} gals found son & {nfindfat} gals found fat (of {ngal:.2f})\n")
        
        logger.info("Find son again (indirect match)...")
        print("Find son again (indirect match)...")
        iterobj = tqdm(p.nout)
        for iout in iterobj:
            igals = inst[iout]
            for ihalo_iout in igals:
                if(ihalo_iout['son']>0):
                    continue
                iid = gal2id(ihalo_iout)
                idesc, idscore = maxdesc(ihalo_iout, all=True, offset=offset, nout=p.nout, nstep=p.nstep)
                if(idesc==0):
                    logger.debug(f"\t{iid} No desc")
                    pass
                else:
                    dhalo = gethalo(idesc//100000, idesc%100000, halos=inst, complete=complete)
                    if(iid in dhalo['prog']):
                        ihalo_iout['son'] = -idesc
                        ihalo_iout['son_score'] = idscore
                        logger.debug(f"\t{iid} changes son {ihalo_iout['son']}({ihalo_iout['son_score']:.2f}) -> {idesc}({idscore:.2f})")
                        if(dhalo['fat']>0):
                            pass
                        elif(dhalo['fat']==0):
                            pscore = dhalo['prog_score'][np.where(dhalo['prog']==iid)[0][0]]
                            logger.debug(f"\tAlso, {idesc} changes father {dhalo['fat']}({dhalo['fat_score']:.2f}) -> {iid}({pscore:.2f})")
                            dhalo['fat'] = iid
                            dhalo['fat_score'] = pscore
                        else:
                            pscore = dhalo['prog_score'][np.where(dhalo['prog']==iid)[0][0]]
                            if(dhalo['fat_score'] < pscore):
                                logger.debug(f"\tAlso, {idesc} changes father {dhalo['fat']}({dhalo['fat_score']:.2f}) -> {iid}({pscore:.2f})")
                                dhalo['fat'] = iid
                                dhalo['fat_score'] = pscore
        
        
        logger.info("Find father again (indirect match)...")
        print("Find father again (indirect match)...")
        iterobj = tqdm(p.nout)
        for iout in iterobj:
            igals = inst[iout]
            for ihalo_iout in igals:
                if(ihalo_iout['fat']>0):
                    continue
                iid = gal2id(ihalo_iout)
                iprog, ipscore = maxprog(ihalo_iout, all=True, offset=offset, nout=p.nout, nstep=p.nstep)
                if(iprog==0):
                    logger.debug(f"\t{iid} No prog")
                    pass
                else:
                    phalo = gethalo(iprog//100000, iprog%100000, halos=inst, complete=complete)
                    if(iid in phalo['desc']):
                        ihalo_iout['fat'] = -iprog
                        ihalo_iout['fat_score'] = ipscore
                        logger.debug(f"\t{iid} changes father {ihalo_iout['fat']}({ihalo_iout['fat_score']:.2f}) -> {iprog}({ipscore:.2f})")
                        if(phalo['son']>0):
                            pass
                        elif(phalo['son']==0):
                            dscore = phalo['desc_score'][np.where(phalo['desc']==iid)[0][0]]
                            logger.debug(f"\tAlso, {iprog} changes son {phalo['son']}({phalo['son_score']:.2f}) -> {iid}({dscore:.2f})")
                            phalo['son'] = iid
                            phalo['son_score'] = dscore
                        else:
                            dscore = phalo['desc_score'][np.where(phalo['desc']==iid)[0][0]]
                            if(phalo['son_score'] < dscore):
                                logger.debug(f"\tAlso, {iprog} changes son {phalo['son']}({phalo['son_score']:.2f}) -> {iid}({dscore:.2f})")
                                phalo['son'] = iid
                                phalo['son_score'] = dscore

        logger.info("Check the rest...")
        print("Check the rest...")
        iterobj = tqdm(p.nout)
        for iout in iterobj:
            igals = inst[iout]
            for ihalo_iout in igals:
                if(ihalo_iout['son']==0)&(len(ihalo_iout['desc'])>0):
                    iid = gal2id(ihalo_iout)
                    idesc, idscore = maxdesc(ihalo_iout, all=True, offset=offset, nout=p.nout, nstep=p.nstep)
                    ihalo_iout['son'] = -idesc
                    ihalo_iout['son_score'] = idscore
                    logger.debug(f"\t{iid} newly has son{idesc}({idscore:.2f})")
                if(ihalo_iout['fat']==0)&(len(ihalo_iout['prog'])>0):
                    iid = gal2id(ihalo_iout)
                    iprog, ipscore = maxprog(ihalo_iout, all=True, offset=offset, nout=p.nout, nstep=p.nstep)
                    ihalo_iout['fat'] = -iprog
                    ihalo_iout['fat_score'] = ipscore
                    logger.debug(f"\t{iid} newly has fat{iprog}({ipscore:.2f})")

        pklsave(inst, f"{p.resultdir}/{p.logprefix}fatson.pickle", overwrite=True)
        logger.info(f"`{p.resultdir}/{p.logprefix}fatson.pickle` saved\n")    
        print(f"`{p.resultdir}/{p.logprefix}fatson.pickle` saved\n")

def build_branch(p:DotDict, logger:logging.Logger):
    logger.info("Build branches...")
    print("Build branches...")
    inst = pklload(f"{p.resultdir}/{p.logprefix}fatson.pickle")
    complete = True
    for iout in p.nout:
        temp = inst[iout]
        if(len(temp)!=np.max(temp['id'])):
            complete = False
    go=True
    if os.path.exists(f"{p.resultdir}/{p.logprefix}stable.pickle"):
        ans=input(f"You already have `{p.resultdir}/{p.logprefix}stable.pickle`. Ovewrite? [Y/N]")
        go = ans in yess
    if go:
        logger.info("Build branches forward...")
        print("Build branches forward...")
        iterobj = tqdm(p.nout[::-1]) # From first galaxies
        for iout in iterobj:
            igals = inst[iout]
            for ihalo_iout in igals:
                iid = gal2id(ihalo_iout)
                fid = ihalo_iout['fat']
                if(fid==0):
                    ihalo_iout['first'] = iid
                    ihalo_iout['from'] = iid
                else:
                    fhalo = gethalo(np.abs(fid)//100000, np.abs(fid)%100000, halos=inst, complete=complete)
                    ihalo_iout['first'] = fhalo['first']
                    if(fid>0):
                        ihalo_iout['from'] = fhalo['from']
                    else:
                        ihalo_iout['from'] = iid
        
        logger.info("Build branches backward...")
        print("Build branches backward...")
        iterobj = tqdm(p.nout) # From last galaxies
        for iout in iterobj:
            igals = inst[iout]
            for ihalo_iout in igals:
                iid = gal2id(ihalo_iout)
                sid = ihalo_iout['son']
                if(sid==0):
                    ihalo_iout['last'] = iid
                    ihalo_iout['final'] = iid
                else:
                    shalo = gethalo(np.abs(sid)//100000, np.abs(sid)%100000, halos=inst, complete=complete)
                    ihalo_iout['final'] = shalo['final']
                    if(sid>0):
                        ihalo_iout['last'] = shalo['last']
                    else:
                        ihalo_iout['last'] = iid

        gals = None
        logger.info("Collect galaxies...")
        print("Collect galaxies...")
        iterobj = tqdm(p.nout)
        for iout in iterobj:
            iinst = inst[iout]
            gals = iinst if gals is None else np.hstack((gals, iinst))
        pklsave(gals, f"{p.resultdir}/{p.logprefix}stable.pickle", overwrite=True)
        logger.info(f"`{p.resultdir}/{p.logprefix}stable.pickle` saved\n")                                    
        print(f"`{p.resultdir}/{p.logprefix}stable.pickle` saved\n")
                            


# @debugf(ontime=True, onmem=True, oncpu=True)
def connect_legacy(p:DotDict, logger:logging.Logger):
    gals = pklload(f"{p.resultdir}/{p.logprefix}all.pickle")
    complete = True
    for iout in p.nout:
        temp = gals[gals['timestep']==iout]
        if(len(temp)!=np.max(temp['id'])):
            complete = False
    go=True
    if os.path.exists(f"{p.resultdir}/{p.logprefix}stable.pickle"):
        ans=input(f"You already have `{p.resultdir}/{p.logprefix}stable.pickle`. Ovewrite? [Y/N]")
        go = ans in yess
    if go:
        if not (os.path.exists(f"{p.resultdir}/{p.logprefix}stage_4.pickle")):
            if not (os.path.exists(f"{p.resultdir}/{p.logprefix}stage_3.pickle")):
                if not (os.path.exists(f"{p.resultdir}/{p.logprefix}stage_2.pickle")):
                    if not (os.path.exists(f"{p.resultdir}/{p.logprefix}stage_1.pickle")):
                        logger.info("Make dictionary from catalogue...")
                        gals = append_fields(gals, "from", np.zeros(len(gals), dtype=np.int32), usemask=False)
                        gals = append_fields(gals, "fat", np.zeros(len(gals), dtype=np.int32), usemask=False)
                        gals = append_fields(gals, "son", np.zeros(len(gals), dtype=np.int32), usemask=False)
                        gals = append_fields(gals, "fat_score", np.zeros(len(gals), dtype=np.float64), usemask=False)
                        gals = append_fields(gals, "son_score", np.zeros(len(gals), dtype=np.float64), usemask=False)
                        gals = append_fields(gals, "merged", np.zeros(len(gals), dtype=np.int8), usemask=False)
                        inst = {}
                        for iout in p.nout:
                            inst[iout] = gals[gals['timestep']==iout]
                        gals = None

                        logger.info("Find son & father...")
                        offsets = np.arange(1, 1+p.nsnap)
                        for offset in offsets:
                            logger.info(f"\n\n  OFFSET={offset}\n")
                            iterobj = p.nout
                            for iout in iterobj:
                                dstep = out2step(iout, p.nout, p.nstep)+offset
                                if dstep in p.nstep:
                                    igals = inst[iout]
                                    do = np.ones(np.max(igals['id']), dtype=bool)
                                    for ihalo_iout in igals:
                                        do[ihalo_iout['id']-1] = False
                                        # For halos who don't have confirmed son
                                        if ihalo_iout['son']<=0:
                                            iid = gal2id(ihalo_iout)
                                            idesc, idscore = maxdesc(ihalo_iout, all=False, offset=offset, nout=p.nout, nstep=p.nstep)
                                            if idesc==0:
                                                logger.debug(f"\t{iid} No desc")
                                                pass
                                            else:
                                                dhalo = gethalo(idesc//100000, idesc%100000, halos=inst, complete=complete) # desc of ihalo_iout
                                                prog, pscore = maxprog(dhalo, all=False, offset=offset, nout=p.nout, nstep=p.nstep)  # prog of desc of ihalo_iout
                                                # Choose each other (prog <-> desc)
                                                if prog==iid:
                                                    nrival = 0
                                                    for jhalo_iout, ido in zip(igals, do):
                                                        # Loop only for those not yet found
                                                        if(ido):
                                                            # Loop only rivals
                                                            if (idesc in jhalo_iout['desc']):
                                                                # Loop only rival who doesn't have confirmed son
                                                                if (jhalo_iout['son']<=0):
                                                                    jid = gal2id(jhalo_iout)
                                                                    if jid != iid:
                                                                        jdesc, jdscore = maxdesc(jhalo_iout, all=False, offset=offset, nout=p.nout, nstep=p.nstep)
                                                                        # Loop only rival who has the same max-desc
                                                                        if jdesc==idesc:
                                                                            do[jhalo_iout['id']-1] = False
                                                                            nrival += 1
                                                                            # jhalo hasn't been accessed
                                                                            if jhalo_iout['son'] == 0:
                                                                                logger.debug(f"\t\trival {jid} newly have son {-idesc} ({jdscore:.4f})")
                                                                                jhalo_iout['son'] = -idesc
                                                                                jhalo_iout['son_score'] = jdscore
                                                                            # jhalo has hesitated son
                                                                            elif jhalo_iout['son'] < 0:
                                                                                if jhalo_iout['son_score'] > jdscore:
                                                                                    logger.debug(f"\t\trival {jid} keep original son {jhalo_iout['son']} ({jhalo_iout['son_score']:.4f}) rather than {-idesc} ({jdscore:.4f})")
                                                                                else:
                                                                                    logger.debug(f"\t\trival {jid} change original son {jhalo_iout['son']} ({jhalo_iout['son_score']:.4f}) to {-jdesc} ({jdscore:.4f})")
                                                                                    jhalo_iout['son'] = -jdesc
                                                                                    jhalo_iout['son_score'] = jdscore
                                                                            # jhalo has confirmed son
                                                                            else:
                                                                                logger.debug(f"\t\trival {jid} keep original son {jhalo_iout['son']} ({jhalo_iout['son_score']:.4f}) rather than {-idesc} ({jdscore:.4f})")
                                                    # ihalo hasn't been accessed
                                                    if ihalo_iout['son'] == 0:
                                                        logger.debug(f"\t{iid} change original son {ihalo_iout['son']} ({ihalo_iout['son_score']:.4f}) to {idesc} ({idscore:.4f})")
                                                        ihalo_iout['son'] = idesc
                                                        ihalo_iout['son_score'] = idscore
                                                    # ihalo has been accessed
                                                    else:
                                                        if ihalo_iout['son_score'] > idscore:
                                                            logger.debug(f"\t {iid} keep original son {ihalo_iout['son']} ({ihalo_iout['son_score']:.4f}) rather than {idesc} ({idscore:.4f})")
                                                        else:
                                                            logger.debug(f"\t{iid} change original son {ihalo_iout['son']} ({ihalo_iout['son_score']:.4f}) to {idesc} ({idscore:.4f})")
                                                            ihalo_iout['son'] = idesc
                                                            ihalo_iout['son_score'] = idscore
                                                            
                                                    # dhalo hasn't been accessed
                                                    if dhalo['fat'] == 0:
                                                        logger.debug(f"\tAlso, son {idesc} have father {iid} with {pscore:.4f}")
                                                        dhalo['fat'] = iid
                                                        dhalo['fat_score'] = pscore
                                                    # dhalo has been accessed
                                                    elif(dhalo['fat'] < 0):
                                                        if dhalo['fat_score'] > pscore:
                                                            logger.debug(f"\tHowever, son {idesc} keep original father {dhalo['fat']} ({dhalo['fat_score']:.4f}) rather than {iid} ({pscore:.4f})")
                                                        else:
                                                            logger.debug(f"\tAlso, son {idesc} change original father {dhalo['fat']} ({dhalo['fat_score']:.4f}) to {iid} ({pscore:.4f})")
                                                            dhalo['fat'] = iid
                                                            dhalo['fat_score'] = pscore
                                                    else:
                                                        logger.debug(f"\tAlso, son {idesc} keep original father {dhalo['fat']} ({dhalo['fat_score']:.4f}) rather than {iid} ({pscore:.4f})")
                                                # Not choose each other (prog <-/-> desc)
                                                else:
                                                    # ihalo_iout, "my desc is `dhalo`!"
                                                    # dhalo, "No my prog is other `prog`!"
                                                    logger.debug(f"\t{iid} has desc {idesc}, but his prog is {prog}")
                                                    if(ihalo_iout['son_score'] < idscore):
                                                        logger.debug(f"\t\t{iid} change original son {ihalo_iout['son']} ({ihalo_iout['son_score']:.4f}) to {-idesc} ({idscore:.4f})")
                                                        ihalo_iout['son'] = -idesc
                                                        ihalo_iout['son_score'] = idscore
                                                    if(dhalo['fat_score'] < pscore):
                                                        logger.debug(f"\t\t{idesc} change original fat {dhalo['fat']} ({dhalo['fat_score']:.4f}) to {-prog} ({pscore:.4f})")
                                                        dhalo['fat'] = -prog
                                                        dhalo['fat_score'] = pscore
                        pklsave(inst, f"{p.resultdir}/{p.logprefix}stage_1.pickle", overwrite=True)
                        logger.info(f"`{p.resultdir}/{p.logprefix}stage_1.pickle` saved\n")                                    
                    inst = pklload(f"{p.resultdir}/{p.logprefix}stage_1.pickle")

                    logger.info("Connect same Last...")
                    for iout in p.nout:
                        if(not 'merged' in inst[iout].dtype.names):
                            inst[iout] = append_fields(inst[iout], "merged", np.zeros(len(inst[iout]), dtype=np.int8), usemask=False)
                        gals = inst[iout]
                        for gal in gals:
                            last = gal2id(gal) if(gal['last'] == 0) else gal['last']
                            if(gal['son'] != 0):
                                desc = gethalo(np.abs(gal['son']), halos=inst, complete=complete)
                                last = desc['last']
                                gal['merged'] = 1 if(gal['son'] < 0) else 0
                                
                            gal['last'] = last
                            if(gal['fat'] > 0):
                                prog = gethalo(gal['fat'], halos=inst, complete=complete)
                                if(np.abs(prog['son']) == gal2id(gal)):
                                    prog['last'] = last
                    pklsave(inst, f"{p.resultdir}/{p.logprefix}stage_2.pickle", overwrite=True)
                    logger.info(f"`{p.resultdir}/{p.logprefix}stage_2.pickle` saved\n")                                    
                inst = pklload(f"{p.resultdir}/{p.logprefix}stage_2.pickle")

                logger.info("Connect same From...")
                for iout in p.nout[::-1]:
                    gals = inst[iout]
                    assert 'merged' in gals.dtype.names
                    for gal in gals:
                        From = gal2id(gal) if(gal['from'] == 0) else gal['from']
                        if (gal['fat'] > 0):
                            prog = gethalo(np.abs(gal['fat']), halos=inst, complete=complete)
                            From = prog['from']
                        elif(gal['fat'] < 0):
                            prog = gethalo(np.abs(gal['fat']), halos=inst, complete=complete)
                            From = -np.abs(prog['from'])
                            gal['merged'] = -1

                        gal['from'] = From
                        if (gal['son']>0):
                            desc = gethalo(gal['son'], halos=inst, complete=complete)
                            if np.abs(desc['fat']) == gal2id(gal):
                                desc['from'] = From
                pklsave(inst, f"{p.resultdir}/{p.logprefix}stage_3.pickle", overwrite=True)
                logger.info(f"`{p.resultdir}/{p.logprefix}stage_3.pickle` saved\n")                                    
            inst = pklload(f"{p.resultdir}/{p.logprefix}stage_3.pickle")

            logger.info("Recover catalogue from dictionary...")
            gals = None
            for iout in p.nout:
                iinst = inst[iout]
                gals = iinst if gals is None else np.hstack((gals, iinst))
            pklsave(gals, f"{p.resultdir}/{p.logprefix}stage_4.pickle", overwrite=True)
            logger.info(f"`{p.resultdir}/{p.logprefix}stage_4.pickle` saved\n")                                    
        gals = pklload(f"{p.resultdir}/{p.logprefix}stage_4.pickle")
        inst = pklload(f"{p.resultdir}/{p.logprefix}stage_3.pickle")

        logger.info("Find fragmentation...")
        # 1) Pick up each branch based on `from`
        # 2) Find their "first" leaf
        # 3) If "first" leaf has progenitors, check the connection
        Froms = np.unique(gals['from'])
        feedback = [] # (From_now, Last_now, Status_now, From_wannabe, Last_wannabe, Status_wannabe)
        for From in Froms:
            first = gals[gals['from'] == From][-1]
            if len(first['prog'])>0:
                prog, score = maxprog(first, nout=p.nout, nstep=p.nstep)
                pgal = gethalo(prog, halos=inst, complete=complete)
                # first and pfirst are not connected
                pfirst = gals[gals['from'] == pgal['from']][-1]
                # However, they have same last
                if pgal['last'] == first['last']:
                    ### (i) is fragmented from (p)
                    feedback.append( (From, first['last'], first['merged'], -np.abs(pfirst['from']), pfirst['last'], -1) )
                # They have different last
                else:
                    # pfirst is broken before first --> connect two branches
                    # (p) [pfrom]---------[plast]
                    # (i)                         [ifrom]------[ilast]
                    ### (i) is descendant of (p)
                    if pfirst['last']//100000 < first['timestep']:
                        imerged = first['merged']
                        pmerged = pfirst['merged']
                        if(pmerged<0):
                            merged = -1
                        else:
                            if(imerged<0):
                                merged = pmerged
                            else:
                                merged = imerged
                        feedback.append( (From, first['last'], first['merged'], np.abs(pfirst['from']), first['last'], merged) )
                        if(From==-17500482): print("#2", feedback[-1])
                        feedback.append( (pfirst['from'], pfirst['last'], pfirst['merged'], np.abs(pfirst['from']), first['last'], merged) )
                        if(pfirst['from']==-17500482): print("#3", feedback[-1])
                    # pfirst is broken after first
                    # (p) [pfrom]----------------------[plast]
                    # (i)            [ifrom]---------------[ilast]
                    ### (i) is fragmented from (p)
                    else:
                        feedback.append( (From, first['last'], first['merged'], -np.abs(pfirst['from']), first['last'], -1) )
                        if(From==-17500482): print("#4", feedback[-1])
        From = np.copy(gals['from'])
        Last = np.copy(gals['last'])
        Merg = np.copy(gals['merged'])
        for feed in feedback:
            ind = (From==feed[0])&(Last==feed[1])
            if(True in ind):
                From[ind] = feed[3]
                Last[ind] = feed[4]
                Merg[ind] = feed[5]
        From_old = [irow[0] for irow in feedback]
        Last_old = [irow[1] for irow in feedback]
        mask1 = np.isin(From, From_old)
        mask2 = np.isin(Last, Last_old)
        mask = mask1&mask2
        ncall = 1
        while(True in mask):
            for feed in feedback:
                ind = (From==feed[0])&(Last==feed[1])
                if(True in ind):
                    From[ind] = feed[3]
                    Last[ind] = feed[4]
                    Merg[ind] = feed[5]
            ncall+=1
            if(ncall>10):
                logger.warning("Too many calls to find fragmentation. Stop.")
                break
            
        gals['from'] = From
        gals['last'] = Last
        gals['merged'] = Merg
        pklsave(gals, f"{p.resultdir}/{p.logprefix}stable.pickle", overwrite=True)
        logger.info(f"`{p.resultdir}/{p.logprefix}stable.pickle` saved\n")




####################################################################################################################
# Debug and Log
####################################################################################################################
def name_log(repo, name, prefix=None):
    if prefix is None: prefix='ytree_'
    count = 0
    fname = f"{repo}/{prefix}{name}_{count}.log"
    while os.path.exists(fname):
        count+=1
        fname = f"{repo}/{prefix}{name}_{count}.log"
    return fname

def make_log(repo:str, name:str, path_in_repo:str='YoungTree', prefix:str=None, detail:bool=False) -> tuple[logging.Logger, str, str]:
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
    return root_logger, resultdir, fname

def follow_log(fname:str, detail:bool=False):
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
    root_logger.info("Restart logging")
    root_logger.propagate=False
    return root_logger

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
    __slots__ = ['ref', 'units', 'corr', 'unit', 'text', 'verbose', 'logger', 'level', 'mint']
    def __init__(self, unit:str="sec",text:str="", verbose:bool=True, logger:logging.Logger=None, level:str='info', mint:float=0):
        self.ref = time.time()
        self.units = {"ms":1/1000, "sec":1, "min":60, "hr":3600}
        self.corr = self.units[unit]
        self.unit = unit
        self.text = text
        self.verbose=verbose
        self.logger=logger
        self.level = level
        self.mint = mint
        if(self.verbose):
            if self.logger is not None:
                if self.level == 'info': self.logger.info(f"{text} START")
                else: self.logger.debug(f"{text} START")
            else: print(f"{text} START")
    
    def done(self, add=None):
        if(self.verbose):
            elapse = time.time()-self.ref
            if(elapse>=self.mint):
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

def debugf(params:DotDict, logger:logging.Logger, ontime=True, onmem=True, oncpu=True):
    def _debug(function):
        return DebugDecorator(function, params=params, logger=logger, ontime=ontime, onmem=onmem, oncpu=oncpu)
    return _debug
