from rur import uri, uhmi
uri.timer.verbose=0
import numpy as np
import functools
from collections import defaultdict
from collections.abc import Iterable
from multiprocessing import Pool, cpu_count, shared_memory
import multiprocessing as mp
import copy
import logging
import traceback
import sys
import gc
from ytool import *
from yrun import *

def _debug(func):
    @functools.wraps(func)
    def wrapper(self:'TreeBase', *args, **kwargs):
        ontime=self.ontime;onmem=self.onmem;oncpu=self.oncpu
        p = self.p
        logger = self.logger
        fprint = self.print
        verblvl:int = kwargs.pop("verbose",0); kwargs["verbose"] = verblvl
        verbose:bool = verblvl <= p.verbose

        prefix = kwargs.pop("prefix","")
        prefix = f"{prefix}[{func.__name__}({verblvl})]"; kwargs["prefix"] = prefix
        level = kwargs.pop("level","debug"); kwargs["level"] = level

        if ontime:
            clock=timer(text=prefix, logger=logger, verbose=verbose, level=level, mint=p.mint)
        if onmem:
            mem = MB()
        if oncpu:
            cpu = DisplayCPU()
            cpu.iq = 0
            cpu.queue = np.zeros(100)-1
            cpu.start()
        result = func(self,*args, **kwargs)
        if ontime:
            clock.done()
        if onmem:
            fprint(f"{prefix}  mem ({MB() - mem:.2f} MB) [{mem:.2f} -> {MB():.2f}]", level=level)
        if oncpu:
            cpu.stop()
            icpu = cpu.queue[cpu.queue >= 0]
            if len(icpu)>0:
                q16, q50, q84 = np.percentile(icpu, q=[16,50,84])
            else:
                q16 = q50 = q84 = 0
            fprint(f"{prefix}  cpu ({q50:.2f} %) [{q16:.1f} ~ {q84:.1f}]", level=level)
        return result
    return wrapper

class TreeBase:
    def __init__(self, params:DotDict, logger:logging.Logger=None, prefix:str=None, **kwargs):
        func = f"[__Treebase__]"; prefix = f"{prefix}{func}"

        self.iniGB = GB()
        self.iniMB = MB()
        self.p = params

        if self.p.galaxy:
            self.partstr = "star"
            self.Partstr = "Star"
            self.galstr = "gal"
            self.Galstr = "GalaxyMaker"
        else:
            self.partstr = "dm"
            self.Partstr = "DM"
            self.galstr = "halo"
            self.Galstr = "HaloMaker"
        
        self.mainlog:logging.Logger = logger
        self.logger:logging.Logger = self.mainlog


        self.dict_snap = {} # in {iout}, RamsesSnapshot object
        self.dict_gals = {} # in {iout}, galaxy array object
        self.dict_pure = {}
        self.avoid_filter = {}
        self.last_pids = None
        self.outs = {'i':None, 'j':None}
        self.leaves = {'i':{}, 'j':{}}
        self.out_on_table = []

        self.part_halo_match = {} # in {iout}, list of int
        self.banned_list = []
        self.out_of_use = []
        self.accesed = []
        gc.collect()

        self.ontime = self.p.ontime
        self.onmem = self.p.onmem
        self.oncpu = self.p.oncpu
        self.memory = GB()
        self.print("TreeBase initialized.", level='info')
    
    def print(self, msg:str, level='debug'):
        if self.logger is None:
            print(msg)
        else:
            if level == 'debug':
                self.logger.debug(msg)
            else:
                self.logger.info(msg)

    def load_snap(self, iout:int, prefix:str="", level='info', verbose=0)->uri.RamsesSnapshot:
        if not iout in self.dict_snap.keys():
            @_debug
            def _load_snap(self, iout:int, prefix:str="", level='info', verbose=verbose):
                if(not iout in self.accesed): self.accesed.append(iout)
                path_in_repo="" if self.p.mode[0] == '/' else "snapshots"
                snap = uri.RamsesSnapshot(self.p.repo, iout, mode=self.p.rurmode, path_in_repo=path_in_repo)
                snap.shmprefix = "YoungTree"
                if(self.p.loadall)and(not iout in self.banned_list):
                    if(self.p.mode[0]=='y')or(self.p.mode=='nh'):
                        snap.get_part(pname=self.partstr, python=(not self.p.usefortran), nthread=self.p.ncpu, target_fields=["x","y","z","vx","vy","vz","m","id","cpu"])
                    else:
                        snap.get_part(pname=self.partstr, python=(not self.p.usefortran), nthread=self.p.ncpu, target_fields=["x","y","z","vx","vy","vz","m","id","cpu","family"])
                    snap.part.table = snap.part.table[np.argsort(np.abs(snap.part.table['id']))] # <- __copy__
                    snap.part.extra_fields = None
                    del snap.part.extra_fields
                    if(snap.part.ptype != self.partstr): snap.part.ptype = self.partstr
                    snap.part_data = np.array([])
                    snap.flush()
                del snap.cell_extra, snap.part_extra
                self.dict_snap[iout] = snap
                del snap
            _load_snap(self, iout, prefix=prefix, level=level, verbose=verbose+1)
        self.memory = GB()
        return self.dict_snap[iout]

    def load_gals(self, iout:int, galid=None, prefix="", level='info', verbose=0):
        if galid is None:
            galid = 'all'

        # Save
        if not iout in self.dict_gals.keys():
            @_debug
            def _load_gals(self, iout:int, galid=None, prefix="", level='info', verbose=0):
                snap = self.load_snap(iout, prefix=prefix, verbose=verbose+1)
                if(not iout in self.part_halo_match.keys())and(not iout in self.banned_list):
                    @_debug
                    def __load_gals(self, snap:uri.RamsesSnapshot, galid=None, prefix="", level='debug', verbose=1):
                        gm, gpids = uhmi.HaloMaker.load(snap, galaxy=self.p.galaxy, double_precision=self.p.dp, load_parts=True, full_path=self.p.fullpath)
                        if(os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout}_partmatch.pkl")):
                            self.part_halo_match[snap.iout] = pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout}_partmatch.pkl")
                        else:
                            tmp = np.repeat(gm['id'], gm['nparts'])
                            self.part_halo_match[snap.iout] = np.zeros(np.max(gpids), dtype=np.int32)
                            self.part_halo_match[snap.iout][gpids-1] = tmp
                            del tmp
                        del gpids
                        return gm
                    gm = __load_gals(self, snap, galid=galid, prefix=prefix, level='debug', verbose=verbose+1)
                else:
                    gm = uhmi.HaloMaker.load(snap, galaxy=self.p.galaxy, double_precision=self.p.dp, full_path=self.p.fullpath)
                # if not (self.p.galaxy):
                # mask = gm['mcontam']/gm['m'] <= self.p.fcontam
                mask = self.p.filtering(gm)
                self.dict_pure[iout] = mask
                self.dict_gals[iout] = gm
                if(not self.p.default)and(iout == np.max(self.p.nout)):
                    self.last_pids = large_isind(self.part_halo_match[iout], gm[mask]['id'])
                    self.print(f"Not default mode: {len(self.last_pids)} particles are selected.", level="info")
                del snap
            _load_gals(self, iout,galid=galid, prefix=prefix, level=level, verbose=verbose)
            if(not iout in self.avoid_filter.keys()):
                if(iout==np.max(self.p.nout))or(not self.p.light):
                    self.avoid_filter[iout] = self.dict_gals[iout]['id'][self.dict_pure[iout]]
                else:
                    self.avoid_filter[iout] = None
                    if(os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/keys.pickle")):
                        self.avoid_filter[iout] = pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/keys.pickle")
        self.memory = GB()
        # Load
        if isinstance(galid,str):           # 'all'
            if galid=='all':
                return self.dict_gals[iout]
        elif isinstance(galid, Iterable):   # Several galaxies
            return np.hstack([self.dict_gals[iout][ia-1] for ia in galid])
        else:                               # One galaxy
            return self.dict_gals[iout][galid-1]

    def read_part_halo_match(self, iout):
        self.memory = GB()
        if not iout in self.part_halo_match.keys():
            self.load_gals(iout, galid='all')
        if iout in self.part_halo_match.keys():
            if len(self.part_halo_match[iout])>0:
                return self.part_halo_match[iout]
            elif os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout}_partmatch.pkl"):
                return pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout}_partmatch.pkl")
            else:
                raise ValueError(f"Cannot find `part_halo_match` at {iout}!")
        else:
            raise ValueError(f"Cannot find a key `{iout}` in `part_halo_match`!")
    
    @_debug
    def read_leaves(self, iorj:str, prefix="", level='info', verbose=0):
        if(iorj!='i')and(iorj!='j'): raise ValueError(f"iorj must be 'i' or 'j'!")
        
        iout = self.outs[iorj]
        prefix2 = f"[read_leaves]({iout})"
        self.print(prefix2)
        assert self.leaves[iorj] == {}
        if(iout not in self.accesed)and(not self.p.takeover):
            if(os.path.isdir(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp")):
                self.mainlog.warning(f"! No takeover ! Remove `{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp`")
                shutil.rmtree(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp")
        snap = self.load_snap(iout, prefix=prefix)
        gals = self.load_gals(iout, galid='all', prefix=prefix, verbose=verbose+1)
        if(self.p.strict): gals = gals[self.dict_pure[iout]]
        elif(iout==np.max(self.p.nout)): gals = gals[self.dict_pure[iout]]
        else: pass
        

        if(os.path.isdir(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp")):
            mask = np.isin(gals['id'], self.avoid_filter[iout], assume_unique=True)
            exists = np.array([os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{igal['id']}.pickle") for igal in gals[mask]])
            for igal in gals[mask][exists]:
                backup = pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{igal['id']}.pickle")
                self.leaves[iorj][igal['id']] = Leaf(self, igal, None, snap, backup=backup)
            self.print(f"{prefix2} {np.sum(exists)}/{len(gals[mask])} leaves are loaded from dumped", level='info')
            if(np.sum(~exists)>0):
                parts = uhmi.HaloMaker.read_member_parts(snap, gals[mask][~exists], galaxy=self.p.galaxy, full_path=self.p.fullpath, usefortran=self.p.usefortran, nthread=self.p.ncpu)
                cursor=0
                for igal in gals[mask][~exists]:
                    self.print(f"{prefix2} Add new {iorj}leaf")
                    part = parts[cursor : cursor+igal['nparts']]
                    if(hasattr(part, 'table')):
                        part = part.table
                    self.leaves[iorj][igal['id']] = Leaf(self, igal, part, snap, backup=None)
                    cursor += igal['nparts']
                snap.clear()
                self.print(f"{prefix2} {np.sum(~exists)}/{len(gals[mask])} leaves are newly loaded", level='info')
            # for igal in gals[mask]:
            #     if(os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{igal['id']}.pickle")):
            #         backup = pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{igal['id']}.pickle")
            #         self.leaves[iorj][igal['id']] = Leaf(self, igal, None, snap, backup=backup)
            #     else:
            #         self.print(f"{prefix2} Add new {iorj}leaf")
            #         part = uhmi.HaloMaker.read_member_part(snap, igal['id'], galaxy=self.p.galaxy, full_path=self.p.fullpath, usefortran=self.p.usefortran)
            #         self.leaves[iorj][igal['id']] = Leaf(self, igal, part.table, snap, backup=None)
        else:
            if(self.p.loadall):
                partmatch = self.read_part_halo_match(iout)
                if(not self.p.default)and(self.last_pids is not None):
                    temp = self.last_pids[self.last_pids < len(partmatch)]
                    self.last_pids = temp
                    temp = gals[np.isin(gals['id'], np.unique(partmatch[temp]), assume_unique=True)]
                    self.print(f"{prefix2} Not default mode: At {iout}, {len(gals)} -> {len(temp)} based on last particles")
                    self.print(temp['id'])
                else:
                    temp = gals
                for igal in temp:
                    self.leaves[iorj][igal['id']] = Leaf(self, igal, snap.part.table[np.where(partmatch == igal['id'])[0]], snap, backup=None)
                self.print(f"{prefix2} {len(temp)}/{len(temp)} leaves are newly loaded", level='info')
                snap.clear()
                del partmatch, gals
                self.banned_list.append(iout)
            else:
                if(not self.p.default)and(self.last_pids is not None):
                    partmatch = self.read_part_halo_match(iout)
                    temp = self.last_pids[self.last_pids < len(partmatch)]
                    self.last_pids = temp
                    temp = gals[np.isin(gals['id'], np.unique(partmatch[temp]), assume_unique=True)]
                    self.print(f"{prefix2} Not default mode: At {iout}, {len(gals)} -> {len(temp)} based on last particles")
                    self.print(temp['id'])
                else:
                    temp = gals
                # This may be improved by using multiprocessing
                parts = uhmi.HaloMaker.read_member_parts(snap, temp, galaxy=self.p.galaxy, full_path=self.p.fullpath, usefortran=self.p.usefortran, nthread=self.p.ncpu)
                cursor=0
                for igal in temp:
                    part = parts[cursor : cursor+igal['nparts']]
                    if(hasattr(part, 'table')):
                        part = part.table
                    self.leaves[iorj][igal['id']] = Leaf(self, igal, part, snap, backup=None)
                    cursor += igal['nparts']
                self.print(f"{prefix2} {len(temp)}/{len(temp)} leaves are newly loaded", level='info')
                # for igal in temp:
                #     part = uhmi.HaloMaker.read_member_part(snap, igal['id'], galaxy=self.p.galaxy, full_path=self.p.fullpath, usefortran=self.p.usefortran)
                #     self.leaves[iorj][igal['id']] = Leaf(self, igal, part.table, snap, backup=None)
                del part
                snap.clear()

    def load_leaf(self, iorj:str, galid:int, backup:dict=None, prefix="", level='info', verbose=0)->'Leaf':
        if(iorj!='i')and(iorj!='j'): raise ValueError(f"iorj must be 'i' or 'j'!")
        iout = self.outs[iorj]
        if(self.leaves[iorj] == {}):
            self.read_leaves[iorj](prefix=prefix, level=level, verbose=verbose+1)
        if(self.leaves[iorj][galid] is None):
            backup = pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{galid}.pickle")
            self.leaves[iorj][galid] = Leaf(self, None, None, None, backup=backup)
        return self.leaves[iorj][galid]

    @_debug
    def write_leaves(self, iorj:str, prefix="", level='info', verbose=0):
        if(iorj!='i')and(iorj!='j'): raise ValueError(f"iorj must be 'i' or 'j'!")
        iout = self.outs[iorj]
        if(iout in self.out_of_use): return
        if(not iout in self.out_on_table): self.out_on_table.append(iout)
        if(os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}.pickle")):
            # CHECK
            fname = f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}.pickle"
            exist = pklload(fname)
            further = False
            vstack = np.vstack(exist['desc'])
            if(vstack.shape[1]==1):
                if( (exist['desc']==None).all() ): further = True
            else:
                douts = np.unique(np.vstack(exist['desc'])[:,0])
                if len(douts) != self.p.nsnap: further = True
            if(further):
                oldcount = 0
                fname_old = f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_{oldcount}.pickle"
                while(os.path.exists(fname_old)):
                    oldcount += 1
                    fname_old = f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_{oldcount}.pickle"
                os.rename(fname, fname_old)
        if( not os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}.pickle") ): #<-------HERE!!!!!!!!!!!!!
            prefix2 = f"[write_leaves]({iout})"

            keys = list(self.leaves[iorj].keys())
            savecount = 0
            dumpcount = 0
            unchanged = 0
            nfiltered = 0
            if( not os.path.isdir(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp") ):
                os.mkdir(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp")
            if(len(keys)>0):
                if( not os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/keys.pickle") ):
                    temp = [key for key in keys if(key in self.avoid_filter[iout])]
                    pklsave(temp, f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/keys.pickle", overwrite=True)
                else:
                    temp1 = pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/keys.pickle"); temp1.sort()
                    temp2 = [key for key in keys if(key in self.avoid_filter[iout])]; temp2.sort()
                    if( temp1 != temp2 ):
                        self.print(f"{prefix2} Update keys: \n{temp1} \n-> \n{temp2}")
                        pklsave(temp2, f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/keys.pickle", overwrite=True)
                backup = {}
                for key in keys:
                    if( self.leaves[iorj][key] is None ): # Dumped
                        dumpcount += 1
                        continue
                    if(key in self.avoid_filter[iout]):
                        leaf = self.load_leaf(iorj, key, prefix=prefix, level=level, verbose=verbose+1)
                        if( leaf.changed ):
                            backup = {}
                            if( os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{key}.pickle") ):
                                backup = pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{key}.pickle")
                            backup = leaf.selfsave(backup=backup)
                            pklsave(backup, f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{key}.pickle", overwrite=True)
                            savecount += 1
                        else:
                            unchanged += 1
                    else:
                        nfiltered += 1
                    self.leaves[iorj][key] = None
                del backup
            self.print(f"{prefix2} Write `{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp`", level=level)
            self.print(f"{prefix2} {savecount}/{len(keys)} leaves", level=level)
            self.print(f"{prefix2}     {dumpcount}: alread dumped", level=level)
            self.print(f"{prefix2}     {unchanged}: nothing changed", level=level)
            self.print(f"{prefix2}     {nfiltered}: filtered from params", level=level)


    
    @_debug
    def flush(self, iout:int, prefix="", logger=None, level='info', verbose=0):
        if logger is None:
            logger = self.logger
        
        istep = out2step(iout, self.p.nout, self.p.nstep)
        cstep = istep+self.p.nsnap
        
        # 1)
        # Remove Out-of-use data in memory and disk
        keys = list(self.dict_snap.keys())
        for jout in keys:
            if(jout in self.out_of_use):
                self.dict_snap[jout].clear()
                del self.dict_snap[jout]
                self.print(f"[flush #1] snapshot at {jout} is released", level='info')
        keys = list(self.dict_gals.keys())
        for jout in keys:
            if(jout in self.out_of_use):
                del self.dict_gals[jout]
                self.print(f"[flush #1] gals at {jout} is released", level='info')
        keys = list(self.part_halo_match.keys())
        for jout in keys:
            if(jout in self.out_of_use):
                del self.part_halo_match[jout]
                if os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{jout}_partmatch.pkl"):
                    os.remove(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{jout}_partmatch.pkl")
                    self.print(f"[flush #1] dumped partmatch at {jout} is removed", level='info')
                self.print(f"[flush #1] part_halo_match at {jout} is released", level='info')
        keys = list(self.dict_pure)
        for jout in keys:
            if(jout in self.out_of_use):
                del self.dict_pure[jout]
                self.print(f"[flush #1] dict_pure at {jout} is released", level='info')
        
        for jout in self.out_of_use:
            if(os.path.isdir(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{jout:05d}_temp")):
                shutil.rmtree(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{jout:05d}_temp")
                self.print(f"[flush #1] leaves at {jout} are removed", level='info')
            if(self.outs['i']==jout): self.leaves['i'] = {}
            if(self.outs['j']==jout): self.leaves['j'] = {}
                    
        gc.collect()
        self.memory = GB()
            
        # 2)    
        # Dumping data in memory
        if(self.memory > self.p.flushGB):
            if(iout in self.part_halo_match.keys()):
                partmatch = self.part_halo_match[iout]
                if len(partmatch)>0:
                    if not os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout}_partmatch.pkl"):
                        pklsave(partmatch, f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout}_partmatch.pkl")
                        self.print(f"[flush #2] partmatch at {iout} is dumped", level='info')
                    else:
                        self.print(f"[flush #2] partmatch at {iout} is released", level='info')
                    self.part_halo_match[iout] = np.array([], dtype=int)
                partmatch = None
            if(iout==self.outs['i']):
                keys = list(self.leaves['i'].keys())
                for key in keys:
                    leaf:Leaf = self.leaves['i'][key]
                    if(not os.path.isdir(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp")):
                        os.mkdir(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp")
                    temp={}
                    if(os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{key}.pickle")):
                        temp = pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{key}.pickle")
                    backup = leaf.selfsave(backup=temp)
                    pklsave(backup, f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{key}.pickle", overwrite=True)
                    self.print(f"[flush #2] iLeaf{key} at {iout} is dumped", level='info')
                    self.leaves['i'][key] = None
                del leaf, backup, temp
            if(iout==self.outs['j']):
                keys = list(self.leaves['j'].keys())
                for key in keys:
                    leaf:Leaf = self.leaves['j'][key]
                    if(not os.path.isdir(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp")):
                        os.mkdir(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp")
                    temp={}
                    if(os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{key}.pickle")):
                        temp = pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{key}.pickle")
                    backup = leaf.selfsave(backup=temp)
                    pklsave(backup, f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{key}.pickle", overwrite=True)
                    self.print(f"[flush #2] jLeaf{key} at {iout} is dumped", level='info')
                    self.leaves['j'][key] = None
                del leaf, backup, temp

            self.print(self.summary(), level='info')
            
        gc.collect()
        self.memory = GB()

    def summary(self, isprint=False):
        gc.collect()
        
        # For each isnap, How many particles
        temp = []
        for key in self.dict_snap.keys():
            val = sys.getsizeof(self.dict_snap[key])
            val += self.dict_snap[key].part_data.nbytes if(self.dict_snap[key].part_data is not None) else 0
            val += self.dict_snap[key].part.nbytes if(self.dict_snap[key].part is not None) else 0
            temp.append(f"\t{key}: {val / 2**20:.2f} MB\n")
        tsnap = "".join(temp)

        # For each iout, How many leaves
        temp = []
        idkeys = list(self.leaves['i'].keys())
        ndump = 0
        val = sys.getsizeof(self.leaves['i'])
        for idkey in idkeys:
            ileaf = self.leaves['i'][idkey]
            if(ileaf is not None):
                if(not ileaf.contam):
                    val += ileaf.size()
            else:
                ndump += 1
        temp.append(f"\t[i]{self.outs['i']}: {len(idkeys)-ndump} leaves ({ndump} dumped) {val / 2**20:.2f} MB\n")
        idkeys = list(self.leaves['j'].keys())
        ndump = 0
        val = sys.getsizeof(self.leaves['j'])
        for idkey in idkeys:
            ileaf = self.leaves['j'][idkey]
            if(ileaf is not None):
                if(not ileaf.contam):
                    val += ileaf.size()
            else:
                ndump += 1
        temp.append(f"\t[j]{self.outs['j']}: {len(idkeys)-ndump} leaves ({ndump} dumped) {val / 2**20:.2f} MB\n")
        tleaf = "".join(temp)        

        # For each iout, How many matched particles
        temp = []
        for key in self.part_halo_match.keys():
            leng = len(self.part_halo_match[key])
            if(leng>0):
                temp.append(f"\t{key}: {leng} matched parts ({self.part_halo_match[key].nbytes / 2**20:.2f} MB)\n")
            else:
                if(os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{key}_partmatch.pkl")):
                    temp.append(f"\t{key}: matched parts are dumped\n")
                else:
                    temp.append(f"\t{key}: no matched parts\n")
        tmatch = "".join(temp)
        
        text = f"\n[Tree Data Report]\n\n>>> Snaps\n{tsnap}>>> Leaves\n{tleaf}>>> Matched particles\n{tmatch}\n>>> Used Memory: {GB():.4f} GB\n"
        self.memory = GB()

        if isprint:
            print(text)
        return text
    
    def out2step(self, iout):
        try:
            arg = np.argwhere(iout==self.p.nout)[0][0]
            return self.p.nstep[arg]
        except IndexError:
            print(f"\n!!! {iout} is not in nout({np.min(self.p.nout)}~{np.max(self.p.nout)}) !!!\n")
            traceback.print_stack()

    def step2out(self, istep):
        try:
            arg = np.argwhere(istep==self.p.nstep)[0][0]
            return self.p.nout[arg]
        except IndexError:
            print(f"\n!!! {istep} is not in nstep({np.min(self.p.nstep)}~{np.max(self.p.nstep)}) !!!\n")
            traceback.print_stack()

    def update_debugger(self, iorj:str):
        for ikey in self.leaves[iorj].keys():
            ileaf = self.load_leaf(iorj, ikey)
            ileaf.logger = self.logger


    @_debug
    def find_cands(self, prefix="", level='debug', verbose=0):
        """
        Finds candidates based on certain conditions and performs calculations.
        1) It starts by retrieving the keys from the 'i' and 'j' dictionaries stored in the leaves attribute.
        2) For each key in ikeys, it loads the corresponding leaf object (ileaf).
        3) It determines whether a calculation is required based on the relationship between the 'j' and 'i' values.
        4) If a calculation is needed, it performs the main calculation, which involves filtering and manipulating data using numpy functions.
        5) If no calculation is required, it retrieves pre-calculated values from the leaf object.
        6) It generates a debugging message based on the number of candidates and prints it if the verbosity level allows.

        Parameters:
            `prefix` (str): Prefix for debugging message (default="").
            `level` (str): Debug level (default='debug').
            `verbose` (int): Verbosity level (default=0).

        Returns:
            None
        """
        ikeys = list(self.leaves['i'].keys())
        jkeys = list(self.leaves['j'].keys())
        self.print(f"{prefix} {self.outs['i']} <-> {self.outs['j']}", level='info')
        jhalos_mem = None
        for key in ikeys:
            ileaf:Leaf = self.load_leaf('i', key, prefix=prefix, level=level, verbose=verbose+1)
            # Calc, or not?
            calc = True
            if(self.outs['j']>self.outs['i']):
                if(ileaf.desc is not None):
                    if(self.outs['j'] in ileaf.desc[:,0]): calc=False
            if(self.outs['j']<self.outs['i']):
                if(ileaf.prog is not None):
                    if(self.outs['j'] in ileaf.prog[:,0]): calc=False

            # Main calculation
            if calc:
                if jhalos_mem is None:
                    jhalos_mem = self.read_part_halo_match(self.outs['j'])
                pid = ileaf.pid
                pid = pid[pid <= len(jhalos_mem)]
                hosts = jhalos_mem[pid-1]
                hosts = hosts[hosts>0]
                hosts, count = np.unique(hosts, return_counts=True) # CPU?
                hosts = hosts[count/len(pid) > self.p.mcut]
                hosts = hosts[ np.isin(hosts, jkeys, assume_unique=True) ] # Short, so no need to use large_isin
                if len(hosts)>0:
                    otherleaves = [self.load_leaf('j', iid, prefix=prefix, level=level, verbose=verbose+1) for iid in hosts]
                    otherleaves = [ileaf for ileaf in otherleaves if(not ileaf.contam)]
                    # Change above
                    # If contam, remake that leaf
                    ids, scores = ileaf.calc_score(self.outs['j'], otherleaves, prefix=f"<{ileaf._name}>",level='debug', verbose=verbose+1)
                    # if(len(otherleaves)>1):
                    #     self.print(f"Multiprocessing for {len(otherleaves)} cpus", level='debug')
                    #     ids, scores = ileaf.calc_score(self.outs['j'], otherleaves, prefix=f"<{ileaf._name}>",level='debug', verbose=verbose+1) # CPU?
                    # else:
                    #     self.print(f"Singleprocessing for {len(otherleaves)} cpus", level='debug')
                    #     ids, scores = ileaf.calc_score(self.outs['j'], otherleaves, prefix=f"<{ileaf._name}>",level='debug', verbose=verbose+1)
                    # if(rcount==500):
                    #     self.print(f"500: {time.time()-ref:.3f}")
                    #     ref = time.time()
                    #     raise ValueError("Stop")
                    self.avoid_filter[self.outs['j']] = ids.flatten() if(self.avoid_filter[self.outs['j']] is None) else np.union1d(self.avoid_filter[self.outs['j']], ids.flatten())
                else:
                    ids = np.array([[self.outs['j'], 0]])
                    scores = np.array([[-10, -10, -10, -10, -10]])
                if self.outs['j']<self.outs['i']:
                    ileaf.prog = ids if ileaf.prog is None else np.vstack((ileaf.prog, ids))
                    ileaf.prog_score = scores if ileaf.prog_score is None else np.vstack((ileaf.prog_score, scores))
                    ileaf.changed = True
                elif self.outs['j']>self.outs['i']:
                    ileaf.desc = ids if ileaf.desc is None else np.vstack((ileaf.desc, ids))
                    ileaf.desc_score = scores if ileaf.desc_score is None else np.vstack((ileaf.desc_score, scores))
                    ileaf.changed = True
                else:
                    raise ValueError(f"Same output {self.outs['i']} and {self.outs['j']}!")
            
            # No need to calculation
            else:
                if self.outs['j']<self.outs['i']:
                    arg = ileaf.prog[:,0]==self.outs['j']
                    ids = ileaf.prog[arg]
                    if(self.avoid_filter[self.outs['j']] is None): self.avoid_filter[self.outs['j']] = ids.flatten()
                    scores = ileaf.prog_score[arg]
                elif self.outs['j']>self.outs['i']:
                    arg = ileaf.desc[:,0]==self.outs['j']
                    ids = ileaf.desc[arg]
                    if(self.avoid_filter[self.outs['j']] is None): self.avoid_filter[self.outs['j']] = ids.flatten()
                    scores = ileaf.desc_score[arg]
                else:
                    raise ValueError(f"Same output {self.outs['i']} and {self.outs['j']}!")
                
            # Debugging message
            msg = f"{prefix}<{ileaf.name()}> has {len(ids)} candidates"
            if len(ids)>0:
                if np.sum(scores[0])>0:
                    if len(ids) < 6:
                        msg = f"{msg} {[f'{ids[i][-1]}({scores[i][0]:.3f})' for i in range(len(ids))]}"
                    else:
                        msg = f"{msg} {[f'{ids[i][-1]}({scores[i][0]:.3f})' for i in range(5)]+['...']}"
                else:
                    msg = f"{prefix}<{ileaf.name()}> has {len(ids)-1} candidates"
            if(verbose <= self.p.verbose):self.print(msg, level='debug')
        if(not self.p.default): self.print(f"Pass filtered at {self.outs['j']}: ({len(self.avoid_filter[self.outs['j']])}){self.avoid_filter[self.outs['j']]}")

    @_debug
    def finalize(self, iout:int, prefix="", level='info', verbose=0):
        """
        Finalizes the backup file by combining and saving the individual pickled files into a single file.

        Parameters
        ----------
        iout : int
            The backup file index.
        prefix : str, optional
            Prefix to be added to log messages. Defaults to an empty string.
        level : str, optional
            Log level for printing messages. Defaults to 'info'.
        verbose : int, optional
            Verbosity level. Defaults to 0.

        Raises
        ------
        FileNotFoundError
            If the temporary backup file directory is not found.
        """
        prefix2 = f"[Reduce Backup file] ({iout})"
        if(os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}.pickle")):
            self.print(f"{prefix2} Already saved `{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}.pickle`", level=level)
            return
        if not os.path.exists(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp"):
            raise FileNotFoundError(f"`{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp` is not found!")
        keys = pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/keys.pickle")
        count = 0
        for key in keys:
            gal = pklload(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp/{key}.pickle")['gal']
            gals = gal if count==0 else np.hstack((gals, gal))
            count += 1
        pklsave(gals, f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}.pickle", overwrite=True)
        self.print(f"{prefix2} Save `{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}.pickle`", level=level)
        self.out_of_use.append(iout)
        shutil.rmtree(f"{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp")
        self.print(f"{prefix2} Remove `{self.p.resultdir}/by-product/{self.p.fileprefix}{iout:05d}_temp`", level=level)
        del gals








def _debug_leaf(func):
    @functools.wraps(func)
    def wrapper(self:'Leaf', *args, **kwargs):
        Tree = self.base
        ontime=Tree.ontime;onmem=Tree.onmem;oncpu=Tree.oncpu
        p = Tree.p
        logger = Tree.logger
        fprint = Tree.print
        verblvl = kwargs.pop("verbose",0); kwargs["verbose"]=verblvl
        verbose = verblvl <= p.verbose

        prefix = kwargs.pop("prefix","")
        prefix = f"{prefix}[{func.__name__}({verblvl})]"; kwargs["prefix"]=prefix
        level = kwargs.pop("level","debug"); kwargs["level"]=level

        if ontime:
            clock=timer(text=prefix, logger=logger, verbose=verbose, level=level, mint=Tree.p.mint)
        if onmem:
            mem = MB()
        if oncpu:
            cpu = DisplayCPU()
            cpu.iq = 0
            cpu.queue = np.zeros(100)-1
            cpu.start()
        result = func(self,*args, **kwargs)
        if ontime:
            clock.done()
        if onmem:
            fprint(f"{prefix}  mem ({MB() - mem:.2f} MB) [{mem:.2f} -> {MB():.2f}]", level=level)
        if oncpu:
            cpu.stop()
            icpu = cpu.queue[cpu.queue >= 0]
            if len(icpu)>0:
                q16, q50, q84 = np.percentile(icpu, q=[16,50,84])
            else:
                q16 = q50 = q84 = 0
            fprint(f"{prefix}  cpu ({q50:.2f} %) [{q16:.1f} ~ {q84:.1f}]", level=level)
        return result
    return wrapper

class Leaf:
    __slots__ = ['backupstr', 'changed', 'cat', 'id', 'iout', '_name', 'base', 'logger',
                 'contam', 'changed', 'nparts', 'pid', 'pvx', 'pvy', 'pvz', 'pweight',
                 'prog', 'prog_score', 'desc', 'desc_score', 'saved_matchrate', 'saved_veloffset']
    def __init__(self, base:TreeBase, gal:np.ndarray, parts:np.ndarray, snap:uri.RamsesSnapshot, backup:dict=None, prefix:str=None, **kwargs):
        backupstr = "w/o backup"
        self.changed = True
        if backup is not None:
            backupstr = "w/ backup"
            gal = backup['gal']          
            self.changed = False
        self.cat = gal
        self.id:int = self.cat['id']
        self.iout:int = self.cat['timestep']
        self._name = f"L{self.id} at {self.iout}"
        self.base:TreeBase = base
        self.logger = base.logger
        self.contam = False
        try:
            if(self.base.p.strict)and(not self.base.p.filtering(self.cat)):
            # if self.cat['mcontam']/self.cat['m'] > self.base.p.fcontam:
                self.contam = True
            if(not self.base.p.galaxy):
                self._name = f"L{self.id} at {self.iout} ({int(100*self.cat['mcontam']/self.cat['m'])}%)"
        except:
            pass
        if(self.base.avoid_filter[self.iout] is not None):
            if(self.id in self.base.avoid_filter[self.iout]): self.contam=False
        # Mem part
        if not self.contam:
            if(backup is not None):
                self.nparts = backup['part']['nparts']
                self.pid = backup['part']['id']
                self.pvx = backup['part']['vx']
                self.pvy = backup['part']['vy']
                self.pvz = backup['part']['vz']
                self.pweight = backup['part']['weight']
            else:
                part = np.array(parts)
                del parts
                self.nparts = len(part)
                self.pid:np.ndarray[int] = np.abs(part['id']) if self.base.p.galaxy else part['id']
                self.pvx:np.ndarray[float] = part['vx']/snap.unit['km/s']#, 'km/s']
                self.pvy:np.ndarray[float] = part['vy']/snap.unit['km/s']#, 'km/s']
                self.pvz:np.ndarray[float] = part['vz']/snap.unit['km/s']#, 'km/s']
                dist = distance(self.cat, part)
                dist /= np.std(dist)
                vels = distance3d(self.cat['vx'], self.cat['vy'], self.cat['vz'], self.pvx, self.pvy, self.pvz)
                vels /= np.std(vels)
                self.pweight:np.ndarray[float] = part['m'] / np.sqrt(dist**2 + vels**2)
                self.pweight /= np.sum(self.pweight)
                del part

            # Progenitor & Descendant
            if(backup is not None):
                self.prog = self.cat['prog']
                self.prog_score = self.cat['prog_score']
                self.desc = self.cat['desc']
                self.desc_score = self.cat['desc_score']
                self.saved_matchrate = backup['saved']['matchrate']
                self.saved_veloffset = backup['saved']['veloffset']
            else:
                self.prog = None
                self.desc = None
                self.prog_score = None
                self.desc_score = None
                self.saved_matchrate = {}
                self.saved_veloffset = {}
    
    # def __del__(self):
    #     print(f"Delete {self._name}, {id(self)}")

    def name(self):
        return self._name
    def __repr__(self):
        return self._name
    def __str__(self):
        return self._name
    def size(self):
        val = sys.getsizeof(self)
        # Add array
        attrs = ['pid', 'pvx', 'pvy', 'pvz', 'pweight', 'prog', 'prog_score', 'desc', 'desc_score']
        for iattr in attrs:
            arr = getattr(self, iattr)
            val += 0 if(arr is None) else arr.nbytes
        # Add dictionary
        # for key_iout in self.saved_matchrate.keys():
        #     for key_id in self.saved_matchrate[key_iout].keys():
        #         val += 0 if(self.saved_matchrate[key_iout][key_id] is None) else self.saved_matchrate[key_iout][key_id][1].nbytes
        return val

    def summary(self):
        t1 = f"[Leaf Summary] {self._name} ({self.changed}))"
        t2 = f"\t{self.nparts} particles"
        t3 = f"Progenitors:"
        if(self.prog is not None):
            temp = []
            for prog,pscore in zip(self.prog, self.prog_score):
                temp += f"\t\t{prog[0]*100000+prog[1]} ({pscore[0]:.4f})\n"
            t4 = "".join(temp)        
        else:
            t4 = "\t\tNone"
        t5 = f"Descendants:"
        if(self.desc is not None):
            temp = []
            for desc,dscore in zip(self.desc, self.desc_score):
                temp += f"\t\t{desc[0]*100000+desc[1]} ({dscore[0]:.4f})\n"
            t6 = "".join(temp)
        else:
            t6 = "\t\tNone"
        return f"\n{t1}\n{t2}\n{t3}\n{t4}\n{t5}\n{t6}"

        
    def clear(self):
        del self.cat; self.cat = None
        if not self.contam:
            del self.nparts; self.nparts = 0
            del self.pid; self.pid = np.array([])
            del self.pvx; self.pvx = np.array([])
            del self.pvy; self.pvy = np.array([])
            del self.pvz; self.pvz = np.array([])
            del self.pweight; self.pweight = None
            del self.saved_matchrate; self.saved_matchrate = {}
            del self.saved_veloffset; self.saved_veloffset = {}    
            del self.prog; self.prog = None
            del self.prog_score; self.prog_score = None    
            del self.desc; self.desc = None    
            del self.desc_score; self.desc_score = None    

    def selfsave(self, backup={}) -> dict:
        # Save GalaxyMaker
        if not self.contam:
            if('gal' in backup.keys()):
                arr = backup['gal']
                arr['prog'] = self.prog
                arr['prog_score'] = self.prog_score
                arr['desc'] = self.desc
                arr['desc_score'] = self.desc_score
            else:
                backup_keys = ["nparts", "id", "timestep", "host", "aexp", "m", "x", "y", "z", "vx", "vy", "vz", "r", "rvir", "mvir"]
                lis = [ia for ia,ib in zip(self.cat, self.cat.dtype.names) if ib in backup_keys] + [self.prog] + [self.prog_score] + [self.desc] + [self.desc_score]
                dtype = np.dtype([idtype for idtype in self.cat.dtype.descr if idtype[0] in backup_keys] + [('prog','O'), ("prog_score",  'O'), ('desc','O'), ("desc_score",  'O')])
                arr = np.empty(1, dtype=dtype)[0]
                for i, ilis in enumerate(lis):
                    arr[i] =  ilis
            self.cat = arr

            backup['gal'] = self.cat
            if not ('part' in backup.keys()):backup['part'] = {'nparts':self.nparts, 'id':self.pid, 'vx':self.pvx, 'vy':self.pvy, 'vz':self.pvz, 'weight':self.pweight}
            backup['saved'] = {'matchrate':self.saved_matchrate, 'veloffset':self.saved_veloffset}

            return backup
        return None

    @_debug_leaf
    def calc_score(self, jout:int, otherleaves:list['Leaf'], prefix="", level='debug', verbose=0):
        if not self.contam:
            # for otherleaf in otherleaves:
            leng = len(otherleaves)
            ids = np.empty((leng,2), dtype=int)
            scores = np.empty((leng,5), dtype=float)
            for i in range(leng):
                otherleaf = otherleaves[i]
                if otherleaf.iout != jout:
                    raise ValueError(f"Calc score at {jout} but found <{otherleaf.name}>!")
                score1, selfind = self.calc_matchrate(otherleaf, prefix=prefix, verbose=verbose+1)
                score2, otherind = otherleaf.calc_matchrate(self, prefix=prefix, verbose=verbose+1)
                score3 = self.calc_veloffset(otherleaf, selfind=selfind, otherind=otherind, prefix=prefix, verbose=verbose+1)
                score4 = np.exp( -np.abs(np.log10(self.cat['m']/otherleaf.cat['m'])) ) 
                scores_tot = score1 + score2 + score3 + score4
                scores[i] = (scores_tot, score1, score2, score3, score4)
                ids[i] = (jout, otherleaf.id)
            arg = np.argsort(scores[:, 0])
            return ids[arg][::-1], scores[arg][::-1]

    @_debug_leaf
    def _calc_matchrate(self:'Leaf', otherleaf:'Leaf', prefix="", level='debug', verbose=5) -> float:
        large=False
        ilen = len(self.pid); jlen = len(otherleaf.pid)
        if(ilen >= 1e6)or(jlen >= 1e6):
            large=True
        elif(ilen*jlen >= 1e6):
            if(min(ilen, jlen) >= 1e3):
                large=True
        
        # ind = large_isin(self.pid, otherleaf.pid) if(large) else np.isin(self.pid, otherleaf.pid, assume_unique=True)
        ind = np.isin(self.pid, otherleaf.pid, assume_unique=True)
        if not True in ind:
            val = -1
        else:
            val = np.sum( self.pweight[ind] )
        jout = otherleaf.iout
        if not jout in self.saved_matchrate.keys():
            self.saved_matchrate[jout] = {}
            self.changed = True
        if not otherleaf.id in self.saved_matchrate[jout].keys():
            # self.saved_matchrate[jout][otherleaf.id] = (val, ind)
            self.saved_matchrate[jout][otherleaf.id] = val
            self.changed = True
        return val, ind
    
    def calc_matchrate(self, otherleaf:'Leaf', prefix="", level='debug', verbose=0) -> float:
        # jout = otherleaf.iout
        # if jout in self.saved_matchrate.keys():
        #     if otherleaf.id in self.saved_matchrate[jout].keys():
        #         val, ind = self.saved_matchrate[jout][otherleaf.id]
        #         return val, ind
        val, ind = self._calc_matchrate(otherleaf, prefix=prefix, level=level, verbose=verbose)
        return val, ind

    # @_debug_leaf
    def calc_bulkmotion(self, checkind:list[bool]=None, prefix="", level='debug', verbose=0):
        if checkind is None:
            checkind = np.full(self.nparts, True)

        weights = self.pweight[checkind]
        weights /= np.sum(weights)
        vx = nbsum( self.pvx[checkind], weights ) - self.cat['vx']
        vy = nbsum( self.pvy[checkind], weights ) - self.cat['vy']
        vz = nbsum( self.pvz[checkind], weights ) - self.cat['vz']
        # vx = np.sum( self.pvx[checkind] * weights ) - self.cat['vx']
        # vy = np.sum( self.pvy[checkind] * weights ) - self.cat['vy']
        # vz = np.sum( self.pvz[checkind] * weights ) - self.cat['vz']

        return np.array([vx, vy, vz])

    @_debug_leaf
    def _calc_veloffset(self:'Leaf', otherleaf:'Leaf', selfind:list[bool]=None, otherind:list[bool]=None, prefix="", level='debug', verbose=5) -> float:
        if selfind is None:
            val, selfind = self.calc_matchrate(otherleaf, prefix=prefix, verbose=verbose+1)
        if otherind is None:
            val, otherind = otherleaf.calc_matchrate(self, prefix=prefix, verbose=verbose+1)

        if howmany(selfind, True) < 3:
            val = 0
        else:
            selfv = self.calc_bulkmotion(checkind=selfind, prefix=prefix, verbose=verbose+1)
            otherv = otherleaf.calc_bulkmotion(checkind=otherind, prefix=prefix, verbose=verbose+1)
            val = 1 - nbnorm(otherv - selfv)/(nbnorm(selfv)+nbnorm(otherv))
        jout = otherleaf.iout
        if not jout in self.saved_veloffset.keys():
            self.saved_veloffset[jout] = {}
            self.changed = True
        if not otherleaf.id in self.saved_veloffset[jout].keys():
            self.saved_veloffset[jout][otherleaf.id] = val
            self.changed = True
        if not self.iout in otherleaf.saved_veloffset.keys():
            otherleaf.saved_veloffset[self.iout] = {}
            otherleaf.changed = True
        if not otherleaf.id in otherleaf.saved_veloffset[self.iout].keys():
            otherleaf.saved_veloffset[self.iout][self.id] = val
            otherleaf.changed = True
        return val


    def calc_veloffset(self, otherleaf:'Leaf', selfind:list[bool]=None, otherind:list[bool]=None, prefix="", level='debug', verbose=0) -> float:
        calc=True
        jout = otherleaf.iout
        if jout in self.saved_veloffset.keys():
            if otherleaf.id in self.saved_veloffset[jout].keys():
                val = self.saved_veloffset[jout][otherleaf.id]
                calc = False
        if calc:
            val = self._calc_veloffset(otherleaf, selfind=selfind, otherind=otherind, prefix=prefix, level=level, verbose=verbose)
        return val



    @_debug_leaf
    def calc_score_mp(self, jout:int, otherleaves:list['Leaf'], prefix="", level='debug', verbose=0):
        if not self.contam:
            leng = len(otherleaves)
            man = mp.Manager()
            _changed = man.list([0 for _ in range(leng)])
            _ids = man.list([0 for _ in range(leng)])
            _scores = man.list([0 for _ in range(leng)])
            addresses = [_changed, _ids, _scores]
            ileaf_dict = self.selfsave(backup={})
            jleaf_dicts = []
            jleaf_dicts = [otherleaf.selfsave(backup={}) for otherleaf in otherleaves]
            with Pool(processes=min(self.base.p.ncpu, leng)) as pool:
                async_results = [pool.apply_async(_scores_mp, args=(ileaf_dict, jleaf_dicts[i], jout, i, prefix, level, verbose, leng, addresses)) for i in range(leng)]
                for r in async_results: r.get()
                pool.close()
                pool.join()

            _changed = np.asarray(_changed)
            _ids = np.asarray(_ids)
            _scores = np.asarray(_scores)
            for i in range(leng):
                ichange1, jchange2, ichange3, jchange4 = _changed[i]
                if ichange1:
                    if(not jout in self.saved_matchrate.keys()):
                        self.saved_matchrate[jout] = {}
                        self.changed = True
                    if(not otherleaves[i].id in self.saved_matchrate[jout].keys()):
                        self.saved_matchrate[jout][otherleaves[i].id] = _scores[i][1]
                        self.changed = True
                if jchange2:
                    if(not self.iout in otherleaves[i].saved_matchrate.keys()):
                        otherleaves[i].saved_matchrate[self.iout] = {}
                        otherleaves[i].changed = True
                    if(not self.id in otherleaves[i].saved_matchrate[self.iout].keys()):
                        otherleaves[i].saved_matchrate[self.iout][self.id] = _scores[i][2]
                        otherleaves[i].changed = True
                if ichange3:
                    if(not jout in self.saved_veloffset.keys()):
                        self.saved_veloffset[jout] = {}
                        self.changed = True
                    if(not otherleaves[i].id in self.saved_veloffset[jout].keys()):
                        self.saved_veloffset[jout][otherleaves[i].id] = _scores[i][3]
                        self.changed = True
                if jchange4:
                    if(not self.iout in otherleaves[i].saved_veloffset.keys()):
                        otherleaves[i].saved_veloffset[self.iout] = {}
                        otherleaves[i].changed = True
                    if(not self.id in otherleaves[i].saved_veloffset[self.iout].keys()):
                        otherleaves[i].saved_veloffset[self.iout][self.id] = _scores[i][3]
                        otherleaves[i].changed = True
            arg = np.argsort(_scores[:, 0])
            _ids = np.array(_ids)
            result_ids = np.copy(_ids[arg][::-1])
            result_scores = np.copy(_scores[arg][::-1])

            return result_ids, result_scores


def _scores_mp(ileaf_dict, jleaf_dict, jout, i, prefix, level, verbose, leng, addresses):
    _changed = addresses[0]
    _ids = addresses[1]
    _scores = addresses[2]
    score1, selfind, ichange1 = _calc_matchrate_mp(ileaf_dict, jleaf_dict)
    score2, otherind, jchange2 = _calc_matchrate_mp(jleaf_dict, ileaf_dict)
    score3, ichange3, jchange4 = _calc_veloffset_mp(ileaf_dict, jleaf_dict, selfind=selfind, otherind=otherind)
    score4 = np.exp( -np.abs(np.log10(ileaf_dict['gal']['m']/jleaf_dict['gal']['m'])) )
    _scores[i] = (score1 + score2 + score3 + score4, score1, score2, score3, score4)
    _ids[i] = (jout, jleaf_dict['gal']['id'])
    _changed[i] = [ichange1, jchange2, ichange3, jchange4]



def _calc_matchrate_mp(ileaf_dict, jleaf_dict):
    large=False
    ilen = len(ileaf_dict['part']['id']); jlen = len(jleaf_dict['part']['id'])
    if(ilen >= 1e6)or(jlen >= 1e6):
        large=True
    elif(ilen*jlen >= 1e6):
        if(min(ilen, jlen) >= 1e3):
            large=True
    large=False
    # ind = large_isin(ileaf_dict['part']['id'], jleaf_dict['part']['id']) if(large) else np.isin(ileaf_dict['part']['id'], jleaf_dict['part']['id'], assume_unique=True)
    ind = np.isin(ileaf_dict['part']['id'], jleaf_dict['part']['id'], assume_unique=True)

    val = -1 if not True in ind else np.sum( ileaf_dict['part']['weight'][ind] )
    jout = jleaf_dict['gal']['timestep']
    changed = False
    if not jout in ileaf_dict['saved']['matchrate'].keys():
        ileaf_dict['saved']['matchrate'][jout] = {}
        changed = True
    if not jleaf_dict['gal']['id'] in ileaf_dict['saved']['matchrate'][jout].keys():
        ileaf_dict['saved']['matchrate'][jout][jleaf_dict['gal']['id']] = val
        changed = True
    return val, ind, changed

def _calc_veloffset_mp(ileaf_dict, jleaf_dict, selfind=None, otherind=None):
    calc=True
    jout = jleaf_dict['gal']['timestep']
    if jout in ileaf_dict['saved']['veloffset'].keys():
        if jleaf_dict['gal']['id'] in ileaf_dict['saved']['veloffset'][jout].keys():
            val = ileaf_dict['saved']['veloffset'][jout][jleaf_dict['gal']['id']]
            calc = False
            ichanged=False; jchanged=False
    if calc:
        if selfind is None: val, selfind = _calc_matchrate_mp(ileaf_dict, jleaf_dict)
        if otherind is None: val, otherind = _calc_matchrate_mp(jleaf_dict, ileaf_dict)

        if howmany(selfind, True) < 3: val=0
        else:
            selfv = _calc_bulkmotion_mp(ileaf_dict, selfind)
            otherv = _calc_bulkmotion_mp(jleaf_dict, otherind)
            # val = 1 - nbnorm(otherv - selfv)/(nbnorm(selfv)+nbnorm(otherv))
            val = 1 - norm(otherv - selfv)/(norm(selfv)+norm(otherv))
        jout = jleaf_dict['gal']['timestep']
        ichanged = False
        jchanged = False
        if not jout in ileaf_dict['saved']['veloffset'].keys():
            ileaf_dict['saved']['veloffset'][jout] = {}
            ichanged = True
        if not jleaf_dict['gal']['id'] in ileaf_dict['saved']['veloffset'][jout].keys():
            ileaf_dict['saved']['veloffset'][jout][jleaf_dict['gal']['id']] = val
            ichanged = True
        if not ileaf_dict['gal']['timestep'] in jleaf_dict['saved']['veloffset'].keys():
            jleaf_dict['saved']['veloffset'][ileaf_dict['gal']['timestep']] = {}
            jchanged = True
        if not ileaf_dict['gal']['id'] in jleaf_dict['saved']['veloffset'][ileaf_dict['gal']['timestep']].keys():
            jleaf_dict['saved']['veloffset'][ileaf_dict['gal']['timestep']][ileaf_dict['gal']['id']] = val
            jchanged = True
    return val, ichanged, jchanged

def _calc_bulkmotion_mp(ileaf_dict, checkind=None):
    if checkind is None: checkind = np.full(ileaf_dict['part']['nparts'], True)
    weights = ileaf_dict['part']['weight'][checkind]
    weights /= np.sum(weights)
    # vx = nbsum( ileaf_dict['part']['vx'][checkind], weights ) - ileaf_dict['gal']['vx']
    # vy = nbsum( ileaf_dict['part']['vy'][checkind], weights ) - ileaf_dict['gal']['vy']
    # vz = nbsum( ileaf_dict['part']['vz'][checkind], weights ) - ileaf_dict['gal']['vz']
    vx = np.sum( ileaf_dict['part']['vx'][checkind] * weights ) - ileaf_dict['gal']['vx']
    vy = np.sum( ileaf_dict['part']['vy'][checkind] * weights ) - ileaf_dict['gal']['vy']
    vz = np.sum( ileaf_dict['part']['vz'][checkind] * weights ) - ileaf_dict['gal']['vz']
    return np.array([vx, vy, vz])
