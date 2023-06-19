from rur import uri, uhmi
uri.timer.verbose=0
import numpy as np
import functools
from collections import defaultdict
from collections.abc import Iterable
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
        prefix = f"{prefix}[{func.__name__}]"; kwargs["prefix"] = prefix
        level = kwargs.pop("level","debug"); kwargs["level"] = level

        if ontime:
            clock=timer(text=prefix, logger=logger, verbose=verbose, level=level)
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
        self.dict_leaves = defaultdict(dict) # in {iout}, in {galid}, Leaf object
        self.part_halo_match = {} # in {iout}, list of int
        self.banned_list = []
        self.out_of_use = []
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
                path_in_repo="" if self.p.mode[0] == '/' else "snapshots"
                snap = uri.RamsesSnapshot(self.p.repo, iout, mode=self.p.rurmode, path_in_repo=path_in_repo)
                if(self.p.loadall)and(not iout in self.banned_list):
                    if(self.p.mode[0]=='y')or(self.p.mode=='nh'):
                        snap.get_part(pname=self.partstr, python=(not self.p.usefortran), nthread=self.p.ncpu, target_fields=["x","y","z","vx","vy","vz","m","epoch","id","cpu"])
                    else:
                        snap.get_part(pname=self.partstr, python=(not self.p.usefortran), nthread=self.p.ncpu, target_fields=["x","y","z","vx","vy","vz","m","epoch","id","cpu","family"])
                    snap.part.table = snap.part.table[np.argsort(np.abs(snap.part.table['id']))] # <- __copy__
                    snap.part.extra_fields = None
                    del snap.part.extra_fields
                    if(snap.part.ptype != self.partstr): snap.part.ptype = self.partstr
                    snap.part_data = np.array([])
                    snap.flush()
                del snap.cell_extra, snap.part_extra
                self.dict_snap[iout] = snap
                del snap
            _load_snap(self, iout, prefix=prefix, level=level, verbose=verbose)
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
                    def __load_gals(self, snap:uri.RamsesSnapshot, galid=None, prefix="", level='info', verbose=1):
                        gm, gpids = uhmi.HaloMaker.load(snap, galaxy=self.p.galaxy, double_precision=self.p.dp, load_parts=True, full_path=self.p.fullpath)
                        tmp = np.repeat(gm['id'], gm['nparts'])
                        self.part_halo_match[snap.iout] = np.zeros(np.max(gpids), dtype=np.int32)
                        self.part_halo_match[snap.iout][gpids-1] = tmp
                        del gpids, tmp
                        return gm
                    gm = __load_gals(self, snap, galid=galid, prefix=prefix, level=level, verbose=verbose+1)
                else:
                    gm = uhmi.HaloMaker.load(snap, galaxy=self.p.galaxy, double_precision=self.p.dp, full_path=self.p.fullpath)
                if not (self.p.galaxy):
                    mask = gm['mcontam']/gm['m'] <= self.p.fcontam
                    self.dict_pure[iout] = mask
                self.dict_gals[iout] = gm
                del snap
            _load_gals(self, iout,galid=galid, prefix=prefix, level=level, verbose=verbose)
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
            elif os.path.isfile(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_partmatch.pkl"):
                return pklload(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_partmatch.pkl")
            else:
                raise ValueError(f"Cannot find `part_halo_match` at {iout}!")
        else:
            raise ValueError(f"Cannot find a key `{iout}` in `part_halo_match`!")
            
    def load_leaf(self, iout:int, galid:int, backup:dict=None, backups=None, prefix="", level='debug', verbose=0)->'Leaf':
        calc = False
        switch = False
        if not galid in self.dict_leaves[iout].keys():
            calc=True
        else:
            if self.dict_leaves[iout][galid] is None:
                calc=True
            else:
                if(self.dict_leaves[iout][galid].cat is None):
                    calc=True
                    if(self.p.loadall):
                        switch=True
                        self.p.loadall = False
        if(calc):
            @_debug
            def _load_leaf(self, iout:int, galid:int, backup:dict=None, backups=None, prefix="", level='debug', verbose=verbose+1):
                if os.path.isfile(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_leaf_{galid}.pkl"):
                    backup = pklload(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_leaf_{galid}.pkl")
                    leaf = Leaf(self, None, None, None, backup=backup)
                    if not leaf.contam:
                        leaf.base = self
                        leaf.logger = self.logger
                        self.dict_leaves[iout][galid] = leaf
                else:
                    snap = self.load_snap(iout, prefix=prefix)
                    if self.p.loadall:
                        gals = self.load_gals(iout, galid='all')
                        if(not self.p.galaxy): gals = gals[ self.dict_pure[iout] ]
                        partmatch = self.read_part_halo_match(iout)
                        for igal in gals:
                            if(backups is not None):
                                ileaf = Leaf(self, igal, None,None, backup=backups[igal['id']])
                                self.dict_leaves[iout][igal['id']] = ileaf
                            else:
                                ileaf = Leaf(self, igal, snap.part.table[np.where(partmatch == igal['id'])[0]], snap, backup=backup) # <- __copy__
                                if(not ileaf.contam):
                                    self.dict_leaves[iout][igal['id']] = ileaf
                            del ileaf
                        snap.clear()
                        # self.dict_gals[iout] = np.array([])
                        del partmatch, gals
                        self.banned_list.append(iout)
                    else:
                        gal = self.load_gals(iout, galid, prefix=prefix)
                        go=True
                        if(not self.p.galaxy):
                            if(gal['mcontam']/gal['m'] > self.p.fcontam):
                                go=False
                        if(go):
                            part = uhmi.HaloMaker.read_member_part(snap, galid, galaxy=self.p.galaxy, full_path=self.p.fullpath, usefortran=self.p.usefortran)
                            ileaf = Leaf(self, gal, part.table, snap, backup=backup)
                            if not ileaf.contam:
                                self.dict_leaves[iout][galid]=ileaf
                            del ileaf, part
                    del snap
            _load_leaf(self, iout, galid, backup=backup, backups=backups, prefix=prefix, level=level, verbose=verbose+1)
            self.memory = GB()
            if(switch): self.p.loadall = True
        if galid in self.dict_leaves[iout].keys():
            # [Debugging]
            # if(iout==186)and(galid==1):
            #     msg = self.dict_leaves[iout][galid].summary()
            #     self.print(msg)
            return self.dict_leaves[iout][galid]
    
    @_debug
    def flush(self, iout:int, prefix="", leafclear=False, logger=None, level='info', verbose=0):
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
                # self.dict_gals[jout] = {}
                del self.dict_gals[jout]
                self.print(f"[flush #1] gals at {jout} is released", level='info')
        keys = list(self.part_halo_match.keys())
        for jout in keys:
            if(jout in self.out_of_use):
                del self.part_halo_match[jout]
                if os.path.isfile(f"{self.p.resultdir}/by-product/{self.p.logprefix}{jout}_partmatch.pkl"):
                    os.remove(f"{self.p.resultdir}/by-product/{self.p.logprefix}{jout}_partmatch.pkl")
                    self.print(f"[flush #1] dumped partmatch at {jout} is removed", level='info')
                self.print(f"[flush #1] part_halo_match at {jout} is released", level='info')
        keys = list(self.dict_pure)
        for jout in keys:
            if(jout in self.out_of_use):
                # self.dict_pure[jout] = []
                del self.dict_pure[jout]
                self.print(f"[flush #1] dict_pure at {jout} is released", level='info')
        keys = list(self.dict_leaves.keys())
        for jout in keys:
            if(jout in self.out_of_use):
                keys2 = list(self.dict_leaves[jout].keys())
                for key in keys2:
                    if(self.dict_leaves[jout][key] is not None):self.dict_leaves[jout][key].clear()
                    del self.dict_leaves[jout][key]
                    if os.path.isfile(f"{self.p.resultdir}/by-product/{self.p.logprefix}{jout}_leaf_{key}.pkl"):
                        os.remove(f"{self.p.resultdir}/by-product/{self.p.logprefix}{jout}_leaf_{key}.pkl")
                        self.print(f"[flush #1] dumped leaf{key} at {jout} is removed", level='info')
                del self.dict_leaves[jout]
                self.print(f"[flush #1] leaves at {jout} are released", level='info')                
        gc.collect()
        self.memory = GB()
            
        # 2)    
        # Dumping data in memory
        if(self.memory > self.p.flushGB):
            if(iout in self.part_halo_match.keys()):
                partmatch = self.part_halo_match[iout]
                if len(partmatch)>0:
                    if not os.path.isfile(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_partmatch.pkl"):
                        pklsave(partmatch, f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_partmatch.pkl")
                        self.print(f"[flush #2] partmatch at {iout} is dumped", level='info')
                    else:
                        self.print(f"[flush #2] partmatch at {iout} is released", level='info')
                    self.part_halo_match[iout] = np.array([], dtype=int)
                partmatch = None
            if(iout in self.dict_leaves.keys()):
                keys = list(self.dict_leaves[iout].keys())
                for key in keys:
                    leaf:Leaf = self.dict_leaves[iout][key]
                    if(leaf is None):
                        assert os.path.isfile(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_leaf_{key}.pkl")
                        self.print(f"[flush #2] Leaf{key} at {iout} is released", level='info')
                    else:
                        temp={}
                        if(os.path.exists(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_leaf_{key}.pkl")):
                            temp = pklload(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_leaf_{key}.pkl")
                        backup = leaf.selfsave(backup=temp)
                        pklsave(backup, f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_leaf_{key}.pkl", overwrite=True)
                        self.print(f"[flush #2] Leaf{key} at {iout} is dumped", level='info')
                    self.dict_leaves[iout][key] = None
                leaf = None                
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
        for key in self.dict_leaves.keys():
            idkeys = list(self.dict_leaves[key].keys())
            ndump = 0
            val = sys.getsizeof(self.dict_leaves[key])
            for idkey in idkeys:
                ileaf = self.dict_leaves[key][idkey]
                if(ileaf is not None):
                    val += ileaf.size()
                else:
                    ndump += 1
            temp.append(f"\t{key}: {len(idkeys)-ndump} leaves ({ndump} dumped) {val / 2**20:.2f} MB\n")
            # if(self.dict_leaves[key][idkeys[0]] is None):
            #     # Check dumped
            #     ndump = 0
            #     for idkey in idkeys:
            #         if(os.path.exists(f"{self.p.resultdir}/by-product/{self.p.logprefix}{key}_leaf_{idkey}.pkl")):
            #             ndump += 1
            #     temp += f"\t{key}: {len(idkeys)-ndump} leaves ({ndump} leaves are dumped)\n"
            # else:
            #     temp += f"\t{key}: {len(idkeys)} leaves\n"
        tleaf = "".join(temp)        

        # For each iout, How many matched particles
        temp = []
        for key in self.part_halo_match.keys():
            leng = len(self.part_halo_match[key])
            if(leng>0):
                temp.append(f"\t{key}: {leng} matched parts ({self.part_halo_match[key].nbytes / 2**20:.2f} MB)\n")
            else:
                if(os.path.exists(f"{self.p.resultdir}/by-product/{self.p.logprefix}{key}_partmatch.pkl")):
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

    def update_debugger(self, iout:int=None):
        if iout is None:
            for iout in self.dict_leaves.keys():
                for jkey in self.dict_leaves[iout].keys():
                    jleaf = self.load_leaf(iout, jkey)
                    jleaf.logger = self.logger
        else:
            for jkey in self.dict_leaves[iout].keys():
                jleaf = self.load_leaf(iout, jkey)
                jleaf.logger = self.logger



    @_debug
    def find_cands(self, iout:int, jout:int, mcut=0.01, prefix="", level='debug', verbose=0):
        keys = list(self.dict_leaves[iout].keys())
        jhalos = None
        # backups = {}
        for key in keys:
            ileaf:Leaf = self.load_leaf(iout, key, prefix=prefix, level=level, verbose=verbose+1)
            # Calc, or not?
            calc = True
            if(jout>iout):
                if(ileaf.desc is not None):
                    if(jout in ileaf.desc[:,0]): calc=False
            if(jout<iout):
                if(ileaf.prog is not None):
                    if(jout in ileaf.prog[:,0]): calc=False

            # Main calculation
            if calc:
                if jhalos is None:
                    jhalos = self.read_part_halo_match(jout)
                pid = ileaf.pid
                pid = pid[pid <= len(jhalos)]
                hosts = jhalos[pid-1]
                hosts = hosts[hosts>0]
                hosts, count = np.unique(hosts, return_counts=True) # CPU?
                hosts = hosts[count/len(pid) > mcut]
                hosts = hosts[ np.isin(hosts, list(self.dict_leaves[jout].keys()),assume_unique=True) ]
                if len(hosts)>0:
                    otherleaves = [self.load_leaf(jout, iid, prefix=prefix, level=level, verbose=verbose+1) for iid in hosts]
                    ids, scores = ileaf.calc_score(jout, otherleaves, prefix=f"<{ileaf._name}>",level='debug', verbose=verbose+1) # CPU?
                else:
                    ids = np.array([[jout, 0]])
                    scores = np.array([[-10, -10, -10, -10, -10]])
                if jout<iout:
                    ileaf.prog = ids if ileaf.prog is None else np.vstack((ileaf.prog, ids))
                    ileaf.prog_score = scores if ileaf.prog_score is None else np.vstack((ileaf.prog_score, scores))
                    ileaf.changed = True
                elif jout>iout:
                    ileaf.desc = ids if ileaf.desc is None else np.vstack((ileaf.desc, ids))
                    ileaf.desc_score = scores if ileaf.desc_score is None else np.vstack((ileaf.desc_score, scores))
                    ileaf.changed = True
                else:
                    raise ValueError(f"Same output {iout} and {jout}!")
            
            # No need to calculation
            else:
                if jout<iout:
                    arg = ileaf.prog[:,0]==jout
                    ids = ileaf.prog[arg]
                    scores = ileaf.prog_score[arg]
                elif jout>iout:
                    arg = ileaf.desc[:,0]==jout
                    ids = ileaf.desc[arg]
                    scores = ileaf.desc_score[arg]
                else:
                    raise ValueError(f"Same output {iout} and {jout}!")
                
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
            if(verbose <= self.p.verbose):self.print(msg, level=level)

    @_debug
    def finalize(self, iout:int, prefix="", level='info', verbose=0):
        prefix2 = f"[Reduce Backup file] ({iout})"
        if not os.path.isfile(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle"):
            raise FileNotFoundError(f"`{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle` is not found!")
        leaves = pklload(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle")
        if isinstance(leaves, dict):
            keys = list(leaves.keys())
            count = 0
            for key in keys:
                gals = leaves[key]['gal'] if count==0 else np.hstack((gals, leaves[key]['gal']))
                count += 1
            pklsave(gals, f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}.pickle", overwrite=True)
            self.print(f"{prefix2} Save `{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}.pickle`", level=level)
            self.out_of_use.append(iout)
            os.remove(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle")
            self.print(f"{prefix2} Remove `{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle`", level=level)
            del gals
        else:
            raise TypeError(f"Type of `{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle` is not dict!")

    @_debug
    def leaf_write(self, prefix="", level='info', verbose=0):
        iouts = list(self.dict_leaves.keys())
        for iout in iouts:
            if(iout in self.out_of_use):
                continue
            if not (os.path.isfile(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}.pickle")):
                prefix2 = f"[leaf_write]({iout})"

                keys = list(self.dict_leaves[iout].keys())
                leaves = {}
                if os.path.isfile(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle"):
                    leaves = pklload(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle")
                for key in keys:
                    leaf = self.load_leaf(iout, key, prefix=prefix, level=level, verbose=verbose+1)
                    if leaf.changed:
                        backup = leaves[key] if(key in leaves.keys()) else {}
                        leaves[key] = leaf.selfsave(backup=backup)
                pklsave(leaves, f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle", overwrite=True)
                self.print(f"{prefix2} Write `{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle`", level=level)
                del leaves

    @_debug
    def leaf_read(self, iout:int, prefix="", level='info', verbose=0):
        leaves:dict=None
        if os.path.isfile(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle"):
            leaves = pklload(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout:05d}_temp.pickle")
        self.load_snap(iout, prefix=prefix, verbose=verbose+1) # INEFFICIENT
        gals = self.load_gals(iout, galid='all', prefix=prefix, verbose=verbose+1)
        
        if(leaves is None):
            if(self.p.galaxy):
                self.mainlog.info(f"[Queue] {len(gals)} {self.galstr}s at {iout}")
            else:
                try:
                    self.mainlog.info(f"[Queue] {howmany(gals['mcontam']/gals['m'] < self.p.fcontam, True)}/{len(gals)} {self.galstr}s at {iout}")
                except:
                    self.mainlog.info(f"[Queue] {len(gals)} {self.galstr}s at {iout}")
        gals = self.dict_gals[iout]['id'] if self.p.galaxy else self.dict_gals[iout]['id'][self.dict_pure[iout]]
        for galid in gals:
            if(galid in self.dict_leaves[iout].keys()):
                if(self.dict_leaves[iout][galid] is not None):
                    continue
            backup:dict = None
            if (leaves is not None):
                if galid in leaves.keys():
                    backup = leaves[galid]
            if (backup is None) and (os.path.exists(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_leaf_{galid}.pkl")):
                backup = pklload(f"{self.p.resultdir}/by-product/{self.p.logprefix}{iout}_leaf_{galid}.pkl")
            self.load_leaf(iout, galid, backup=backup, backups=leaves, prefix=prefix, verbose=verbose+1)
            if(self.p.loadall):
                break
        self.flush(iout, prefix=prefix)
        leaves = None; del leaves
        backup = None; del backup









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
        prefix = f"{prefix}[{func.__name__}]"; kwargs["prefix"]=prefix
        level = kwargs.pop("level","debug"); kwargs["level"]=level

        if ontime:
            clock=timer(text=prefix, logger=logger, verbose=verbose, level=level)
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
            if self.cat['mcontam']/self.cat['m'] > self.base.p.fcontam:
                self.contam = True
            self._name = f"L{self.id} at {self.iout} ({int(100*self.cat['mcontam']/self.cat['m'])}%)"
        except:
            pass
        
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
        for key_iout in self.saved_matchrate.keys():
            for key_id in self.saved_matchrate[key_iout].keys():
                val += 0 if(self.saved_matchrate[key_iout][key_id] is None) else self.saved_matchrate[key_iout][key_id][1].nbytes
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
    def _calc_matchrate(self:'Leaf', otherleaf:'Leaf', prefix="", level='debug', verbose=0) -> float:
        ind = large_isin(self.pid, otherleaf.pid)
        if not True in ind:
            val = -1
        else:
            val = np.sum( self.pweight[ind] )
        jout = otherleaf.iout
        if not jout in self.saved_matchrate.keys():
            self.saved_matchrate[jout] = {}
            self.changed = True
        if not otherleaf.id in self.saved_matchrate[jout].keys():
            self.saved_matchrate[jout][otherleaf.id] = (val, ind)
            self.changed = True
        return val, ind
    
    def calc_matchrate(self, otherleaf:'Leaf', prefix="", level='debug', verbose=0) -> float:
        calc = True
        jout = otherleaf.iout
        if jout in self.saved_matchrate.keys():
            if otherleaf.id in self.saved_matchrate[jout].keys():
                val, ind = self.saved_matchrate[jout][otherleaf.id]
                calc = False
        if calc:
            val, ind = self._calc_matchrate(otherleaf, prefix=prefix, level=level, verbose=verbose)
        return val, ind

    @_debug_leaf
    def calc_bulkmotion(self, checkind:list[bool]=None, prefix="", level='debug', verbose=0):
        if checkind is None:
            checkind = np.full(self.nparts, True)

        weights = self.pweight[checkind]
        weights /= np.sum(weights)
        vx = nbsum( self.pvx[checkind], weights ) - self.cat['vx']
        vy = nbsum( self.pvy[checkind], weights ) - self.cat['vy']
        vz = nbsum( self.pvz[checkind], weights ) - self.cat['vz']

        return np.array([vx, vy, vz])

    @_debug_leaf
    def _calc_veloffset(self:'Leaf', otherleaf:'Leaf', selfind:list[bool]=None, otherind:list[bool]=None, prefix="", level='debug', verbose=0) -> float:
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