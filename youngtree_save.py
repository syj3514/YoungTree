#########################################################
###############         Import          #################
#########################################################
import numpy as np
import matplotlib.pyplot as plt
import sys
if not "/home/jeon/jeonpkg" in sys.path:
    sys.path.append("/home/jeon/jeonpkg")
    print("/home/jeon/jeonpkg update")
from importlib import reload
import jeon as jn
import jfiles as jf
import jplots as jp
from rur import uri, uhmi, painter, drawer

import os
# from rur import uri, uhmi, painter
import time
from matplotlib.ticker import MultipleLocator
import gc
from collections.abc import Iterable
import inspect
import psutil
gc.collect()
import logging
import copy

import traceback

#########################################################
###############         Minor Function          #########
#########################################################
is_sorted = lambda a: np.all(a[:-1] <= a[1:])

class timer():
    def __init__(self, unit="sec",text="", verbose=2, debugger=None):
        self.ref = time.time()
        self.units = {"ms":1/1000, "sec":1, "min":60, "hr":3600}
        self.corr = self.units[unit]
        self.unit = unit
        self.text = text
        self.verbose=verbose
        self.debugger=debugger
        
        if self.verbose>0:
            print(f"{text} START")
        if self.debugger is not None:
            self.debugger.info(f"{text} START")
    
    def done(self):
        elapse = time.time()-self.ref
        if self.verbose>0:
            print(f"{self.text} Done ({elapse/self.corr:.3f} {self.unit})")
        if self.debugger is not None:
            self.debugger.info(f"{self.text} Done ({elapse/self.corr:.3f} {self.unit})")

def dprint(msg, debugger):
    debugger.debug(msg)
def MB():
    return psutil.Process().memory_info().rss / 2 ** 20
def GB():
    return psutil.Process().memory_info().rss / 2 ** 30


#########################################################
###############         Tree Class                #######
#########################################################
class Treebase():
    def __init__(self, simmode='hagn', flush_GB=50, verbose=2, debugger=None, loadall=False, prefix=""):
        func = f"[__Treebase__]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=verbose, debugger=debugger)

        self.iniGB = GB()
        self.iniMB = MB()
        self.flush_GB = flush_GB
        self.simmode = simmode
        self.verbose = verbose
        self.debugger = debugger
        if simmode[0] == 'h':
            self.rurmode = 'hagn'
            self.repo = f"/storage4/Horizon_AGN/"
        elif simmode[0] == 'y':
            self.rurmode = 'yzics'
            self.repo = f"/storage3/Clusters/{simmode[1:]}"
        self.loadall = loadall
        self.nout, self.nstep, self.zred, self.aexp, self.gyr = jn.pklload(f"/storage6/jeon/data/{self.simmode}/{self.simmode}_nout_nstep_zred_aexp_gyr.pickle")
        self.dm_nout, self.dm_nstep, self.dm_zred, self.dm_aexp, self.dm_gyr = jn.pklload(f"/storage6/jeon/data/{self.simmode}/dm_{self.simmode}_nout_nstep_zred_aexp_gyr.pickle")
        self.dict_snap = {} # in {iout}, RamsesSnapshot object
        self.dict_star = {} # in {iout}, in {galid}, Particle object
        self.dict_dm = {} # in {iout}, in {haloid}, Particle object
        self.dict_cell = {} # not used
        self.dict_gals = {"galaxymakers":{}, "gmpids":{}} # in each key, in {iout}, Recarray obejct
        self.dict_halos = {"halomakers":{}, "hmpids":{}} # in each key, in {iout}, Recarray obejct
        self.saved_out = {"snap":[], "gal":[], "halo":[], "star":[], "dm":[], "cell":[]}
        self.running_mp = False
        self.branches_queue = {}
        gc.collect()

        clock.done()

    def summary(self, isprint=False):
        text = "\n[Tree Data Report]\n"
        text+= "\n>>> Snapshot\n"
        a = [f"{key} | " for key in self.dict_snap.keys()]
        for ia in a:
            text+= ia
        text+= "\n>>> Star\n"
        for key in self.dict_star.keys():
            keys = list(self.dict_star[key].keys())
            text += f"\t{key}: {len(self.dict_star[key])} gals with {np.sum([len(self.dict_star[key][ia]['id']) for ia in keys])} stars\n"
            for ikey in keys:
                text += f"\t\tID{ikey:05d} Nparts{len(self.dict_star[key][ikey]['id'])}\n"
        text+= ">>> DM\n"
        for key in self.dict_dm.keys():
            keys = list(self.dict_dm[key].keys())
            text += f"\t{key}: {len(self.dict_dm[key])} halos with {np.sum([len(self.dict_dm[key][ia]['id']) for ia in keys])} DMs\n"
            for ikey in keys:
                text += f"\t\tID{ikey:05d} Nparts{len(self.dict_dm[key][ikey]['id'])}\n"
        text+= ">>> GalaxyMaker\n"
        for key in self.dict_gals["galaxymakers"].keys():
            text += f"\t{key}: {len(self.dict_gals['galaxymakers'][key])} gals with {np.sum([len(ia) for ia in self.dict_gals['gmpids'][key]])} stars\n"
        text+= ">>> HaloMaker\n"
        for key in self.dict_halos["halomakers"].keys():
            text += f"\t{key}: {len(self.dict_halos['halomakers'][key])} halos with {len(self.dict_halos['hmpids'][key])} DMs\n"
        text+= f">>> Used Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB\n"
        
        if isprint:
            print(text)
        return text
        
        
    def make_branches(self, iout, galids=None, galaxy=True, prefix=""):
        # Subject to queue
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        if galaxy:
            gals = self.load_gal(iout, galid=galids, prefix=prefix)
        else:
            gals = self.load_halo(iout, haloid=galids, prefix=prefix)

        for gal in gals:
            self.branches_queue[gal['id']]=Branch(gal, self, galaxy=galaxy, mode=self.simmode, verbose=self.verbose, prefix=prefix, debugger=self.debugger)

        clock.done()

    def queue(self, iout, gals, treeleng=None, galaxy=True, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        self.make_branches(iout, gals['id'], galaxy=galaxy, prefix=prefix)
        self.debugger.info(f"{prefix}\n{self.summary()}")
        try:
            go=True
            ntree = 1
            while go:
                keys = list( self.branches_queue.keys() )
                for key in keys:
                    go = False
                    iprefix = prefix+f"<{key:05d}> "
                    igo = self.branches_queue[key].do_onestep(prefix=iprefix)
                    igo = igo & self.branches_queue[key].go
                    if not igo:
                        self.branches_queue[key].selfsave()
                        self.branches_queue[key] = None
                        del self.branches_queue[key]
                    else: go=True
                    gc.collect()
                self.flush_auto(prefix=prefix)
                self.debugger.info(f"{prefix}\n{self.summary()}")
                ntree += 1
                if treeleng is not None:
                    if ntree >= treeleng:
                        go=False
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            self.debugger.error(traceback.format_exc())
            self.debugger.error(e)
            self.debugger.error(self.summary())

        clock.done()


    def check_outs(self, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        self.saved_out["snap"] = list(self.dict_snap.keys())
        self.saved_out["gal"] = list(self.dict_gals["galaxymakers"].keys())
        self.saved_out["halo"] = list(self.dict_halos["halomakers"].keys())
        self.saved_out["star"] = list(self.dict_star.keys())
        self.saved_out["dm"] = list(self.dict_dm.keys())
        self.saved_out["cell"] = list(self.dict_cell.keys())
        # clock.done()

    def flush_auto(self, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        reftot = MB()

        cout = -1
        keys = list( self.dict_snap.keys() )
        if len(keys)>0:
            cout = max(np.min(keys), cout)
        keys = list( self.dict_gals["galaxymakers"].keys() )
        if len(keys)>0:
            cout = max(np.min(keys), cout)
        keys = list( self.dict_halos["halomakers"].keys() )
        if len(keys)>0:
            cout = max(np.min(keys), cout)
        keys = list( self.dict_star.keys() )
        if len(keys)>0:
            cout = max(np.min(keys), cout)
        keys = list( self.dict_dm.keys() )
        if len(keys)>0:
            cout = max(np.min(keys), cout)
        keys = list( self.dict_cell.keys() )
        if len(keys)>0:
            cout = max(np.min(keys), cout)
        
        # Snapshot
        keys = list( self.dict_snap.keys() )
        if len(keys)>0:
            for iout in keys:
                if iout > cout+4:
                    refmem = MB()
                    self.dict_snap[iout].clear()
                    self.dict_snap[iout] = None
                    del self.dict_snap[iout]
                    gc.collect()
                    self.debugger.info(f"* [flush][snapshot] remove iout={iout} ({refmem-MB():.2f} MB saved)")
        # GalaxyMaker
        keys = list( self.dict_gals["galaxymakers"].keys() )
        if len(keys)>0:
            for iout in keys:
                if iout > cout+4:
                    refmem = MB()
                    self.dict_gals['galaxymakers'][iout] = None
                    self.dict_gals['gmpids'][iout] = None
                    del self.dict_gals['galaxymakers'][iout]
                    del self.dict_gals['gmpids'][iout]
                    gc.collect()
                    self.debugger.info(f"* [flush][GalaxyMaker] remove iout={iout} ({refmem-MB():.2f} MB saved)")
        # HaloMaker
        keys = list( self.dict_halos["halomakers"].keys() )
        if len(keys)>0:
            for iout in keys:
                if iout > cout+4:
                    refmem = MB()
                    self.dict_halos['halomakers'][iout] = None
                    self.dict_halos['hmpids'][iout] = None
                    del self.dict_halos['halomakers'][iout]
                    del self.dict_halos['hmpids'][iout]
                    gc.collect()
                    self.debugger.info(f"* [flush][HaloMaker] remove iout={iout} ({refmem-MB():.2f} MB saved)")
        # Star
        keys = list( self.dict_star.keys() )
        if len(keys)>0:
            for iout in keys:
                if iout > cout+4:
                    jkeys = list(self.dict_star[iout].keys())
                    for galid in jkeys:
                        refmem = MB()
                        self.dict_star[iout][galid].table = []
                        self.dict_star[iout][galid].snap.clear()
                        self.dict_star[iout][galid] = None
                        del self.dict_star[iout][galid]
                        gc.collect()
                        self.debugger.info(f"* [flush][Star] remove iout={iout} gal={galid} ({refmem-MB():.2f} MB saved)")
                    refmem = MB()
                    self.dict_star[iout] = None
                    del self.dict_star[iout]
                    gc.collect()
                    self.debugger.info(f"* [flush][Star] remove iout={iout} garbage ({refmem-MB():.2f} MB saved)")
        # DM
        keys = list( self.dict_dm.keys() )
        if len(keys)>0:
            for iout in keys:
                if iout > cout+4:
                    jkeys = list(self.dict_dm[iout].keys())
                    for haloid in jkeys:
                        refmem = MB()
                        self.dict_dm[iout][haloid].table = []
                        self.dict_dm[iout][haloid].snap.clear()
                        self.dict_dm[iout][haloid] = None
                        del self.dict_dm[iout][haloid]
                        gc.collect()
                        self.debugger.info(f"* [flush][DM] remove iout={iout} halo={haloid} ({refmem-MB():.2f} MB saved)")
                    refmem = MB()
                    self.dict_dm[iout] = None
                    del self.dict_dm[iout]
                    gc.collect()
                    self.debugger.info(f"* [flush][DM] remove iout={iout} garbage ({refmem-MB():.2f} MB saved)")
        # Cell
        keys = list( self.dict_cell.keys() )
        if len(keys)>0:
            for iout in keys:
                if iout > cout+4:
                    self.dict_cell[iout] = None
                    del self.dict_cell[iout]
                    gc.collect()
        
        self.check_outs(prefix=prefix)
        self.debugger.info(f"* [flush][Total] {reftot-MB():.2f} MB saved")

        clock.done()

    def load_snap(self, iout, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        if psutil.Process().memory_info().rss / 2 ** 30 > self.flush_GB+self.iniGB:
            self.flush_auto(prefix=prefix)

        if not iout in self.dict_snap.keys():
            self.dict_snap[iout] = uri.RamsesSnapshot(self.repo, iout, mode=self.rurmode, path_in_repo='snapshots')
        if not iout in self.dict_star.keys():
            self.dict_star[iout] = {}
        if not iout in self.dict_dm.keys():
            self.dict_dm[iout] = {}
        if not iout in self.dict_cell.keys():
            self.dict_cell[iout] = {}
        self.check_outs(prefix=prefix)

        # clock.done()
        return self.dict_snap[iout]

    def load_gal(self, iout, galid, return_part=False, prefix=""):
        # func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        if psutil.Process().memory_info().rss / 2 ** 30 > self.flush_GB+self.iniGB:
            self.flush_auto(prefix=prefix)

        if not iout in self.dict_gals["galaxymakers"].keys():
            func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
            # clock2 = timer(text=prefix+"[GalaxyMaker load]", verbose=self.verbose, debugger=self.debugger)
            snap = self.load_snap(iout, prefix=prefix)
            gm, gmpid = uhmi.HaloMaker.load(snap, galaxy=True, load_parts=True)
            self.dict_gals["galaxymakers"][iout] = copy.deepcopy(gm)
            cumparts = np.insert(np.cumsum(gm["nparts"]), 0, 0)
            gmpid = [gmpid[ cumparts[i]:cumparts[i+1] ] for i in range(len(gm))]
            self.dict_gals["gmpids"][iout] = copy.deepcopy(gmpid)
            del gm; del gmpid; del cumparts
            gc.collect()
            if self.loadall:
                self.debugger.info(f"{prefix} *** loadall=True: loadall stars...")
                snap.box = np.array([[0, 1], [0, 1], [0, 1]])
                snap.get_part()
                for gal in self.dict_gals["galaxymakers"][iout]:
                    self.load_star(iout, gal['id'], prefix=prefix, silent=True)
                self.debugger.info(f"{prefix} *** loadall=True: loadall stars Done")
            self.check_outs(prefix=prefix)
            # clock2.done()

        # clock.done()
        if isinstance(galid,str):
            if galid=='all':
                return self.dict_gals["galaxymakers"][iout]
        elif isinstance(galid, Iterable):
            a = np.hstack([self.dict_gals["galaxymakers"][iout][ia-1] for ia in galid])
            if return_part:
                b = [self.dict_gals["gmpids"][iout][ia-1] for ia in galid]
                return a, b
            return a
        else:
            if return_part:
                return self.dict_gals["galaxymakers"][iout][galid-1], self.dict_gals["gmpids"][iout][galid-1]
            return self.dict_gals["galaxymakers"][iout][galid-1]
    
    def load_halo(self, iout, haloid, return_part=False, prefix=""):
        # func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        if psutil.Process().memory_info().rss / 2 ** 30 > self.flush_GB+self.iniGB:
            self.flush_auto(prefix=prefix)

        if not iout in self.dict_halos["halomakers"].keys():
            func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
            # clock2 = timer(text=prefix+"[HaloMaker load]", verbose=self.verbose, debugger=self.debugger)
            snap = self.load_snap(iout, prefix=prefix)
            gm, gmpid = uhmi.HaloMaker.load(snap, galaxy=False, load_parts=True)
            self.dict_halos["halomakers"][iout] = copy.deepcopy(gm)
            cumparts = np.insert(np.cumsum(gm["nparts"]), 0, 0)
            gmpid = [gmpid[ cumparts[i]:cumparts[i+1] ] for i in range(len(gm))]
            self.dict_halos["hmpids"][iout] = copy.deepcopy(gmpid)
            del gm; del gmpid; del cumparts
            gc.collect()
            if self.loadall:
                self.debugger.info(f"{prefix} *** loadall=True: loadall DMs...")
                snap.box = np.array([[0, 1], [0, 1], [0, 1]])
                snap.get_part()
                for halo in self.dict_halos["halomakers"][iout]:
                    self.load_dm(iout, halo['id'], prefix=prefix, silent=True)
                self.debugger.info(f"{prefix} *** loadall=True: loadall DMs Done")
            self.check_outs(prefix=prefix)
            # clock2.done()

        # clock.done()
        if haloid == 'all':
            return self.dict_halos["halomakers"][iout]
        if return_part:
            return self.dict_halos["halomakers"][iout][haloid-1], self.dict_halos["hmpids"][iout][haloid-1]
        return self.dict_halos["halomakers"][iout][haloid-1]

    def load_star(self, iout, galid, prefix="", silent=False):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        if psutil.Process().memory_info().rss / 2 ** 30 > self.flush_GB+self.iniGB:
            self.flush_auto(prefix=prefix)

        snap = self.load_snap(iout, prefix=prefix)
        gal, gpid = self.load_gal(iout, galid, return_part=True, prefix=prefix)
        if not galid in self.dict_star[iout].keys():
            if snap.part_data is not None:
                if snap.part_data.nbytes / 2**30 > self.flush_GB:
                    self.debugger.info(prefix+f" snap.clear!! ({snap.part_data.nbytes / 2**30:.2f} GB)")
                    snap.clear()
            leng = 0
            scale = 1
            while leng < len(gpid):
                snap.set_box_halo(gal, 1.1*scale, use_halo_radius=True, radius_name='r')
                if not silent:
                    clock2 = timer(text=prefix+"[get_part]", verbose=self.verbose, debugger=self.debugger)
                snap.get_part()
                if not silent:
                    clock2.done()
                try:
                    star = copy.deepcopy(snap.part['star'].table)
                    star['id'] = np.abs(star['id'])
                    if jn.atleast_isin(star['id'], gpid):
                        star = star[jn.large_isin(star['id'], gpid)]
                        leng = len(star['id'])
                    else:
                        leng = 0
                except: # didin't found star??
                    leng = 0
                scale *= 2
                if scale > 130:
                    break
            star = uri.RamsesSnapshot.Particle(star, snap)
            self.dict_star[iout][galid] = star
            self.check_outs(prefix=prefix)
        
        # clock.done()
        return self.dict_star[iout][galid]

    def load_dm(self, iout, haloid, prefix="", silent=False):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        if psutil.Process().memory_info().rss / 2 ** 30 > self.flush_GB+self.iniGB:
            self.flush_auto(prefix=prefix)

        snap = self.load_snap(iout, prefix=prefix)
        halo, hpid = self.load_halo(iout, haloid, return_part=True)
        if not haloid in self.dict_dm[iout].keys():
            if snap.part_data is not None:
                if snap.part_data.nbytes / 2**30 > self.flush_GB:
                    self.debugger.info(prefix+f" snap.clear!! ({snap.part_data.nbytes / 2**30:.2f} GB)")
                    snap.clear()
            ################## not yet
            snap.set_box_halo(halo, 1.1, use_halo_radius=True, radius_name='r')
            if not silent:
                clock2 = timer(text=prefix+"[get_part]", verbose=self.verbose, debugger=self.debugger)
            snap.get_part()
            if not silent:
                clock2.done()
            dm = copy.deepcopy(snap.part['dm'])
            dm['id'] = np.abs(dm['id'])
            dm = dm[jn.large_isin(dm['id'], hpid)]
            dm = uri.RamsesSnapshot.Particle(dm, snap)
            self.dict_dm[iout][haloid] = dm
            self.check_outs(prefix=prefix)

        clock.done()
        return self.dict_dm[iout][haloid]


#########################################################
###############         Leaf Class                #######
#########################################################
class Leaf():
    def __init__(self, gal, BranchObj, DataObj, verbose=1, prefix="", debugger=None, **kwargs):
        func = f"[__Leaf__]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=verbose, debugger=debugger)
        self.debugger=debugger
        self.verbose = verbose

        self.branch = BranchObj
        self.data = DataObj
        self.mode, self.galaxy = self.branch.mode, self.branch.galaxy
        self.gal_gm = gal
        self.galid, self.iout = self.gal_gm['id'], self.gal_gm['timestep']
        self.istep = jn.out2step(self.iout, galaxy=self.galaxy, mode=self.mode)#
        
        self.havefats = False
        self.refer = 6
        self.part = None
        prefix += f"<ID{self.galid}:iout(istep)={self.iout}({self.istep})>"
        self.load_parts(prefix=prefix)
        self.importance(prefix=prefix, usevel=True)

        # clock.done()

    def clear(self):
        self.part = []

    def load_parts(self, prefix=""):
        # Subject to `__init__`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        if self.galaxy:
            self.part = self.data.load_star(self.iout, self.galid, prefix=prefix)
        else:
            self.part = self.data.load_dm(self.iout, self.galid, prefix=prefix)
        self.debugger.debug(prefix+f" [ID{self.galid} iout{self.iout}] Nparts={self.gal_gm['nparts']}({len(self.part['x'])})")

        clock.done()
    

    def importance(self, prefix="", usevel=True):
        # Subject to `__init__`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
    
        cx, cy, cz = self.gal_gm['x'],self.gal_gm['y'],self.gal_gm['z']
        dist = jn.distance3d(cx,cy,cz, self.part['x'], self.part['y'], self.part['z']) / self.gal_gm['rvir']
        if usevel:
            try:
                cvx, cvy, cvz = self.gal_gm['vx'],self.gal_gm['vy'],self.gal_gm['vz']
                vels = jn.distance3d(cvx,cvy,cvz, self.part['vx'], self.part['vy'], self.part['vz'])
                vels /= np.std(vels)
                dist = np.sqrt( dist**2 + vels**2 )
            except Warning as e:
                print("WARNING!! in importance")
                self.debugger.warning("########## WARNING #########")
                self.debugger.warning(e)
                self.debugger.warning(f"gal velocity {cvx}, {cvy}, {cvz}")
                self.debugger.warning(f"len parts {len(self.part['x'])}")
                self.debugger.warning(f"vels first tens {vels[:10]}")
                self.debugger.warning(f"vels std {np.std(vels)}")
                self.debugger.warning(self.summary())
                breakpoint()
                raise ValueError("velocity wrong!")
        table = jn.addcolumn(self.part.table, "dist", dist, dtype='<f8')
        self.part = uri.RamsesSnapshot.Particle(table, self.part.snap)

        # clock.done()
    

    def load_fatids(self, igals, njump=0, masscut_percent=1, nfat=5, prefix="", **kwargs):
        # Subject to `find_candidates`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        iout, istep = jn.ioutistep(igals[0], galaxy=self.galaxy, mode=self.mode)
        jout = jn.step2out(istep-1-njump, galaxy=self.galaxy, mode=self.mode)
        agals = self.data.load_gal(jout, 'all', return_part=False, prefix=prefix)
        dt = jn.timeconversion(iout, final='gyr', mode=self.mode, galaxy=self.mode) - jn.timeconversion(jout, final='gyr', mode=self.mode, galaxy=self.mode)
        dt *= 1e9 * 365 * 24 * 60 * 60 # Gyr to sec
        fats = np.zeros(3).astype(int)
        for igal in igals:
            ivel = jn.rms(igal['vx'], igal['vy'], igal['vz'])
            isnap = self.data.load_snap(jout, prefix=prefix)
            radii = 5*max(igal['r'],1e-4) + 5*dt*ivel*isnap.unit['km']
            neighbors = jn.cut_sphere(agals, igal['x'], igal['y'], igal['z'], radii, both_sphere=True)
            # self.debugger.info(f"igal[{igal['id']}] len={len(neighbors)} in radii")
            
            if len(neighbors)>0:
                neighbors = neighbors[neighbors['m'] >= igal['m']*masscut_percent/100]
                # self.debugger.info(f"igal[{igal['id']}] len={len(neighbors)} after masscut {masscut_percent} percent")
                if len(neighbors) > nfat:
                    rate = np.zeros(len(neighbors))
                    gals, gmpids = self.data.load_gal(jout, neighbors['id'], return_part=True, prefix=prefix)
                    iout, istep = jn.ioutistep(igal, galaxy=self.galaxy, mode=self.mode)
                    _, checkpart = self.data.load_gal(iout, igal['id'], return_part=True, prefix=prefix)
                    ith = 0
                    for gal, gmpid in zip(gals, gmpids):
                        if jn.atleast_isin(checkpart, gmpid):
                            ind = jn.large_isin(checkpart, gmpid)
                            rate[ith] = jn.howmany(ind, True)/len(checkpart)
                        ith += 1
                    ind = rate>0

                    neighbors, rate = neighbors[ind], rate[ind]
                    # self.debugger.info(f"igal[{igal['id']}] len={len(neighbors)} after crossmatch")
                    if len(neighbors) > 0:
                        if len(neighbors) > nfat:
                            arg = np.argsort(rate)
                            neighbors = neighbors[arg][-nfat:]
                            # self.debugger.info(f"igal[{igal['id']}] len={len(neighbors)} after score sorting")
                        fats = np.concatenate((fats, neighbors['id']))
                elif len(neighbors) > 0:
                    fats = np.concatenate((fats, neighbors['id']))
                else:
                    pass
            
        fats = np.concatenate((fats, np.zeros(3).astype(int)))
        
        clock.done()
        return np.unique( fats[fats>0] ), jout
    

    def find_candidates(self, masscut_percent=1, nstep=5, nfat=5, prefix="", **kwargs):
        ############################################################
        ########    ADD case when no father, jump beyond     ####### Maybe done?
        ############################################################
        if not self.havefats:
            func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
            clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

            igals = np.atleast_1d(self.gal_gm)
            njump=0
            for i in range(nstep):
                fatids, jout = self.load_fatids(igals, njump=njump, masscut_percent=masscut_percent, nfat=nfat, prefix=prefix, **kwargs)
                iout = jn.step2out(self.istep-1-i, galaxy=self.galaxy, mode=self.mode)
                self.debugger.info(f"*** iout(istep)={iout}({self.istep-1-i}), fats={fatids}")
                if len(fatids) == 0:
                    self.debugger.info("JUMP!!")
                    njump += 1
                    if not self.havefats:
                        self.havefats=False
                else:
                    njump = 0
                    self.havefats=True
                    self.branch.update_cands(iout, fatids, checkids=self.part['id'], prefix=prefix) # -> update self.branch.candidates & self.branch.scores
                    igals = self.data.load_gal(iout, fatids, return_part=False, prefix=prefix)

            clock.done()
        return self.havefats
    

    def calc_score(self, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        timekeys = self.branch.candidates.keys()
        for tout in timekeys:
            checkpart = self.part
            if self.galaxy:
                tgyr = jn.timeconversion(tout,start='nout', final='gyr', mode=self.mode, galaxy=self.galaxy)
                igyr = jn.timeconversion(self.iout, final='gyr', mode=self.mode, galaxy=self.galaxy)
                candidates = self.branch.candidates[tout]
                checkpart = self.part[ self.part['age','Gyr'] >= (igyr-tgyr) ]
            for i, ileaf in candidates.items():
                try:
                    score1 = self.calc_matchrate(ileaf, checkpart=checkpart, prefix=prefix) # importance weighted matchrate
                    score2 = 0  # Velocity offset
                    score3 = 0
                    if score1 > 0:
                        score2 = self.calc_velocity_offset(ileaf, prefix=prefix)
                        score3 = np.exp( -np.abs(np.log10(self.gal_gm['m']/ileaf.gal_gm['m'])) )   # Mass difference
                except Warning as e:
                    print("WARNING!! in calc_score")
                    breakpoint()
                    self.debugger.warning("########## WARNING #########")
                    self.debugger.warning(e)
                    self.debugger.warning(f"len parts {len(checkpart['m'])}")
                    self.debugger.warning(f"dtype {checkpart.dtype}")
                    self.debugger.warning(self.summary())
                    raise ValueError("velocity wrong!")
                self.branch.scores[tout][i] += score1+score2+score3
                self.branch.score1s[tout][i] += score1
                self.branch.score2s[tout][i] += score2
                self.branch.score3s[tout][i] += score3

        clock.done()


    def calc_matchrate(self, otherleaf, checkpart=None, prefix=""):
        # Subject to `calc_score`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        otherleaf.refer += 1
        if checkpart is None:
            checkpart = self.part
            if self.galaxy:
                tout = otherleaf.iout
                tgyr = jn.timeconversion(otherleaf.iout, final='gyr', mode=otherleaf.mode, galaxy=otherleaf.galaxy)
                igyr = jn.timeconversion(self.iout, final='gyr', mode=self.mode, galaxy=self.galaxy)
                checkpart = self.part[ self.part['age','Gyr'] >= (igyr-tgyr) ]
        if jn.atleast_isin(checkpart['id'], otherleaf.part['id']):
            ind = jn.large_isin(checkpart['id'], otherleaf.part['id'])
        else:
            return -1
        # clock.done()
        return np.sum( checkpart['m'][ind]/checkpart['dist'][ind] ) / np.sum( checkpart['m']/checkpart['dist'] )


    def calc_bulkmotion(self, checkpart=None, prefix="", **kwargs):
        # Subject to `calc_velocity_offset`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        if checkpart is None:
            checkpart = self.part

        try:
            vx = np.average(checkpart['vx', 'km/s'], weights=checkpart['m']/checkpart['dist'])
            vy = np.average(checkpart['vy', 'km/s'], weights=checkpart['m']/checkpart['dist'])
            vz = np.average(checkpart['vz', 'km/s'], weights=checkpart['m']/checkpart['dist'])
        except Warning as e:
            print("WARNING!! in calc_bulkmotion")
            breakpoint()
            self.debugger.warning("########## WARNING #########")
            self.debugger.warning(e)
            self.debugger.warning(f"len parts {len(checkpart['m'])}")
            self.debugger.warning(f"dtype {checkpart.dtype}")
            self.debugger.warning(self.summary())
            raise ValueError("velocity wrong!")

        # clock.done()
        return np.array([vx, vy, vz])

    
    def calc_velocity_offset(self, otherleaf, prefix="", **kwargs):
        # Subject to `calc_score`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        ind = jn.large_isin(otherleaf.part['id'], self.part['id'])
        if jn.howmany(ind, True) < 3:
            return 0
        else:
            refv = self.calc_bulkmotion(prefix=prefix)
            instar = otherleaf.part[ind]
            inv = self.calc_bulkmotion(checkpart=instar, prefix=prefix) - refv
            totv = np.array([otherleaf.gal_gm['vx'], otherleaf.gal_gm['vy'], otherleaf.gal_gm['vz']]) - refv
            # self.debugger.debug(f"instar velocity: {inv}")
            # self.debugger.debug(f"totstar velocity: {totv}")
            # self.debugger.debug(f"|delta v| = {np.sqrt( np.sum((totv - inv)**2) )}")
            # self.debugger.debug(f"|v_in|+|v_tot| = {np.linalg.norm(inv)+np.linalg.norm(totv)}")


        # clock.done()
        return 1 - np.sqrt( np.sum((totv - inv)**2) )/(np.linalg.norm(inv)+np.linalg.norm(totv))




#########################################################
###############         Branch Class                #####
#########################################################
class Branch():
    def __init__(self, root, DataObj, galaxy=True, mode='hagn', verbose=2, prefix="", debugger=None, **kwargs):
        func = f"[__Branch__]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=verbose, debugger=debugger)
        self.debugger=debugger
        self.verbose = verbose

        self.root = root
        self.galaxy = galaxy
        self.mode = mode
        self.rootout, self.rootstep = jn.ioutistep(self.root, galaxy=self.galaxy, mode=self.mode)
        if self.galaxy:
            self.key='star'
        else:
            self.key='dm'
        if mode[0]=='h':
            self.repo = f'/storage4/Horizon_AGN/'
            self.rurmode = "hagn"
        elif mode[0]=='y':
            self.repo = f'/storage3/Clusters/{mode[1:]}'
            self.rurmode = "yzics"
        else:
            raise ValueError(f"{mode} is not supported!")

        self.data = DataObj
                
        self.candidates = {} # dictionary of Leafs
        self.scores = {}
        self.score1s = {}
        self.score2s = {}
        self.score3s = {}
        self.rootleaf = self.gal2leaf(self.root, prefix=prefix)
        self.leaves = {self.rootout: self.root} # results
        self.leave_scores = {self.rootout: 1} # results
        self.secrecord = 0
        self.go = True

        clock.done()

    def selfsave(self, dir="./data/"):
        a = self.root['id']
        b = self.secrecord
        c = self.leaves
        d = self.leave_scores
        readme = "1) root galaxy, 2) total elapsed time, 3) tree branch results, 4) corresponding score based on matchrate(importance-weighted) & mass difference & velocity offset"
        fname = f"Branch_{self.mode}_{a:05d}.pickle"
        self.debugger.info(f"\n>>> GAL{a:05d} is saved as `{fname}`")
        self.debugger.info(f">>> Treeleng = {len(c.keys())} (elapsed {self.secrecord/60:.2f} min)\n")
        print(f"\n>>> GAL{a:05d} is saved as `{fname}`")
        print(f">>> Treeleng = {len(c.keys())} (elapsed {self.secrecord/60:.2f} min)\n")
        jn.pklsave((readme, self.root,b,c,d), dir+fname, overwrite=True)

    def summary(self, onlyreturn=False):
        text ="\n[Summary report]\n>>> Root:" + "\n\t"
        if not onlyreturn:
            print(">>> Root:")
        text += jn.printgal(self.root, mode=self.mode, onlyreturn=onlyreturn) + "\n"
        if not onlyreturn:
            print(">>> Current root:")
        text += ">>> Current root:" + "\n\t"
        text += jn.printgal(self.rootleaf.gal_gm, mode=self.mode, onlyreturn=onlyreturn) + "\n"
        if not onlyreturn:
            print(">>> Current branch:")
        text += ">>> Current branch:" + "\n\t"
        if not onlyreturn:
            print([f"{a}:{b['id']}" if a%5==4 else f"{a}:{b['id']}\n" for a,b in self.leaves.items()])
        text += "["
        for a,b in self.leaves.items():
            if a%5==4:
                text += f"At {a}: {b['id']} | \n\t"
            else:
                text += f"At {a}: {b['id']} | "
        text += "]\n"
        if not onlyreturn:
            print(">>> Candidates:")
        text += ">>> Candidates:" + "\n"
        for iout in self.candidates.keys():
            keys = [f"{ikey}({self.scores[iout][ikey]:.4f}={self.score1s[iout][ikey]:.2f}+{self.score2s[iout][ikey]:.2f}+{self.score3s[iout][ikey]:.2f})" for ikey in self.candidates[iout].keys()]
            if not onlyreturn:
                print(f"[{iout}]\t{keys}")
            text += f"[{iout}]\t{keys}" + "\n"
        if not onlyreturn:
            print(f">>> Current Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB")
        text += f">>> Current Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB\n"
        text += f">>> Elapsed time: {self.secrecord:.4f} sec\n"
        return text

    def execution(self, prefix="", treeleng=None, **kwargs):
        print(f"Log is written in {self.debugger.handlers[0].baseFilename}")
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        self.go = True
        while self.go:
            self.go = self.do_onestep(prefix=prefix, **kwargs)
            if treeleng is not None:
                if len(self.leaves.keys()) >= treeleng:
                    self.go=False
        clock.done()


    def do_onestep(self, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        ref = time.time()
        try:
            self.go = self.rootleaf.find_candidates(prefix=prefix, **kwargs)
            # dprint(f"*** go? {self.go}", self.debugger)
            if len(self.candidates.keys())>0:
                self.rootleaf.calc_score(prefix=prefix)
                self.choose_winner(prefix=prefix)
            else:
                self.go = False
            if not self.go:
                while len(self.candidates.keys())>0:
                    self.choose_winner(prefix=prefix)
            
            self.secrecord += time.time()-ref
            clock.done()
            self.debugger.info(f"{prefix}\n{self.summary(onlyreturn=True)}")
        except Warning as e:
            print("WARNING!! in do_onestep")
            breakpoint()
            self.debugger.warning("########## WARNING #########")
            self.debugger.warning(e)
            self.debugger.warning(self.summary())
            raise ValueError("velocity wrong!")
        return self.go
        # try:
        #     self.rootleaf.find_candidates(prefix=prefix, **kwargs)
        #     self.rootleaf.calc_score(prefix=prefix)
        #     self.choose_winner(prefix=prefix)
        # except Warning:
        #     self.debugger.warning("########## WARNING #########")
        breakpoint()
        #     self.debugger.warning(self.summary())

        

    def reset_branch(self, gal, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        Newleaf = self.gal2leaf(gal, prefix=prefix)
        keys = list( self.leaves.keys() )
        for key in keys:
            if key < Newleaf.iout:
                refmem = MB()
                self.leaves[key] = None
                del self.leaves[key]
                self.debugger.debug(f"* [Branch][Reset] remove gal at iout={key} ({refmem-MB():.2f} MB saved)")
        keys = list( self.candidates.keys() )
        for key in keys:
            if key < Newleaf.iout:
                ids = list( self.candidates[key].keys() )
                for iid in ids:
                    refmem = MB()
                    self.candidates[key][iid].clear()
                    del self.candidates[key][iid]
                    self.debugger.debug(f"* [Branch][Reset] remove {iid} leaf at iout={key} ({refmem-MB():.2f} MB saved)")
                del self.candidates[key]
        gc.collect()

        clock.done()

    def gal2leaf(self, gal, prefix=""):
        iout, istep = jn.ioutistep(gal, galaxy=self.galaxy, mode=self.mode)
        if not iout in self.candidates.keys():
            self.candidates[iout] = {}
            self.scores[iout] = {}
            self.score1s[iout] = {}
            self.score2s[iout] = {}
            self.score3s[iout] = {}
        if not gal['id'] in self.candidates[iout].keys():
            func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
            # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
            
            self.candidates[iout][gal['id']] = Leaf(
                gal, self, self.data,
                verbose=self.verbose-1, prefix=prefix, debugger=self.debugger)
            self.scores[iout][gal['id']] = 0
            self.score1s[iout][gal['id']] = 0
            self.score2s[iout][gal['id']] = 0
            self.score3s[iout][gal['id']] = 0

            # clock.done()
        return self.candidates[iout][gal['id']]


    def update_cands(self, iout, galids,checkids=None, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        if checkids is not None:
            gals, gmpids = self.data.load_gal(iout, galids, return_part=True, prefix=prefix)
            for gal, gmpid in zip(gals, gmpids):
                # dprint(f"*** <update_cands> ID{gal['id']}: {gal['nparts']} vs {len(gmpid)}", self.debugger)
                # dprint(f"*** <update_cands> common parts = {jn.howmany(jn.large_isin(gmpid, checkids), True)}", self.debugger)
                if jn.atleast_isin(gmpid, checkids):
                    self.gal2leaf(gal, prefix=prefix)
                    # dprint(f"*** <update_cands> ID{gal['id']} Gal becomes candidate!", self.debugger)
                else:
                    pass
        else:
            gals = self.data.load_gal(iout, galids, return_part=False, prefix=prefix)
            for gal in gals:
                # dprint(f"*** <update_cands> ID{gal['id']} Gal becomes candidate!", self.debugger)
                self.gal2leaf(gal, prefix=prefix)

        clock.done()


    def gals_from_candidates(self, iout, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        cands = self.candidates[iout]
        for key in cands.keys():
            cands[key].refer += 1
        gals = np.concatenate([cands[key].gal_gm for key in cands.keys()])

        clock.done()
        return gals
    
    
    def choose_winner(self, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        iout = np.max(list(self.candidates.keys()))
        winid=0; winscore=0
        for iid, iscore in self.scores[iout].items():
            if iscore > winscore:
                winid = iid
                winscore = iscore
        if winscore<=0:
            pass
        else:
            self.debugger.debug(f"** (winner in {iout}) go? {self.go}")
            self.debugger.debug(f"** [Choose_winner] candidates\n{self.candidates}")
            self.debugger.debug(f"** [Choose_winner] scores {self.scores[iout]}")
            self.debugger.debug(f"** [Choose_winner] winid={winid}, winscore={winscore}")
            self.debugger.debug(f"* [Choose_winner] root leaf {self.rootleaf.gal_gm['id']} at {self.rootleaf.gal_gm['timestep']}")
            self.rootleaf = self.candidates[iout][winid]
            self.debugger.debug(f"* [Choose_winner] --> {self.rootleaf.gal_gm['id']} at {self.rootleaf.gal_gm['timestep']}")
            self.leaves[iout] = self.candidates[iout][winid].gal_gm
            self.leave_scores[iout] = winscore
    
        ids = list( self.candidates[iout].keys() )
        for iid in ids:
            refmem = MB()
            if iid != winid:
                self.candidates[iout][iid].clear()
            del self.candidates[iout][iid]
            del self.scores[iout][iid]
            self.debugger.debug(f"* [Choose_winner] remove {iid} leaf&score at iout={iout} ({refmem-MB():.2f} MB saved)")
        del self.candidates[iout]
        del self.scores[iout]

        ikeys = list( self.candidates.keys() )
        for ikey in ikeys:
            jkeys = list( self.candidates[ikey].keys() )
            for jkey in jkeys:
                if ikey-iout > 2:
                    self.candidates[ikey][jkey].refer -= 1
                if self.candidates[ikey][jkey].refer <= 0 or self.scores[ikey][jkey] <= -1:
                    self.debugger.debug(f"ikey={ikey}, jkey={jkey}, refer={self.candidates[ikey][jkey].refer}, score={self.scores[ikey][jkey]}")
                    refmem = MB()
                    self.candidates[ikey][jkey].clear()
                    del self.candidates[ikey][jkey]
                    del self.scores[ikey][jkey]
                    self.debugger.debug(f"* [Choose_winner] remove non-referred {jkey} leaf&score at iout={ikey} ({refmem-MB():.2f} MB saved)")
            if len(self.candidates[ikey].keys())<1:
                del self.candidates[ikey]
                del self.scores[ikey]

        gc.collect()        
        clock.done()




#########################################################
###############         Targets                ##########
#########################################################
Data = None
gc.collect()
print(f"\n>>> Current Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB")
mode = input("Mode=?")
prefix = f"[{mode}]"
dm_nout, dm_nstep, dm_zred, dm_aexp, dm_gyr = jn.pklload(f'/storage6/jeon/data/{mode}/dm_{mode}_nout_nstep_zred_aexp_gyr.pickle')
nout, nstep, zred, aexp, gyr = jn.pklload(f'/storage6/jeon/data/{mode}/{mode}_nout_nstep_zred_aexp_gyr.pickle')
if mode[0]=='y':
    repo = f'/storage3/Clusters/{mode[1:]}'
    rurmode = 'yzics'
elif mode[0] == 'h':
    repo = '/storage4/Horizon_AGN'
    rurmode = 'hagn'
else:
    raise ValueError(f"{mode} is not supported!")
print(f"{prefix} {nout[-1]} ~ {nout[0]}")

print(f"\n{prefix} target galaxies load..."); ref = time.time()
iout = np.max(nout)
uri.timer.verbose = 0
snap_now = uri.RamsesSnapshot(repo, iout, path_in_repo='snapshots', mode=rurmode )
gals_now = uhmi.HaloMaker.load(snap_now, galaxy=True)
jn.printtime(ref, f"{prefix} {len(gals_now)} gals load done")
snap_now.clear()
print(f"\n>>> Current Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB")

targets = gals_now
# targets = gals_now[np.argmin(gals_now['m'])]
# targets = gals_now[11:15]
# targets = gals_now[248]
targets = np.atleast_1d(targets)
# for target in targets:
#     jn.printgal(target, mode=mode)


#########################################################
###############         Debugger                #########
#########################################################
# import warnings
# warnings.simplefilter('error')

debugger = None
fname = f"./output_{mode}.log"
if os.path.isfile(fname):
    os.remove(fname)
debugger = logging.getLogger(f"YoungTree_{mode}")
debugger.handlers = []
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
print(f"Allow {flush_GB:.2f} GB Memory")
MyTree = Treebase(simmode=mode, debugger=debugger, verbose=0, flush_GB=flush_GB, loadall=True)
print(f"\n\nSee {fname}\n\nRunning...\n")


#########################################################
###############         Execution                ########
#########################################################
MyTree.queue(iout, targets, galaxy=True)
print("\nDone\n")