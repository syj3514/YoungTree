from rur import uri, uhmi
import numpy as np
import sys
import os
import time
import gc
from collections.abc import Iterable
import inspect
import psutil
import logging
import copy
import traceback

from tree_utool import *
from tree_branch import Branch
from tree_leaf import Leaf

#########################################################
###############         Tree Class                #######
#########################################################
class Treebase():
    def __init__(self, simmode='hagn', galaxy=True, flush_GB=50, verbose=2, debugger=None, loadall=False, prefix=""):
        func = f"[__Treebase__]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=verbose, debugger=debugger)

        self.iniGB = GB()
        self.iniMB = MB()
        self.flush_GB = flush_GB
        self.simmode = simmode
        self.galaxy = galaxy
        if self.galaxy:
            self.partstr = "star"
            self.Partstr = "Star"
            self.galstr = "gal"
            self.Galstr = "GalaxyMaker"
        else:
            self.partstr = "DM"
            self.Partstr = "DM"
            self.galstr = "halo"
            self.Galstr = "HaloMaker"
        self.verbose = verbose
        self.debugger = debugger

        if simmode[0] == 'h':
            self.rurmode = 'hagn'
            self.repo = f"/storage4/Horizon_AGN"
        elif simmode[0] == 'y':
            self.rurmode = 'yzics'
            self.repo = f"/storage3/Clusters/{simmode[1:]}"
        elif simmode == 'nh':
            self.rurmode = 'nh'
            self.repo = "/storage6/NewHorizon"

        self.loadall = loadall
        self.nout = load_nout(mode=self.simmode, galaxy=self.galaxy)
        self.nstep = load_nstep(mode=self.simmode, galaxy=self.galaxy, nout=self.nout)
        self.dict_snap = {} # in {iout}, RamsesSnapshot object
        self.dict_part = {} # in {iout}, in {galid}, Particle object
        self.dict_gals = {"galaxymakers":{}, "gmpids":{}} # in each key, in {iout}, Recarray obejct
        self.dict_leaves = {} # in {iout}, in {galid}, Leaf object
        self.saved_out = {"snap":[], "gal":[], "part":[]}
        self.branches_queue = {}
        gc.collect()

        clock.done()

    def summary(self, isprint=False):
        temp = [f"{key} | " for key in self.dict_snap.keys()]
        tsnap = "".join(temp)
        
        temp = []
        for key in self.dict_part.keys():
            idict = self.dict_part[key]
            keys = list(idict.keys())
            temp += f"\t{key}: {len(idict)} {self.galstr}s with {np.sum([len(idict[ia]['id']) for ia in keys])} {self.partstr}s\n"
            # for ikey in keys:
            #     text += f"\t\tID{ikey:05d} Nparts{len(self.dict_part[key][ikey]['id'])}\n"
        tpart = "".join(temp)

        temp = []
        for key in self.dict_gals["galaxymakers"].keys():
            temp += f"\t{key}: {len(self.dict_gals['galaxymakers'][key])} {self.galstr}s with {np.sum([len(ia) for ia in self.dict_gals['gmpids'][key]])} {self.partstr}s\n"
        tgm = "".join(temp)

        temp = []
        for key in self.dict_leaves.keys():
            temp1 = np.array([0 if self.dict_leaves[key][iid].parents is None else len(self.dict_leaves[key][iid].parents) for iid in self.dict_leaves[key].keys()])
            temp += f"\t{key}: {len(self.dict_leaves[key])} leaves ({howmany(temp1, 0)} pruned, {howmany(temp1, 1)} single, {howmany(temp1>1, True)} multi)\n"
        tleaf = "".join(temp)
        
        text = f"\n[Tree Data Report]\n\n>>> Snapshot\n{tsnap}\n>>> {self.Partstr}\n{tpart}>>> {self.Galstr}\n{tgm}>>> Leaves\n{tleaf}\n>>> Used Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB\n"

        if isprint:
            print(text)
        return text
        
        
    def make_branches(self, iout, galids=None, interplay=False, prefix=""):
        # Subject to queue
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        gals = self.load_gal(iout, galid=galids, prefix=prefix)

        for gal in gals:
            self.branches_queue[gal['id']]=Branch(gal, self, galaxy=self.galaxy, mode=self.simmode, verbose=self.verbose, prefix=prefix, debugger=self.debugger, interplay=False)

        clock.done()

    def queue(self, iout, gals, treeleng=None, interplay=False, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        self.make_branches(iout, gals['id'], interplay=False, prefix=prefix)
        self.debugger.info(f"{prefix}\n{self.summary()}")
        try:
            go=True
            ntree = 1
            for jout in self.nout:
                keys = list( self.branches_queue.keys() )
                for key in keys:
                    go = False
                    iprefix = prefix+f"<{key:05d}> "
                    igo = self.branches_queue[key].do_onestep(jout, prefix=iprefix)
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
                if not go:
                    break
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
        self.saved_out["part"] = list(self.dict_part.keys())
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
        keys = list( self.dict_part.keys() )
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
                    self.debugger.info(f"* [flush][{self.Galstr}] remove iout={iout} ({refmem-MB():.2f} MB saved)")
        # Star
        keys = list( self.dict_part.keys() )
        if len(keys)>0:
            for iout in keys:
                if iout > cout+4:
                    jkeys = list(self.dict_part[iout].keys())
                    for galid in jkeys:
                        refmem = MB()
                        self.dict_part[iout][galid].table = []
                        self.dict_part[iout][galid].snap.clear()
                        self.dict_part[iout][galid] = None
                        del self.dict_part[iout][galid]
                        gc.collect()
                        self.debugger.info(f"* [flush][{self.Partstr}] remove iout={iout} {self.galstr}={galid} ({refmem-MB():.2f} MB saved)")
                    refmem = MB()
                    self.dict_part[iout] = None
                    del self.dict_part[iout]
                    gc.collect()
                    self.debugger.info(f"* [flush][{self.Partstr}] remove iout={iout} garbage ({refmem-MB():.2f} MB saved)")
        # Leaf
        keys = list( self.dict_leaves.keys() )
        if len(keys)>0:
            for iout in keys:
                if iout > cout+4:
                    jkeys = list(self.dict_leaves[iout].keys())
                    for galid in jkeys:
                        if len(self.dict_leaves[iout][galid].parents)==0:
                            refmem = MB()
                            self.dict_leaves[iout][galid].clear()
                            self.dict_leaves[iout][galid] = None
                            del self.dict_leaves[iout][galid]
                            gc.collect()
                            self.debugger.info(f"* [flush][Leaf] remove iout={iout} {self.galstr}={galid} ({refmem-MB():.2f} MB saved)")
                    if len(self.dict_leaves[iout].keys())==0:
                        refmem = MB()
                        self.dict_leaves[iout] = None
                        del self.dict_leaves[iout]
                        gc.collect()
                        self.debugger.info(f"* [flush][Leaf] remove iout={iout} garbage ({refmem-MB():.2f} MB saved)")

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
        if not iout in self.dict_part.keys():
            self.dict_part[iout] = {}
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
            gm, gmpid = uhmi.HaloMaker.load(snap, galaxy=self.galaxy, load_parts=True)
            self.dict_gals["galaxymakers"][iout] = copy.deepcopy(gm)
            cumparts = np.insert(np.cumsum(gm["nparts"]), 0, 0)
            gmpid = [gmpid[ cumparts[i]:cumparts[i+1] ] for i in range(len(gm))]
            self.dict_gals["gmpids"][iout] = copy.deepcopy(gmpid)
            del gm; del gmpid; del cumparts
            gc.collect()
            if self.loadall:
                self.debugger.info(f"{prefix} *** loadall=True: loadall {self.partstr}s...")
                snap.box = np.array([[0, 1], [0, 1], [0, 1]])
                snap.get_part()
                for gal in self.dict_gals["galaxymakers"][iout]:
                    self.load_part(iout, gal['id'], prefix=prefix, silent=True, galaxy=self.galaxy)
                self.debugger.info(f"{prefix} *** loadall=True: loadall {self.partstr}s Done")
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
    
    def load_leaf(self, iout, galid, branch, gal=None, prefix=""):
        # func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        if not iout in self.dict_leaves.keys():
            self.dict_leaves[iout] = {}
        

        if not galid in self.dict_leaves[iout].keys():
            func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
            # clock2 = timer(text=prefix+"[GalaxyMaker load]", verbose=self.verbose, debugger=self.debugger)
            if gal is None:
                gal = self.load_gal(iout, galid, prefix=prefix)
            self.dict_leaves[iout][galid] = Leaf(gal, branch, self, verbose=self.verbose-1, prefix=prefix, debugger=self.debugger, interplay=branch.interplay)
        
        if not branch.root['id'] in self.dict_leaves[iout][galid].parents:
            self.dict_leaves[iout][galid].branch = branch
            self.dict_leaves[iout][galid].parents += [branch.root['id']]

        return self.dict_leaves[iout][galid]


    def load_part(self, iout, galid, prefix="", silent=False, galaxy=True):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        if psutil.Process().memory_info().rss / 2 ** 30 > self.flush_GB+self.iniGB:
            self.flush_auto(prefix=prefix)

        snap = self.load_snap(iout, prefix=prefix)
        gal, gpid = self.load_gal(iout, galid, return_part=True, prefix=prefix)
        if not galid in self.dict_part[iout].keys():
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
                    if galaxy:
                        part = copy.deepcopy(snap.part['star'].table)
                    else:
                        part = copy.deepcopy(snap.part['dm'].table)
                    part['id'] = np.abs(part['id'])
                    if atleast_numba(part['id'], gpid):
                        part = part[large_isin(part['id'], gpid)]
                        leng = len(part['id'])
                    else:
                        leng = 0
                except: # didin't found part??
                    leng = 0
                scale *= 2
                if scale > 130:
                    break
            part = uri.RamsesSnapshot.Particle(part, snap)
            self.dict_part[iout][galid] = part
            self.check_outs(prefix=prefix)
        
        # clock.done()
        return self.dict_part[iout][galid]