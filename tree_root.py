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
    __slots__ = ['iniGB', 'iniMB', 'flush_GB', 'simmode', 'galaxy', 'logprefix', 'detail',
                'partstr', 'Partstr', 'galstr', 'Galstr','verbose', 'debugger',
                'rurmode', 'repo', 'initial_out', 'initial_galids', 'treeleng', 'interplay',
                'loadall','nout','nstep','dp',
                'dict_snap','dict_part','dict_gals','dict_leaves', 'branches_queue', 'prog']
    def __init__(self, simmode='hagn', galaxy=True, flush_GB=50, verbose=2, debugger=None, loadall=False, prefix="", prog=True, logprefix="output_", detail=True, dp=False):
        func = f"[__Treebase__]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=verbose, debugger=debugger)

        self.iniGB = GB()
        self.iniMB = MB()
        self.flush_GB = flush_GB
        self.simmode = simmode
        self.galaxy = galaxy
        self.logprefix=logprefix
        self.detail=detail
        self.dp=dp

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

        self.repo, self.rurmode, _ = mode2repo(simmode)

        self.loadall = loadall
        self.initial_galids = np.array([])
        self.treeleng = None
        self.interplay = False
        self.nout = load_nout(mode=self.simmode, galaxy=self.galaxy)
        self.nstep = load_nstep(mode=self.simmode, galaxy=self.galaxy, nout=self.nout)
        self.dict_snap = {} # in {iout}, RamsesSnapshot object
        self.dict_part = {} # in {iout}, in {galid}, Particle object
        self.dict_gals = {"galaxymakers":{}, "gmpids":{}} # in each key, in {iout}, Recarray obejct
        self.dict_leaves = {} # in {iout}, in {galid}, Leaf object
        self.branches_queue = {}
        self.prog = prog
        self.initial_out = np.max(self.nout) if prog else np.min(self.nout)
        gc.collect()

        clock.done()

    def export_backup(self, prefix=""):
        # func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func} <Root>"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        status = {
            "snapkeys":list(self.dict_snap.keys()), 
            "partkeys":list((iout, galid) for iout in self.dict_part.keys() for galid in self.dict_part[iout].keys()),
            "galskeys":list(iout for iout in self.dict_gals['galaxymakers'].keys()),
            "branches_queue":list(ib for ib in self.branches_queue.keys())
        }
        # clock.done()
        return status
    
    def import_backup(self, status, prefix=""): # may not used
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func} <Root>"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        for key in status["snapkeys"]:
            self.load_snap(key, prefix=prefix)
        
        for iout, galid in status["partkeys"]:
            self.load_part(iout, galid, galaxy=self.galaxy, prefix=prefix)
        
        for iout in status["galskeys"]:
            self.load_gal(iout, galid='all', prefix=prefix)
        
        galids = np.array([ib for ib in status['branches_queue']])
        self.make_branches(self.initial_out, galids=galids, prefix=prefix)

        clock.done()
        dprint_("\n", debugger=self.debugger)

    def leaf_summary(self):
        # Restore parents
        for ikey in self.dict_leaves.keys():
            for jkey in self.dict_leaves[ikey]:
                temp = [ib.rootid if ib is not None else None for ib in self.dict_leaves[ikey][jkey].otherbranch] + [self.dict_leaves[ikey][jkey].branch.rootid if self.dict_leaves[ikey][jkey].branch is not None else None]
                while None in temp:
                    temp.remove(None)
                temp = np.unique(temp)
                self.dict_leaves[ikey][jkey].parents = temp.tolist()
                # self.dict_leaves[ikey][jkey].report()


    def summary(self, isprint=False):
        temp = [f"{key}({sys.getsizeof(self.dict_snap[key].part_data) / 2**20:.2f} MB) | " for key in self.dict_snap.keys()]
        tsnap = "".join(temp)
        
        temp = []
        for key in self.dict_part.keys():
            idict = self.dict_part[key]
            keys = list(idict.keys())
            temp += f"\t{key}: {len(idict)} {self.galstr}s with {np.sum([len(idict[ia]['id']) for ia in keys])} {self.partstr}s\n"
        tpart = "".join(temp)

        temp = []
        for key in self.dict_gals["galaxymakers"].keys():
            if key in self.dict_gals["gmpids"].keys():
                temp += f"\t{key}: {len(self.dict_gals['galaxymakers'][key])} {self.galstr}s with {np.sum([len(ia) for ia in self.dict_gals['gmpids'][key]])} {self.partstr}s\n"
            else:
                temp += f"\t{key}: {len(self.dict_gals['galaxymakers'][key])} {self.galstr}s with 0 {self.partstr}s\n"
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
        
        
    def make_branches(self, iout, galids=None, prefix=""):
        # Subject to queue
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        gals = self.load_gal(iout, galid=galids, prefix=prefix)

        keys = list(self.branches_queue.keys())
        for gal in gals:
            if not gal['id'] in keys:
                self.branches_queue[gal['id']]=Branch(gal, self, galaxy=self.galaxy, mode=self.simmode, verbose=self.verbose, prefix=prefix, debugger=self.debugger, interplay=self.interplay, prog=self.prog)

        clock.done()

    def queue(self, iout, gals, treeleng=None, interplay=False, prefix="", backup_dict=None):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        ### Write queue information to attributes
        self.initial_out = iout
        self.initial_galids = gals['id']
        if (self.treeleng is None)&(treeleng is None):
            treeleng = len(self.nout)
        self.treeleng = treeleng
        self.interplay = interplay
        self.nout = self.nout[self.nout<=iout] if self.prog else self.nout[self.nout>=iout]

        ### Make branch objects
        self.make_branches(iout, gals['id'], prefix=prefix)
        self.debugger.info(f"{prefix}\n{self.summary()}")

        ### Main run
        try:
            ### Current status
            go=True
            ntree = 1

            ### Find progenitor or descendant
            if self.prog:
                outs = self.nout
            else:
                outs = self.nout[::-1]
            
            
            ### Load backup data and synchronize
            if backup_dict is not None:
                ## Read current status
                prefix_import = f"{prefix} <Import>"
                jout = backup_dict["Queue"]['jout']
                if self.prog:
                    outs = outs[outs < jout]
                else:
                    outs = outs[outs > jout]
                self.debugger.info(f"{prefix_import} Load from {jout} BACKUP")
                ntree = backup_dict["Queue"]['ntree']
                go = backup_dict["Queue"]['go']

                ## Read saved branch keys
                dprint_("\n", self.debugger)
                self.debugger.info(f"{prefix_import} Branch restoration...")
                branchkeys = np.asarray(list(backup_dict["Branch"].keys()))
                keys = list(self.branches_queue.keys())
                for key in keys:
                    if not key in branchkeys[:,1]:
                        self.branches_queue[key] = None
                        del self.branches_queue[key]
                    else:
                        self.branches_queue[key].import_backup(backup_dict["Branch"][(iout, key)])
                self.debugger.info(f"{prefix}\n{self.summary()}")

                ## Read saved leaf keys
                dprint_("\n", self.debugger)
                self.debugger.info(f"{prefix_import} Leaf restoration...")
                for iiout in backup_dict["Leaf"].keys():
                    for name in backup_dict["Leaf"][iiout]:
                        lout, lid = name
                        status_leaf = backup_dict["Leaf"][iiout][name]
                        branch = None
                        leaf = self.load_leaf(lout, lid, branch)
                        leaf.import_backup(status_leaf)
                
                ## Discard useless data
                self.flush_auto(jout=jout, prefix=prefix_import)

            ### Main loop
            for jout in outs:
                ## Make new log file
                jref = time.time()
                backup_dict = {}
                fname = make_logname(self.simmode, jout, logprefix=self.logprefix)
                self.debugger.handlers = []
                self.debugger = custom_debugger(fname, detail=self.detail)

                ## Loop for each branch
                keys = list( self.branches_queue.keys() )
                len_branch = len(keys)
                dprint_(f"Let's do_onestep for {len_branch} branches!", self.debugger)
                self.leaf_summary()
                go = False
                for key in keys:
                    # Find candidates & calculate scores
                    iprefix = prefix+f"<{key:05d}> "
                    igo = self.branches_queue[key].do_onestep(jout, prefix=iprefix)
                    igo = igo & self.branches_queue[key].go
                    if not igo:
                        self.branches_queue[key].selfsave()
                        self.branches_queue[key] = None
                        del self.branches_queue[key]
                    else: go=True
                    gc.collect()
                self.flush_auto(jout=jout, prefix=prefix)
                self.debugger.info(f"{prefix}\n{self.summary()}")
                
                ## Status update
                ntree += 1
                if treeleng is not None:
                    if ntree >= treeleng:
                        go=False
                if not go:
                    break

                clock_backup = timer(text="[export_backup] <Root>", verbose=self.verbose, debugger=self.debugger)
                status = self.export_backup()
                ## Save bakup data
                # Record current status
                backup_dict["Queue"] = {"ntree":ntree, "go":go, "jout":jout}
                backup_dict["Root"] = status
                clock_backup.done()
                # Record branch keys
                clock_backup = timer(text="[export_backup] <Branch>", verbose=self.verbose, debugger=self.debugger)
                backup_dict["Branch"] = {}
                keys = list( self.branches_queue.keys() )
                for key in keys:
                    branch = self.branches_queue[key]
                    name, status_ib = branch.export_backup()
                    backup_dict["Branch"][name] = status_ib
                clock_backup.done()
                # Record leaf keys
                clock_backup = timer(text="[export_backup] <Leaf>", verbose=self.verbose, debugger=self.debugger)
                backup_dict["Leaf"] = {}
                iouts = list( self.dict_leaves.keys() )
                for iiout in iouts:
                    backup_dict["Leaf"][iiout] = {}
                    galids = list( self.dict_leaves[iiout].keys() )
                    for galid in galids:
                        leaf = self.dict_leaves[iiout][galid]
                        name, status_leaf = leaf.export_backup()
                        backup_dict["Leaf"][iiout][name] = status_leaf
                clock_backup.done()
                # Save backup file
                pklsave(backup_dict, f"{fname}.pickle")             
                self.debugger.info(f"\n Elapsed time for jout={jout} ({len_branch} branches)\n---> {(time.time()-jref)/60:.4f} min")

        except Exception as e:
            print(traceback.format_exc())
            print(e)
            self.debugger.error(traceback.format_exc())
            self.debugger.error(e)
            self.debugger.error(self.summary())

        clock.done()

    def flush_auto(self, jout=None, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        reftot = MB()

        if jout is None:
            cout = -1 if self.prog else 10000
            keys = list( self.dict_snap.keys() )
            if len(keys)>0:
                cout = max(np.min(keys), cout) if self.prog else min(np.max(keys), cout)
            keys = list( self.dict_gals["galaxymakers"].keys() )
            if len(keys)>0:
                cout = max(np.min(keys), cout) if self.prog else min(np.max(keys), cout)
            keys = list( self.dict_part.keys() )
            if len(keys)>0:
                cout = max(np.min(keys), cout) if self.prog else min(np.max(keys), cout)
            cstep = out2step(cout, galaxy=self.galaxy, mode=self.simmode, nout=self.nout, nstep=self.nstep)
            jstep = cstep+5 if self.prog else cstep-5
            jout = step2out(jstep, galaxy=self.galaxy, mode=self.simmode, nout=self.nout, nstep=self.nstep)
        else:
            jstep = out2step(jout, galaxy=self.galaxy, mode=self.simmode, nout=self.nout, nstep=self.nstep)
        if self.prog:
            dprint_(f"[flush notice]", self.debugger)
            dprint_(f"iout > {jout} will be removed", self.debugger)
            dprint_(f"Thank you", self.debugger)
        else:
            dprint_(f"[flush notice]", self.debugger)
            dprint_(f"iout < {jout} will be removed", self.debugger)
            dprint_(f"Thank you", self.debugger)
        
        # Snapshot
        keys = list( self.dict_snap.keys() )
        temp = []
        if len(keys)>0:
            refmem = MB()
            for iout in keys:
                istep = out2step(iout, galaxy=self.galaxy, mode=self.simmode, nout=self.nout, nstep=self.nstep)
                if (istep > jstep and self.prog) or (istep < jstep and not self.prog):
                    self.dict_snap[iout].clear()
                    self.dict_snap[iout] = None
                    del self.dict_snap[iout]
                    temp.append(iout)
            gc.collect()
            if len(temp)>0:
                self.debugger.info(f"* [flush][snapshot] remove {len(temp)} iouts ({np.min(temp)}~{np.max(temp)}) ({refmem-MB():.2f} MB saved)")
        
        # GalaxyMaker
        dprint_(f"\n", self.debugger)
        keys = list( self.dict_gals["galaxymakers"].keys() )
        temp = []
        if len(keys)>0:
            refmem = MB()
            for iout in keys:
                istep = out2step(iout, galaxy=self.galaxy, mode=self.simmode, nout=self.nout, nstep=self.nstep)
                if (istep > jstep and self.prog) or (istep < jstep and not self.prog):
                    self.dict_gals['galaxymakers'][iout] = None
                    self.dict_gals['gmpids'][iout] = None
                    del self.dict_gals['galaxymakers'][iout]
                    del self.dict_gals['gmpids'][iout]
                    temp.append(iout)
            gc.collect()
            if len(temp)>0:
                self.debugger.info(f"* [flush][{self.Galstr}] remove {len(temp)} iouts ({np.min(temp)}~{np.max(temp)}) ({refmem-MB():.2f} MB saved)")
        
        # Star
        dprint_(f"\n", self.debugger)
        keys = list( self.dict_part.keys() )
        if len(keys)>0:
            for iout in keys:
                istep = out2step(iout, galaxy=self.galaxy, mode=self.simmode, nout=self.nout, nstep=self.nstep)
                if (istep > jstep and self.prog) or (istep < jstep and not self.prog):
                    jkeys = list(self.dict_part[iout].keys())
                    refmem = MB()
                    for galid in jkeys:
                        self.dict_part[iout][galid].table = []
                        self.dict_part[iout][galid].snap.clear()
                        self.dict_part[iout][galid] = None
                        del self.dict_part[iout][galid]
                    self.dict_part[iout] = None
                    del self.dict_part[iout]
                    gc.collect()
                    self.debugger.info(f"* [flush][{self.Partstr}] remove iout={iout} ({len(jkeys)} gals) ({refmem-MB():.2f} MB saved)")
        # Leaf
        dprint_(f"\n", self.debugger)
        keys = list( self.dict_leaves.keys() )
        if len(keys)>0:
            for iout in keys:
                istep = out2step(iout, galaxy=self.galaxy, mode=self.simmode, nout=self.nout, nstep=self.nstep)
                if (istep > jstep and self.prog) or (istep < jstep and not self.prog):
                    jkeys = list(self.dict_leaves[iout].keys())
                    temp = [0, 0, 0]
                    for galid in jkeys:
                        # self.dict_leaves[iout][galid].report(prefix=prefix+"_before")
                        if len(self.dict_leaves[iout][galid].parents)==0:
                            if self.dict_leaves[iout][galid].clear_ready:
                                refmem = MB()
                                if self.dict_leaves[iout][galid].branch is not None:
                                    self.dict_leaves[iout][galid].branch.disconnect(self.dict_leaves[iout][galid], prefix=prefix)
                                    self.dict_leaves[iout][galid].branch = None
                                self.dict_leaves[iout][galid].clear(msgfrom="flush_auto")
                                self.dict_leaves[iout][galid] = None
                                del self.dict_leaves[iout][galid]
                                temp[0] += 1
                            else:
                                self.dict_leaves[iout][galid].clear()
                                temp[1] += 1
                        else:
                            iname = self.dict_leaves[iout][galid].name()
                            switch = True
                            for parent in self.dict_leaves[iout][galid].parents:
                                if parent in self.branches_queue.keys():
                                    parent_branch = self.branches_queue[parent]
                                    if parent_branch.currentleaf.name() == iname:
                                        switch = False
                                        temp[2] += 1
                                        break
                            if switch:
                                self.dict_leaves[iout][galid].parents = []
                                temp[1] += 1
                        # if galid in self.dict_leaves[iout].keys():
                        #     self.dict_leaves[iout][galid].report(prefix=prefix+"_after")
                    gc.collect()
                    self.debugger.info(f"* [flush][Leaf] (at {iout}) {temp[0]} leaves removed, {temp[1]} leaves will be removed, {temp[2]} leaves have parents ({refmem-MB():.2f} MB saved)")
                    if len(self.dict_leaves[iout].keys())==0:
                        refmem = MB()
                        self.dict_leaves[iout] = None
                        del self.dict_leaves[iout]
                        gc.collect()
                        self.debugger.info(f"* [flush][Leaf] remove iout={iout} garbage ({refmem-MB():.2f} MB saved)")

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

        # clock.done()
        return self.dict_snap[iout]

    def load_gal(self, iout, galid, return_part=False, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        if psutil.Process().memory_info().rss / 2 ** 30 > self.flush_GB+self.iniGB:
            self.flush_auto(prefix=prefix)

        if return_part:
            if not iout in self.dict_gals["gmpids"].keys():
                func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
                snap = self.load_snap(iout, prefix=prefix)
                gm, temp = uhmi.HaloMaker.load(snap, galaxy=self.galaxy, load_parts=True, double_precision=self.dp) #<-- Bottleneck!
                gmpid = np.array(temp)
                self.dict_gals["galaxymakers"][iout] = gm
                cumparts = np.insert(np.cumsum(gm["nparts"]), 0, 0)
                gmpid = tuple(gmpid[ cumparts[i]:cumparts[i+1] ] for i in range(len(gm)))
                self.dict_gals["gmpids"][iout] = gmpid #<-- Bottleneck!
                del gm; del gmpid; del cumparts; del temp
                if self.loadall:
                    clock2 = timer(text=prefix+f"loadall ({iout}): <get_part>", verbose=self.verbose, debugger=self.debugger)
                    snap.box = np.array([[0, 1], [0, 1], [0, 1]])
                    snap.get_part(onlystar=self.galaxy, target_fields=['id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'm'])
                    snap.part_data['id'] = np.abs(snap.part_data['id'])
                    clock2.done()
                    clock2 = timer(text=prefix+f"loadall ({iout}): <argsort>", verbose=self.verbose, debugger=self.debugger)
                    arg = np.argsort(snap.part_data['id'])
                    snap.part_data = snap.part_data[arg]
                    snap.part.table = snap.part_data
                    self.dict_snap[iout] = snap
                    clock2.done()
                    clock2 = timer(text=prefix+f"loadall ({iout}): <part2dict>", verbose=self.verbose, debugger=self.debugger)
                    for gal in self.dict_gals["galaxymakers"][iout]:
                        self.load_part(iout, gal['id'], prefix=prefix, silent=False, galaxy=self.galaxy)
                    clock2.done()
        else:
            if not iout in self.dict_gals["galaxymakers"].keys():
                func = f"[{inspect.stack()[0][3]}_fast]"; prefix = f"{prefix}{func}"
                snap = self.load_snap(iout, prefix=prefix)
                gm = uhmi.HaloMaker.load(snap, galaxy=self.galaxy, load_parts=False, double_precision=self.dp) #<-- Bottleneck!
                self.dict_gals["galaxymakers"][iout] = gm
                del gm

        # clock.done()
        if isinstance(galid,str):
            if galid=='all':
                return self.dict_gals["galaxymakers"][iout]
        elif isinstance(galid, Iterable):
            a = np.hstack([self.dict_gals["galaxymakers"][iout][ia-1] for ia in galid])
            if return_part:
                # b = [self.dict_gals["gmpids"][iout][ia-1] for ia in galid]
                b = tuple(self.dict_gals["gmpids"][iout][ia-1] for ia in galid)
                return a, b
            return a
        else:
            if return_part:
                return self.dict_gals["galaxymakers"][iout][galid-1], self.dict_gals["gmpids"][iout][galid-1]
            return self.dict_gals["galaxymakers"][iout][galid-1]
    
    def load_leaf(self, iout, galid, branch, gal=None, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        if not iout in self.dict_leaves.keys():
            self.dict_leaves[iout] = {}
        

        if not galid in self.dict_leaves[iout].keys():
            # clock2 = timer(text=prefix+"[GalaxyMaker load]", verbose=self.verbose, debugger=self.debugger)
            if gal is None:
                gal = self.load_gal(iout, galid, prefix=prefix)
            self.dict_leaves[iout][galid] = Leaf(gal, branch, self, verbose=self.verbose-1, prefix=prefix, debugger=self.debugger, interplay=self.interplay, prog=self.prog)
        
        if self.dict_leaves[iout][galid].pruned:
            if gal is None:
                gal = self.load_gal(iout, galid, prefix=prefix)
            self.dict_leaves[iout][galid] = Leaf(gal, branch, self, verbose=self.verbose-1, prefix=prefix, debugger=self.debugger, interplay=self.interplay, prog=self.prog)

        if branch is not None:
            if not branch.rootid in self.dict_leaves[iout][galid].parents:
                self.dict_leaves[iout][galid].parents += [branch.rootid]
        
            if self.dict_leaves[iout][galid].branch != branch:
                if not self.dict_leaves[iout][galid].branch in self.dict_leaves[iout][galid].otherbranch:
                    self.dict_leaves[iout][galid].otherbranch += [self.dict_leaves[iout][galid].branch]
                self.dict_leaves[iout][galid].branch = branch
            self.dict_leaves[iout][galid].clear_ready=False
            branch.connect(self.dict_leaves[iout][galid], prefix=prefix)
        return self.dict_leaves[iout][galid]


    def load_part(self, iout, galid, prefix="", silent=False, galaxy=True):
        # BUG: For NH2, galaxymaker overfind member particles. So I should some cut radius for finding member stars
        # In this case, we may want to check other dependent functions for example match rate score.
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        if psutil.Process().memory_info().rss / 2 ** 30 > self.flush_GB+self.iniGB:
            self.flush_auto(prefix=prefix)

        snap = self.load_snap(iout, prefix=prefix)
        gal, gpid = self.load_gal(iout, galid, return_part=True, prefix=prefix)
        if not galid in self.dict_part[iout].keys():
            # if snap.part_data is not None:
            #     if snap.part_data.nbytes / 2**30 > self.flush_GB:
            #         self.debugger.info(prefix+f" snap.clear!! ({snap.part_data.nbytes / 2**30:.2f} GB)")
            #         snap.clear()
            if self.loadall:
                if snap.part_data is None:
                    snap.get_part(onlystar=self.galaxy, target_fields=['id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'm'])
                # clock2 = timer(text=prefix+"[get_part]00", verbose=self.verbose, debugger=self.debugger)
                # if len(snap.part_data['id']) != np.max(snap.part_data['id']):
                #     raise ValueError(f"ID:{np.min(snap.part_data['id'])}~{np.max(snap.part_data['id'])} vs len:{len(snap.part_data['id'])}")
                # clock2.done();clock2 = timer(text=prefix+"[get_part]01", verbose=self.verbose, debugger=self.debugger)
                part = snap.part_data[gpid-1]
                # clock2.done();clock2 = timer(text=prefix+"[get_part]02", verbose=self.verbose, debugger=self.debugger)
                part = snap.Particle(part, snap)
                # clock2.done();clock2 = timer(text=prefix+"[get_part]03", verbose=self.verbose, debugger=self.debugger)
                self.dict_part[iout][galid] = part
                # clock2.done()

            else:
                leng = 0
                scale = 1
                while leng < len(gpid):
                    snap.set_box_halo(gal, 1.1*scale, use_halo_radius=True, radius_name='r')
                    # if not silent:
                        # clock2 = timer(text=prefix+"[get_part]1", verbose=self.verbose, debugger=self.debugger)
                    snap.get_part(onlystar=self.galaxy, target_fields=['id', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'm'])
                    # if not silent:
                        # clock2.done();clock2 = timer(text=prefix+"[get_part]2", verbose=self.verbose, debugger=self.debugger)
                    try:
                        if galaxy:
                            part = snap.part.table
                        else:
                            part = snap.part['dm'].table
                        part['id'] = np.abs(part['id'])
                        # clock2.done();clock2 = timer(text=prefix+"[get_part]3", verbose=self.verbose, debugger=self.debugger)
                        # self.debugger.debug(f"[NUMBA TEST][load_part] -> [large_isin(a,b)]:")
                        # self.debugger.debug(f"            type(a)={type(part['id'])}, type(a[0])={type(part['id'][0])}")
                        # self.debugger.debug(f"            type(b)={type(gpid)}, type(b[0])={type(gpid[0])}")
                        part = part[large_isin(part['id'], gpid)]
                        # clock2.done();clock2 = timer(text=prefix+"[get_part]4", verbose=self.verbose, debugger=self.debugger)
                        leng = len(part['id'])
                    except: # didin't found part??
                        leng = 0
                    self.debugger.debug(f"[get_part] {scale}")
                    scale *= 2
                    if scale > 4:
                        self.debugger.info(prefix+f"scale exceeds 4xRadius! Cut-off nparts! {len(gpid)}->{leng}")
                        break
                part = snap.Particle(part, snap)
                # clock2.done();clock2 = timer(text=prefix+"[get_part]5", verbose=self.verbose, debugger=self.debugger)
                self.dict_part[iout][galid] = part
                # clock2.done();clock2 = timer(text=prefix+"[get_part]6", verbose=self.verbose, debugger=self.debugger)
        
        # clock.done()
        return self.dict_part[iout][galid]