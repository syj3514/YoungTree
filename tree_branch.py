import inspect
import gc
import copy

from tree_utool import *
from tree_leaf import Leaf

#########################################################
###############         Branch Class                #####
#########################################################
class Branch():
    __slots__ = [
        'verbose', 'data','root','rootid','galaxy','mode', 'rootout',
        'rootid', 'rootstep', 'key', 'repo', 'rurmode', 'interplay', 'candidates',
        'scores','score0s','score1s','score2s','score3s','rootleaf',
        'inipid','inipwei','leaves','leave_scores','secrecord','go','prog','progstr'
        ]
    def __init__(self, root, DataObj, galaxy=True, mode='hagn', verbose=2, prefix="", debugger=None, interplay=False,prog=True, **kwargs):
        func = f"[__Branch__]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=verbose, debugger=debugger)
        # self.data.debugger=debugger
        self.verbose = verbose

        self.data = DataObj
        self.root = root
        self.rootid = root['id']
        self.galaxy = galaxy
        self.mode = mode
        self.rootout, self.rootstep = ioutistep(self.root, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout)
        if self.galaxy:
            self.key='star'
        else:
            self.key='dm'

        self.repo, self.rurmode, _ = mode2repo(mode)
        # if mode[0]=='h':
        #     self.repo = f'/storage4/Horizon_AGN/'
        #     self.rurmode = "hagn"
        # elif mode[0]=='y':
        #     self.repo = f'/storage3/Clusters/{mode[1:]}'
        #     self.rurmode = "yzics"
        # elif mode == 'nh':
        #     self.repo = f"/storage6/NewHorizon/"
        #     self.rurmode = 'nh'
        # else:
        #     raise ValueError(f"{mode} is not supported!")

        
        self.interplay=interplay
        self.candidates = {} # dictionary of Leafs
        self.scores = {}
        self.score0s = {}
        self.score1s = {}
        self.score2s = {}
        self.score3s = {}
        self.rootleaf = self.gal2leaf(self.root, prefix=prefix)
        self.inipid = copy.deepcopy(self.rootleaf.pid)
        self.inipwei = copy.deepcopy(self.rootleaf.pweight)
        self.leaves = {self.rootout: self.root} # results
        self.leave_scores = {self.rootout: 1} # results
        self.secrecord = 0
        self.go = True
        self.prog = prog
        self.progstr = "Descendant"
        if self.prog:
            self.progstr = "Progenitor"


        clock.done()
    
    def name(self):
        return (self.rootout, self.rootid)

    def export_backup(self, prefix=""):
        # func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func} <B{self.rootid} at {self.rootout}>"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)
        # What only I have
        #   Unchanged
        #       rootid, rootout, rootstep, inipid, inipwei, rootleaf
        #   Changed
        #       candidates, scores, score0s, score1s, score2s, score3s, leaves, leave_scores, secrecord, go
        #   From data
        #       verbose, data, galaxy, mode, key, repo, rurmode, interplay, prog, progstr

        # root.make_branches  <- iout, galid, interplay   [needs: rootout, rootid, (interplay)]
        # and then, changed:
        name = self.name()
        status = {
            "rootleaf":self.rootleaf.name(),
            "candidates":tuple(self.candidates[iout][galid].name() for iout in self.candidates.keys() for galid in self.candidates[iout]),
            "scores":tuple((iout, galid, self.scores[iout][galid]) for iout in self.scores.keys() for galid in self.scores[iout]),
            "score0s":tuple((iout, galid, self.score0s[iout][galid]) for iout in self.score0s.keys() for galid in self.score0s[iout]),
            "score1s":tuple((iout, galid, self.score1s[iout][galid]) for iout in self.score1s.keys() for galid in self.score1s[iout]),
            "score2s":tuple((iout, galid, self.score2s[iout][galid]) for iout in self.score2s.keys() for galid in self.score2s[iout]),
            "score3s":tuple((iout, galid, self.score3s[iout][galid]) for iout in self.score3s.keys() for galid in self.score3s[iout]),
            "leaves_and_scores":tuple((iout, self.leaves[iout]['id'], self.leave_scores[iout]) for iout in self.leaves.keys()),
            "secrecord":self.secrecord,
            "go":self.go
            }
        
        # clock.done()
        return name, status
    
    def import_backup(self, status, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func} <B{self.rootid} at {self.rootout}>"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        self.rootleaf = self.data.load_leaf(*status["rootleaf"], self, prefix=prefix)
        # dprint_(np.array(status["candidates"]), debugger=self.data.debugger)
        cands = np.array(status["candidates"])
        iouts = np.unique(cands[:,0])
        for iout in iouts:
            galids = cands[:,1][cands[:,0]==iout]
            self.update_cands(iout, np.atleast_1d(galids),checkids=None, prefix=prefix)
        
        iouts = list(self.candidates.keys())
        if self.rootout in iouts:
            ids = list(self.candidates[self.rootout].keys())
            if len(ids)>0:
                for iid in ids:
                    self.disconnect(self.candidates[self.rootout][iid], prefix=prefix)
                    del self.candidates[self.rootout][iid]
            del self.candidates[self.rootout]
            
        
        zips = zip(status["scores"], status["score0s"], status["score1s"], status["score2s"], status["score3s"])
        for scores, score0s, score1s, score2s, score3s in zips:
            iout, galid, score = scores
            if not iout in self.scores.keys():
                self.scores[iout] = {}
                self.score0s[iout] = {}
                self.score1s[iout] = {}
                self.score2s[iout] = {}
                self.score3s[iout] = {}
            _, _, score0 = score0s
            _, _, score1 = score1s
            _, _, score2 = score2s
            _, _, score3 = score3s
            self.scores[iout][galid] = score
            self.score0s[iout][galid] = score0
            self.score1s[iout][galid] = score1
            self.score2s[iout][galid] = score2
            self.score3s[iout][galid] = score3
        
        switch = False
        if self.data.loadall:
            switch = True
            self.data.loadall = False                

        for iout, galid, score in status["leaves_and_scores"]:
            self.leaves[iout] = self.data.load_gal(iout, galid)
            self.leave_scores[iout] = score
        
        if switch:
            self.data.loadall = True
            
        
        self.secrecord = status["secrecord"]
        self.go = status["go"]

        clock.done()


    def __del__(self):
        self.data.debugger.info(f"[DEL] Branch (root={self.rootid}) is destroyed")

    def clear(self, msgfrom='self'):
        self.data.debugger.info(f"[CLEAR] Branch (root={self.rootid}) [from {msgfrom}]")
        ikeys = list( self.candidates.keys() )
        for ikey in ikeys:
            jkeys = list( self.candidates[ikey].keys() )
            for jkey in jkeys:
                if self.scores[ikey][jkey] <= -1:
                    self.disconnect(self.candidates[ikey][jkey], prefix="[Branch Clear]")
                    del self.candidates[ikey][jkey]
                    del self.scores[ikey][jkey]
            if len(self.candidates[ikey].keys())<1:
                del self.candidates[ikey]
                del self.scores[ikey]
        self.inipid = None; self.inipwei=None
        self.root = None
        self.candidates = {}
        self.disconnect(self.rootleaf, prefix="[Branch Clear]")
        # self.data = None
        self.rootleaf = None
        self.leaves = {}

    def selfsave(self):
        a = self.rootid
        b = self.secrecord
        c = self.leaves
        d = self.leave_scores
        readme = "1) root galaxy, 2) total elapsed time, 3) tree branch results, 4) corresponding score based on matchrate(importance-weighted) & mass difference & velocity offset"
        fname = f"./result/{self.mode}/{self.progstr}_Branch_ID{a:05d}_iout{self.rootout:05d}.pickle"
        self.data.debugger.info(f"\n>>> GAL{a:05d} is saved as `{fname}`")
        self.data.debugger.info(f">>> Treeleng = {len(c.keys())} (elapsed {self.secrecord/60:.2f} min)\n")
        print(f"\n>>> GAL{a:05d} is saved as `{fname}`")
        print(f">>> Treeleng = {len(c.keys())} (elapsed {self.secrecord/60:.2f} min)\n")
        pklsave((readme, self.root,b,c,d), fname, overwrite=True)
        self.clear(msgfrom="selfsave")

    def summary(self, isprint=False):
        import textwrap
        troot = printgal(self.root, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep, isprint=isprint)
        tcurrent = printgal(self.rootleaf.gal_gm, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep, isprint=isprint)
        
        temp = [f"At {a}: {b['id']}" for a,b in self.leaves.items()]
        temp = " | ".join(temp)
        tbranch = "\n\t".join(textwrap.wrap(temp, 50))
        
        
        if len(self.candidates.keys())>0:
            temp = []
            outkeys = list(self.candidates.keys())
            outkeys = np.sort(outkeys)[::-1] if self.prog else np.sort(outkeys)
            for iout in outkeys:
                temp1 = [f"{ikey}({self.scores[iout][ikey]:.3f}={self.score0s[iout][ikey]:.2f}+{self.score1s[iout][ikey]:.2f}+{self.score2s[iout][ikey]:.2f}+{self.score3s[iout][ikey]:.2f})" for ikey in self.candidates[iout].keys()]
                temp += f"[{iout}]\t{temp1}\n"
            tcands = "".join(temp)
        else:
            tcands = ""

        text = f"\n[Branch summary report] (keep going? {'Yes' if self.go else 'No'})\n>>> Root:\n\t{troot}\n>>> Current root:\n\t{tcurrent}\n>>> Current branch:\n\t[{tbranch}]\n>>> Candidates:\n{tcands}\n>>> Current Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB\n>>> Elapsed time: {self.secrecord:.4f} sec\n"
        if isprint:
            print(text)
        return text

    def execution(self, prefix="", treeleng=None, **kwargs):
        print(f"Log is written in {self.data.debugger.handlers[0].baseFilename}")
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        self.go = True
        for jout in self.data.nout:
        # while self.go:
            self.go = self.do_onestep(jout, prefix=prefix, **kwargs)
            if treeleng is not None:
                if len(self.leaves.keys()) >= treeleng:
                    self.go=False
            if not self.go:
                break
        clock.done()


    def do_onestep(self, jout, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        self.data.debugger.info(f"\n{self.summary(isprint=False)}")
        ref = time.time()
        self.connect(self.rootleaf, prefix=prefix)
        self.go = self.rootleaf.find_candidates(prefix=prefix, **kwargs)
        # dprint(f"*** go? {self.go}", self.data.debugger)
        if len(self.candidates.keys())>0:
            if jout != self.rootout:
                self.rootleaf.calc_score(prefix=prefix)
            self.choose_winner(jout, prefix=prefix)
        else:
            self.go = False
        
        if not self.go:
            keys = list(self.candidates.keys())
            for key in keys:
            # while len(self.candidates.keys())>0:
                self.choose_winner(key, prefix=prefix)
        
        self.secrecord += time.time()-ref
        clock.done()
        self.data.debugger.info(f"\n{self.summary(isprint=False)}\n\n\n")
        # except Warning as e:
        #     print("WARNING!! in do_onestep")
        #     breakpoint()
        #     self.data.debugger.warning("########## WARNING #########")
        #     self.data.debugger.warning(e)
        #     self.data.debugger.warning(self.summary())
        #     raise ValueError("velocity wrong!")
        return self.go

        

    def reset_branch(self, gal, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        Newleaf = self.gal2leaf(gal, prefix=prefix)
        keys = list( self.leaves.keys() )
        for key in keys:
            if (key < Newleaf.iout and self.prog) or (key > Newleaf.iout and not self.prog):
                refmem = MB()
                self.leaves[key] = None
                del self.leaves[key]
                self.data.debugger.debug(f"* [Branch][Reset] remove {self.data.galstr} at iout={key} ({refmem-MB():.2f} MB saved)")
        
        keys = list( self.candidates.keys() )
        for key in keys:
            if (key < Newleaf.iout and self.prog) or (key > Newleaf.iout and not self.prog):
                ids = list( self.candidates[key].keys() )
                for iid in ids:
                    refmem = MB()
                    self.disconnect(self.candidates[key][iid], prefix=prefix)
                    del self.candidates[key][iid]
                    self.data.debugger.debug(f"* [Branch][Reset] remove (L{iid} at {key}) ({refmem-MB():.2f} MB saved)")
                del self.candidates[key]
        gc.collect()

        clock.done()

    def gal2leaf(self, gal, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        iout, _ = ioutistep(gal, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
        if not iout in self.candidates.keys():
            self.data.debugger.debug(f"{prefix} *** no iout{iout} in candidates --> make it!")
            self.candidates[iout] = {}
            self.scores[iout] = {}
            self.score0s[iout] = {}
            self.score1s[iout] = {}
            self.score2s[iout] = {}
            self.score3s[iout] = {}

        if not gal['id'] in self.candidates[iout].keys():
            # func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
            # clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)
            
            self.data.debugger.debug(f"{prefix} *** no (L{gal['id']} at {iout}) in candidates --> make it!")
            self.candidates[iout][gal['id']] = self.data.load_leaf(iout, gal['id'], self, gal=gal, prefix=prefix)
            self.scores[iout][gal['id']] = 0
            self.score0s[iout][gal['id']] = 0
            self.score1s[iout][gal['id']] = 0
            self.score2s[iout][gal['id']] = 0
            self.score3s[iout][gal['id']] = 0

            # clock.done()
        if self.candidates[iout][gal['id']].pruned:
            self.data.debugger.debug(f"{prefix} *** (L{gal['id']} at {iout}) exists but pruned --> remake it!")
            self.data.load_leaf(iout, gal['id'], self, gal=gal, prefix=prefix)
        
        else:
            self.candidates[iout][gal['id']].report(prefix=prefix)

        clock.done() # <---
        return self.candidates[iout][gal['id']]


    def update_cands(self, iout, galids,checkids=None, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        if checkids is not None:
            gals, gmpids = self.data.load_gal(iout, galids, return_part=True, prefix=prefix)
            if iout in self.candidates.keys():
                ind = np.zeros(len(gals), dtype=bool)
                for i, gal in enumerate(gals):
                    if gal['id'] in self.candidates[iout].keys():
                        self.data.debugger.info(f"{prefix} *** (L{gal['id']} at {iout}) is already in candidates!")
                    else:
                        ind[i] = True
                gals, gmpids = gals[ind], tuple(ib for ib, ibool in zip(gmpids, ind) if ibool)
            if len(gals)==0:
                if iout in self.candidates.keys():
                    return list(self.candidates[iout].keys())
                else:
                    return []
            if len(gmpids) < 95:
                # self.data.debugger.debug(f"[NUMBA TEST][update_cands] -> [atleast_numba_para(a,b)]:")
                # self.data.debugger.debug(f"            type(a)={type(gmpids)}, type(a[0])={type(gmpids[0])}")
                # self.data.debugger.debug(f"            type(b)={type(checkids)}, type(b[0])={type(checkids[0])}")
                inds = atleast_numba_para(gmpids, checkids)
            else:
                nth = len(gmpids)//90 + 1
                for inth in range(nth):
                    igmpids = gmpids[inth*90 : (inth+1)*90]
                    # self.data.debugger.debug(f"[NUMBA TEST][update_cands] -> [atleast_numba_para(a,b)]: (split)")
                    # self.data.debugger.debug(f"            type(a)={type(igmpids)}, type(a[0])={type(igmpids[0])}")
                    # self.data.debugger.debug(f"            type(b)={type(checkids)}, type(b[0])={type(checkids[0])}")
                    iinds = atleast_numba_para(igmpids, checkids)
                    if inth == 0:
                        inds = iinds
                    else:
                        inds = np.hstack((inds, iinds))
            for gal, _, ind in zip(gals, gmpids, inds):
                if ind:
                    self.gal2leaf(gal, prefix=prefix)
                else:
                    self.data.debugger.info(f"{prefix} *** (L{gal['id']} at {iout}) has no common parts of {len(checkids)}!")
                    pass
        else:
            gals = self.data.load_gal(iout, galids, return_part=False, prefix=prefix)
            if iout in self.candidates.keys():
                ind = np.zeros(len(gals), dtype=bool)
                for i, gal in enumerate(gals):
                    if gal['id'] in self.candidates[iout].keys():
                        ind[i] = True
                    else:
                        self.data.debugger.info(f"{prefix} *** (L{gal['id']} at {iout}) is already in candidates!")
                gals = gals[ind]
            if len(gals)==0:
                if iout in self.candidates.keys():
                    return list(self.candidates[iout].keys())
                else:
                    return []
            for gal in gals:
                self.gal2leaf(gal, prefix=prefix)

        clock.done()
        if iout in self.candidates.keys():
            return list(self.candidates[iout].keys())
        else:
            return []
    
    def disconnect(self, leaf, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func} (B{self.rootid} <-> L{leaf.galid} at {leaf.iout})"
        dprint_(prefix, self.data.debugger)
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        # leaf.report(prefix=prefix)
        if leaf.branch == self:
            leaf.branch = None
        if self in leaf.otherbranch:
            leaf.otherbranch.remove(self)
        if self.rootid in leaf.parents:
            leaf.parents.remove(self.rootid)
        if len(leaf.parents)>0 and len(leaf.otherbranch)>0:
            if hasattr(leaf.otherbranch[0], "rootid"):
                leaf.branch = leaf.otherbranch[0]
                self.data.debugger.debug(f"{prefix} (L{leaf.galid} at {leaf.iout}) changes its parent {self.rootid} -> {leaf.otherbranch[0].rootid}")
        leaf.clear(msgfrom="disconnect of branch")
        leaf.report(prefix=prefix)

        # clock.done()

    def connect(self, leaf, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func} (B{self.rootid} <-> L{leaf.galid} at {leaf.iout})"
        dprint_(prefix, self.data.debugger)
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        leaf.clear_ready = False
        # isreport = True
        if leaf.pruned:
            # isreport = False
            self.data.load_leaf(leaf.iout, leaf.galid, self, gal=None, prefix=prefix)
        if leaf.branch != self:
            if not leaf.branch in leaf.otherbranch:
                if hasattr(leaf.branch, "rootid"):
                    leaf.otherbranch.append(leaf.branch)
                    self.data.debugger.debug(f"{prefix} (L{leaf.galid} at {leaf.iout}) changes its parent {leaf.branch.rootid} -> {self.rootid}")
            
            if self in leaf.otherbranch:
                leaf.otherbranch.remove(self)
            leaf.branch = self
        else:
            if not self.rootid in leaf.parents:
                leaf.parents.append(self.rootid)
        # if isreport:
        #     leaf.report(prefix=prefix)

        # clock.done()


    def gals_from_candidates(self, iout, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        cands = self.candidates[iout]
        gals = np.concatenate([cands[key].gal_gm for key in cands.keys()])

        clock.done()
        return gals
    
    
    def choose_winner(self, jout, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        if self.prog:
            iout = np.max(list(self.candidates.keys()))
            dprint_(f"** [Choose_winner] winner at {jout} (max_of_cand_out={iout})", self.data.debugger)
        else:
            iout = np.min(list(self.candidates.keys()))
            dprint_(f"** [Choose_winner] winner at {jout} (min_of_cand_out={iout})", self.data.debugger)

        if iout != self.rootout: # Skip start iout
            # Remove weird iout garbage
            ikeys = list( self.candidates.keys() )
            for ikey in ikeys:
                if ikey > jout:
                    jkeys = list( self.candidates[ikey].keys() )
                    for jkey in jkeys: # for galid
                        self.disconnect(self.candidates[ikey][jkey], prefix=prefix)
                        del self.scores[ikey][jkey]
                        del self.candidates[ikey][jkey]
                    del self.candidates[ikey]
                    del self.scores[ikey]
            # Re-assign iout
            if self.prog:
                iout = np.max(list(self.candidates.keys()))
                dprint_(f"** [Choose_winner] winner at {jout} (max_of_cand_out={iout})", self.data.debugger)
            else:
                iout = np.min(list(self.candidates.keys()))
                dprint_(f"** [Choose_winner] winner at {jout} (min_of_cand_out={iout})", self.data.debugger)
            # Main process
            if jout==iout:
                dprint_(f"** [Choose_winner] **********************************", self.data.debugger)

                # Select winner
                winid=0; winscore=-1
                for iid, iscore in self.scores[iout].items():
                    if iscore > winscore:
                        winid = iid
                        winscore = iscore
                dprint_(f"** (winner in {iout}) go? {self.go}", self.data.debugger)
                dprint_(f"** [Choose_winner] winid={winid}, winscore={winscore:.4f}", self.data.debugger)
                dprint_(f"** [Choose_winner] root leaf {self.rootleaf.gal_gm['id']} at {self.rootleaf.gal_gm['timestep']}", self.data.debugger)
                
                # If normal case, crown the winner at iout as king
                if winscore > -1:
                    self.disconnect(self.rootleaf, prefix='')
                    self.rootleaf = self.candidates[iout][winid]
                    self.leaves[iout] = self.rootleaf.gal_gm
                    self.leave_scores[iout] = winscore
                dprint_(f"** [Choose_winner] root leaf --> {self.rootleaf.gal_gm['id']} at {self.rootleaf.gal_gm['timestep']}", self.data.debugger)
                self.go = self.rootleaf.find_candidates(prefix=prefix)
                
                # Remove other candidates at iout
                ids = list( self.candidates[iout].keys() )
                for iid in ids:
                    refmem = MB()
                    if iid != winid:
                        self.disconnect(self.candidates[iout][iid], prefix=prefix)
                    del self.candidates[iout][iid]
                    del self.scores[iout][iid]
                    dprint_(f"** [Choose_winner] remove (L{iid} at {iout}) ({refmem-MB():.2f} MB saved)", self.data.debugger)
                del self.candidates[iout]
                del self.scores[iout]

                # Prune leaves
                ikeys = list( self.candidates.keys() )
                for ikey in ikeys: # for iout
                    jkeys = list( self.candidates[ikey].keys() )
                    for jkey in jkeys: # for galid
                        if self.scores[ikey][jkey] <= -1:
                            dprint_(f"** [Choose_winner] (L{jkey} at {ikey}), score={self.scores[ikey][jkey]}", self.data.debugger)
                            refmem = MB()
                            self.disconnect(self.candidates[ikey][jkey], prefix=prefix)
                            del self.candidates[ikey][jkey]
                            del self.scores[ikey][jkey]
                            dprint_(f"** [Choose_winner] remove negative (L{jkey} at {ikey}) ({refmem-MB():.2f} MB saved)", self.data.debugger)
                    if len(self.candidates[ikey].keys())<1:
                        del self.candidates[ikey]
                        del self.scores[ikey]
                dprint_(f"** [Choose_winner] **********************************", self.data.debugger)
            else:
                dprint_(f"** [Choose_winner] we want to choose winner at {jout}, but maxiout={iout} so skip!", self.data.debugger)
        else:
            del self.candidates[iout]

        gc.collect()        
        clock.done()