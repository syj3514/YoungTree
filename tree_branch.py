import inspect
import gc
import copy

from tree_utool import *
from tree_leaf import Leaf

#########################################################
###############         Branch Class                #####
#########################################################
class Branch():
    def __init__(self, root, DataObj, galaxy=True, mode='hagn', verbose=2, prefix="", debugger=None, interplay=False,prog=True, **kwargs):
        func = f"[__Branch__]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=verbose, debugger=debugger)
        self.debugger=debugger
        self.verbose = verbose

        self.data = DataObj
        self.root = root
        self.galaxy = galaxy
        self.mode = mode
        self.rootout, self.rootstep = ioutistep(self.root, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout)
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
        elif mode == 'nh':
            self.repo = f"/storage6/NewHorizon/"
            self.rurmode = 'nh'
        else:
            raise ValueError(f"{mode} is not supported!")

        
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
    
    def __del__(self):
        self.debugger.info(f"[DEL] Branch (root={self.root['id']}) is destroyed")

    def clear(self, msgfrom='self'):
        self.debugger.info(f"[CLEAR] Branch (root={self.root['id']}) [from {msgfrom}]")
        self.inipid = None; self.inipwei=None
        self.data = None
        self.root = None
        self.candidates = {}
        self.rootleaf = None
        self.leaves = {}

    def selfsave(self):
        a = self.root['id']
        b = self.secrecord
        c = self.leaves
        d = self.leave_scores
        readme = "1) root galaxy, 2) total elapsed time, 3) tree branch results, 4) corresponding score based on matchrate(importance-weighted) & mass difference & velocity offset"
        fname = f"./result/{self.mode}/{self.progstr}_Branch_ID{a:05d}_iout{self.rootout:05d}.pickle"
        self.debugger.info(f"\n>>> GAL{a:05d} is saved as `{fname}`")
        self.debugger.info(f">>> Treeleng = {len(c.keys())} (elapsed {self.secrecord/60:.2f} min)\n")
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
        
        temp = []
        for iout in self.candidates.keys():
            temp1 = [f"{ikey}({self.scores[iout][ikey]:.3f}={self.score0s[iout][ikey]:.2f}+{self.score1s[iout][ikey]:.2f}+{self.score2s[iout][ikey]:.2f}+{self.score3s[iout][ikey]:.2f})" for ikey in self.candidates[iout].keys()]
            temp += f"[{iout}]\t{temp1}\n"
        tcands = "".join(temp)

        text = f"\n[Branch summary report]\n>>> Root:\n\t{troot}\n>>> Current root:\n\t{tcurrent}\n>>> Current branch:\n\t[{tbranch}]\n>>> Candidates:\n{tcands}\n>>> Current Memory: {psutil.Process().memory_info().rss / 2 ** 30:.4f} GB\n>>> Elapsed time: {self.secrecord:.4f} sec\n"
        if isprint:
            print(text)
        return text

    def execution(self, prefix="", treeleng=None, **kwargs):
        print(f"Log is written in {self.debugger.handlers[0].baseFilename}")
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

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
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        ref = time.time()
        self.go = self.rootleaf.find_candidates(prefix=prefix, **kwargs)
        # dprint(f"*** go? {self.go}", self.debugger)
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
        self.debugger.info(f"{prefix}\n{self.summary(isprint=False)}")
        # except Warning as e:
        #     print("WARNING!! in do_onestep")
        #     breakpoint()
        #     self.debugger.warning("########## WARNING #########")
        #     self.debugger.warning(e)
        #     self.debugger.warning(self.summary())
        #     raise ValueError("velocity wrong!")
        return self.go

        

    def reset_branch(self, gal, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        Newleaf = self.gal2leaf(gal, prefix=prefix)
        keys = list( self.leaves.keys() )
        for key in keys:
            if (key < Newleaf.iout and self.prog) or (key > Newleaf.iout and not self.prog):
                refmem = MB()
                self.leaves[key] = None
                del self.leaves[key]
                self.debugger.debug(f"* [Branch][Reset] remove {self.data.galstr} at iout={key} ({refmem-MB():.2f} MB saved)")
        
        keys = list( self.candidates.keys() )
        for key in keys:
            if (key < Newleaf.iout and self.prog) or (key > Newleaf.iout and not self.prog):
                ids = list( self.candidates[key].keys() )
                for iid in ids:
                    refmem = MB()
                    self.candidates[key][iid].parents.remove(self.root['id'])
                    self.debugger.debug(f"*** Branch({self.root['id']}) connection lost to Leaf({iid} at {key})")
                    self.candidates[key][iid].clear(msgfrom="reset_branch")
                    del self.candidates[key][iid]
                    self.debugger.debug(f"* [Branch][Reset] remove {iid} leaf at iout={key} ({refmem-MB():.2f} MB saved)")
                del self.candidates[key]
        gc.collect()

        clock.done()

    def gal2leaf(self, gal, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        iout, _ = ioutistep(gal, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
        if not iout in self.candidates.keys():
            self.debugger.debug(f"{prefix} *** no iout{iout} in candidates --> make it!")
            self.candidates[iout] = {}
            self.scores[iout] = {}
            self.score0s[iout] = {}
            self.score1s[iout] = {}
            self.score2s[iout] = {}
            self.score3s[iout] = {}

        if not gal['id'] in self.candidates[iout].keys():
            # func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
            # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
            
            self.debugger.debug(f"{prefix} *** no id{gal['id']} in candidates at iout{iout} --> make it!")
            self.candidates[iout][gal['id']] = self.data.load_leaf(iout, gal['id'], self, gal=gal, prefix=prefix)
            self.scores[iout][gal['id']] = 0
            self.score0s[iout][gal['id']] = 0
            self.score1s[iout][gal['id']] = 0
            self.score2s[iout][gal['id']] = 0
            self.score3s[iout][gal['id']] = 0

            # clock.done()
        if self.candidates[iout][gal['id']].pruned:
            self.debugger.debug(f"{prefix} *** id{gal['id']} at iout{iout} exists but pruned --> remake it!")
            self.data.load_leaf(iout, gal['id'], self, gal=gal, prefix=prefix)
        
        clock.done() # <---
        return self.candidates[iout][gal['id']]


    def update_cands(self, iout, galids,checkids=None, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        if checkids is not None:
            gals, gmpids = self.data.load_gal(iout, galids, return_part=True, prefix=prefix)
            inds = atleast_numba_para(gmpids, checkids)
            for gal, _, ind in zip(gals, gmpids, inds):
                # if atleast_numba(gmpid, checkids):
                if ind:
                    self.gal2leaf(gal, prefix=prefix)
                else:
                    self.debugger.info(f"{prefix} *** id{gal['id']} at iout{iout} has no common parts of {len(checkids)}!")
                    pass
        else:
            gals = self.data.load_gal(iout, galids, return_part=False, prefix=prefix)
            for gal in gals:
                self.gal2leaf(gal, prefix=prefix)

        clock.done()
        if iout in self.candidates.keys():
            return list(self.candidates[iout].keys())
        else:
            return []


    def gals_from_candidates(self, iout, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        cands = self.candidates[iout]
        gals = np.concatenate([cands[key].gal_gm for key in cands.keys()])

        clock.done()
        return gals
    
    
    def choose_winner(self, jout, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        if self.prog:
            iout = np.max(list(self.candidates.keys()))
            self.debugger.debug(f"** [choose_winner] winner at {jout} (maxout={iout})")
        else:
            iout = np.min(list(self.candidates.keys()))
            self.debugger.debug(f"** [choose_winner] winner at {jout} (minout={iout})")

        if iout != self.rootout: # Skip start iout
            if jout==iout:

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
                    self.debugger.debug(f"** [Choose_winner] winid={winid}, winscore={winscore}")
                    self.debugger.debug(f"** [Choose_winner] root leaf {self.rootleaf.gal_gm['id']} at {self.rootleaf.gal_gm['timestep']}")
                    self.rootleaf.parents.remove(self.root['id'])
                    self.rootleaf = self.candidates[iout][winid]
                    self.debugger.debug(f"** [Choose_winner] root leaf --> {self.rootleaf.gal_gm['id']} at {self.rootleaf.gal_gm['timestep']}")
                    self.go = self.rootleaf.find_candidates(prefix=prefix)
                    self.leaves[iout] = self.candidates[iout][winid].gal_gm
                    self.leave_scores[iout] = winscore
            
                ids = list( self.candidates[iout].keys() )
                for iid in ids:
                    refmem = MB()
                    if iid != winid:
                        self.candidates[iout][iid].parents.remove(self.root['id'])
                        # self.debugger.debug(f"*** Branch({self.root['id']}) connection lost to Leaf({iid} at {iout})")
                    del self.candidates[iout][iid]
                    del self.scores[iout][iid]
                    self.debugger.debug(f"** [Choose_winner] remove ID={iid} leaf&score at iout={iout} ({refmem-MB():.2f} MB saved)")
                del self.candidates[iout]
                del self.scores[iout]

                ikeys = list( self.candidates.keys() )
                for ikey in ikeys:
                    jkeys = list( self.candidates[ikey].keys() )
                    for jkey in jkeys:
                        if self.scores[ikey][jkey] <= -1:
                            self.debugger.debug(f"** [Choose_winner] iout={ikey}, ID={jkey}, score={self.scores[ikey][jkey]}")
                            refmem = MB()
                            self.candidates[ikey][jkey].parents.remove(self.root['id'])
                            # self.debugger.debug(f"*** Branch({self.root['id']}) connection lost to Leaf({iid} at {iout})")
                            del self.candidates[ikey][jkey]
                            del self.scores[ikey][jkey]
                            self.debugger.debug(f"** [Choose_winner] remove non-referred ID={jkey} leaf&score at iout={ikey} ({refmem-MB():.2f} MB saved)")
                    if len(self.candidates[ikey].keys())<1:
                        del self.candidates[ikey]
                        del self.scores[ikey]
        else:
            del self.candidates[iout]

        gc.collect()        
        clock.done()