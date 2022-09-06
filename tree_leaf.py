import inspect
from numpy.lib.recfunctions import append_fields, drop_fields
from rur import uri

from tree_utool import *

#########################################################
###############         Leaf Class                #######
#########################################################
class Leaf():
    __slots__ = ['debugger', 'verbose', 'branch','otherbranch', 'parents', 'data', 
                'mode', 'galaxy', 'gal_gm', 'galid','iout', 'istep', 'clear_ready',
                'nextids','nextnids', 'pruned','interplay',
                'nparts','pid','pm','pweight',
                'px','py','pz','pvx','pvy','pvz', 'prog', 'saved_matchrates', 'saved_veloffsets']
    def __init__(self, gal, BranchObj, DataObj, verbose=1, prefix="", debugger=None, interplay=False, prog=True, **kwargs):
        func = f"[__Leaf__]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=verbose, debugger=debugger)
        # self.data.debugger=debugger
        self.verbose = verbose

        self.branch = BranchObj
        self.otherbranch = []
        self.parents = [self.branch.rootid]
        self.data = DataObj
        self.mode, self.galaxy = self.branch.mode, self.branch.galaxy
        self.gal_gm = gal
        self.clear_ready = False
        self.galid, self.iout = self.gal_gm['id'], self.gal_gm['timestep']
        self.istep = out2step(self.iout, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
        
        self.nextids = {}
        self.nextnids = {}
        self.pruned = False
        self.interplay = interplay
        prefix += f"<L{self.galid} at {self.iout}>"

        # for fast I/O
        self.nparts = 0
        self.pid = None
        self.px = None; self.py = None; self.pz = None
        self.pvx = None; self.pvy = None; self.pvz = None
        self.pm = None; self.pweight = None
        self.prog = prog
        self.saved_matchrates = {}
        self.saved_veloffsets = {}

        self.load_parts(prefix=prefix)
        self.importance(prefix=prefix, usevel=True)

        # clock.done()
    
    def __del__(self):
        self.data.debugger.info(f"[DEL] (L{self.galid} at {self.iout}) is destroyed")

    def __str__(self):
        if self.branch is None:
            return f"<Leaf object> {self.data.galstr} (L{self.galid} at {self.iout})\n\troot parents: {self.parents}\n\tPruned!"
        text = f"<Leaf object> {self.data.galstr} (L{self.galid} at {self.iout})\n\troot parents: {self.parents}\n\tcurrent branch: {self.branch.rootid}\n\t{self.nparts} {self.data.partstr}s"
        return text

    def report(self, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"

        txt = f"{prefix} [L{self.galid} at {self.iout}]\t{self.nparts} particles (pruned? {self.pruned})"
        self.data.debugger.debug(txt)
        txt = f"{prefix} Current branch: {self.branch.rootid if self.branch is not None else None}"
        self.data.debugger.debug(txt)
        txt = f"{prefix} Other branches: {[ib.rootid if ib is not None else None for ib in self.otherbranch]}"
        self.data.debugger.debug(txt)
        txt = f"{prefix} All parents: {self.parents}"
        self.data.debugger.debug(txt)


    def clear(self, msgfrom='self'):
        if len(self.parents)==0:
            if self.clear_ready:
                self.data.debugger.info(f"[CLEAR] Leaf (L{self.galid} at {self.iout}) [from {msgfrom}]")
                self.pid=None; self.pm=None; self.pweight=None
                self.px=None; self.py=None; self.py=None
                self.pvx=None; self.pvy=None; self.pvy=None
                self.nparts=0
                self.gal_gm = None
                if self.branch is not None:
                    self.branch.disconnect(self, prefix="[Leaf Clear]")
                self.branch = None
                if len(self.otherbranch) > 0:
                    for ibranch in self.otherbranch:
                        if ibranch is not None:
                            ibranch.disconnect(self, prefix="[Leaf Clear]")
                self.otherbranch = []
                self.pruned = True
            else:
                self.data.debugger.info(f"[CLEAR_Ready] Leaf (L{self.galid} at {self.iout}) [from {msgfrom}]")
                self.clear_ready = True

    def load_parts(self, prefix=""):
        # Subject to `__init__`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        temp = self.data.load_part(self.iout, self.galid, prefix=prefix, galaxy=self.galaxy)
        self.pid = temp['id']
        self.nparts = len(self.pid)
        self.px = temp['x']; self.py = temp['y']; self.pz = temp['z']
        self.pvx = temp['vx', 'km/s']; self.pvy = temp['vy', 'km/s']; self.pvz = temp['vz', 'km/s']
        self.pm = temp['m']

        self.data.debugger.debug(prefix+f" [ID{self.galid} iout{self.iout}] Nparts={self.gal_gm['nparts']}({self.nparts})")

        clock.done()
    

    def importance(self, prefix="", usevel=True):
        # Subject to `__init__`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)
        
        cx, cy, cz = self.gal_gm['x'],self.gal_gm['y'],self.gal_gm['z']
        dist = distance3d(cx,cy,cz, self.px, self.py, self.pz) / self.gal_gm['rvir']
        if usevel:
            # try:
            cvx, cvy, cvz = self.gal_gm['vx'],self.gal_gm['vy'],self.gal_gm['vz']
            vels = distance3d(cvx,cvy,cvz, self.pvx, self.pvy, self.pvz)
            vels /= np.std(vels)
            dist = np.sqrt( dist**2 + vels**2 )
            # except Warning as e:
            #     print("WARNING!! in importance")
            #     self.data.debugger.warning("########## WARNING #########")
            #     self.data.debugger.warning(e)
            #     self.data.debugger.warning(f"gal velocity {cvx}, {cvy}, {cvz}")
            #     self.data.debugger.warning(f"len parts {self.nparts}")
            #     self.data.debugger.warning(f"vels first tens {vels[:10]}")
            #     self.data.debugger.warning(f"vels std {np.std(vels)}")
            #     self.data.debugger.warning(self.summary())
            #     breakpoint()
            #     raise ValueError("velocity wrong!")
        self.pweight = self.pm/dist

        clock.done()
    

    def load_nextids(self, igals, njump=0, masscut_percent=1, nnext=5, prefix="", **kwargs): # MAIN BOTTLENECK!!
        # Subject to `find_candidates`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        if len(np.unique(igals['timestep']))>1:
            raise ValueError(f"`load_nextids` gots multi-out gals!")
        iout, istep = ioutistep(igals[0], galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
        isnap = self.data.load_snap(iout, prefix=prefix)
        jstep = istep-1-njump if self.prog else istep+1+njump
        jout = step2out(jstep, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
        jsnap = self.data.load_snap(jout, prefix=prefix)
        jgals = self.data.load_gal(jout, 'all', return_part=False, prefix=prefix) # BOTTLENECK!!
        dt = np.abs( isnap.params['age'] - jsnap.params['age'] )
        dt *= 1e9 * 365 * 24 * 60 * 60 # Gyr to sec
        nexts = np.array([0,0,0])
        
        for igal in igals: # tester galaxies at iout
            calc = True
            if iout in self.data.dict_leaves.keys():
                if igal['id'] in self.data.dict_leaves[iout].keys():
                    ileaf = self.data.dict_leaves[iout][igal['id']]
                    if jout in ileaf.nextids.keys():
                        nexts = np.concatenate((nexts, ileaf.nextids[jout]))
                        self.data.debugger.debug(f"{prefix} igal[{igal['id']} at {iout}] already calculated fats at {jout}!")
                        calc = False
            if calc:
                ivel = rms(igal['vx'], igal['vy'], igal['vz'])
                radii = 5*max(igal['r'],1e-4) + 5*dt*ivel*jsnap.unit['km']
                neighbors = cut_sphere(jgals, igal['x'], igal['y'], igal['z'], radii, both_sphere=True)
                self.data.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} in radii")
                
                
                if len(neighbors)>0: # candidates at jout
                    neighbors = neighbors[(neighbors['m'] >= igal['m']*masscut_percent/100) & (~np.isin(neighbors['id'], nexts))]
                    self.data.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} after masscut {masscut_percent} percent")
                    if (len(neighbors)>0) and ((len(neighbors)>nnext) or (len(igals)>2*nnext)):
                        _, checkpid = self.data.load_gal(iout, igal['id'], return_part=True, prefix=prefix)

                        rate = np.zeros(len(neighbors))
                        gals, gmpids = self.data.load_gal(jout, neighbors['id'], return_part=True, prefix=prefix)
                        # iout, istep = ioutistep(igal, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
                        ith = 0
                        for _, gmpid in zip(gals, gmpids):
                            try:
                                if atleast_numba(checkpid, gmpid):
                                    ind = large_isin(checkpid, gmpid)
                                    rate[ith] = howmany(ind, True)/len(checkpid)
                            except Warning:
                                if atleast_numba(checkpid, gmpid):
                                    ind = large_isin(checkpid, gmpid)
                                    rate[ith] = howmany(ind, True)/len(checkpid)
                                # self.data.debugger.debug(f"[NUMBA TEST][load_nextids] -> [atleast_numba(a,b)]:")
                                # self.data.debugger.debug(f"            type(a)={type(checkpid)}, type(a[0])={type(checkpid[0])}")
                                # self.data.debugger.debug(f"            type(b)={type(gmpid)}, type(b[0])={type(gmpid[0])}")
                            ith += 1
                        ind = rate>0

                        neighbors, rate = neighbors[ind], rate[ind]
                        self.data.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} after crossmatch")
                        if len(neighbors) > 0:
                            if len(neighbors) > nnext:
                                arg = np.argsort(rate)
                                neighbors = neighbors[arg][-nnext:]
                                self.data.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} after score sorting")
                            nid = neighbors['id']
                            nexts = np.concatenate((nexts, nid))
                        else:
                            nid = np.array([0])

                    elif len(neighbors) > 0:
                        nid = neighbors['id']
                        nexts = np.concatenate((nexts, nid))
                    else:
                        nid = np.array([0])
                else:
                    nid = np.array([0])
                
                if iout in self.data.dict_leaves.keys():
                    if igal['id'] in self.data.dict_leaves[iout].keys():
                        ileaf = self.data.dict_leaves[iout][igal['id']]
                        ileaf.nextids[jout] = nid
            
        nexts = np.concatenate((nexts, np.array([0,0,0])))
        
        clock.done()
        return np.unique( nexts[nexts>0] ), jout
    

    def find_candidates(self, masscut_percent=1, nstep=5, nnext=5, prefix="", **kwargs):
        ############################################################
        ########    ADD case when no nexther, jump beyond     ####### Maybe done?
        ############################################################
        # if not self.nextnids:
        keys = list(self.nextnids.keys())
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        if self.branch is None:
            self.data.debugger.info(f"??? why (L{self.galid} at {self.iout}) lost its branch?")
            self.report(prefix=prefix)
            if len(self.otherbranch)>0:
                self.otherbranch[0].connect(self, prefix=prefix)

        igals = np.atleast_1d(self.gal_gm)
        njump=0
        for i in range(nstep):
            jstep = self.istep-1-i if self.prog else self.istep+1+i
            if jstep > 0 and jstep <= np.max(self.data.nstep):
                jout = step2out(jstep, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
                if jout in keys:
                    self.data.debugger.info(f"{prefix} *** jout(jstep)={jout}({jstep}) is already calculated!")
                    nextnids = self.nextnids[jout]
                else:
                    nextnids, jout = self.load_nextids(igals, njump=njump, masscut_percent=masscut_percent, nnext=nnext, prefix=prefix, **kwargs)
                    self.nextnids[jout] = nextnids
                self.data.debugger.info(f"*** jout(jstep)={jout}({jstep}), nextns={nextnids}")
                if len(nextnids) == 0:
                    self.data.debugger.info("JUMP!!")
                    njump += 1
                else:
                    nextnids = self.branch.update_cands(jout, nextnids, checkids=self.pid, prefix=prefix) # -> update self.branch.candidates & self.branch.scores
                    if len(nextnids) == 0:
                        self.data.debugger.info("JUMP!!")
                        njump += 1
                    else:
                        njump = 0
                        igals = self.data.load_gal(jout, nextnids, return_part=False, prefix=prefix)

        clock.done()
        keys = list(self.nextnids.keys())
        if len(keys)==0:
            return False
        temp = np.sum([len(self.nextnids[key]) for key in keys])
        return temp > 0
    

    def calc_score(self, prefix=""):    # MAIN BOTTLENECK!!
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func} from [L{self.galid} at {self.iout}]"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger)

        timekeys = self.branch.candidates.keys()
        if self.prog:
            igyr = self.data.load_snap(self.iout, prefix=prefix).params['age']
        for tth, tout in enumerate(timekeys):
            candidates = self.branch.candidates[tout]
            
            ith = 0
            for i, ileaf in candidates.items():
                self.branch.connect(ileaf, prefix=prefix)
                # try:
                score0 = self.calc_matchrate(ileaf, checkpid=self.branch.inipid, weight=self.branch.inipwei, prefix=prefix, checkiout=self.branch.rootout, checkid=self.branch.rootid)
                score1 = self.calc_matchrate(ileaf, checkpid=self.pid, weight=self.pweight, prefix=prefix) # importance weighted matchrate
                score2 = 0  # Velocity offset
                score3 = 0
                if score1 > 0 or score0 > 0:
                    if self.interplay:
                        score1 *= self.calc_matchrate(self, checkpid=ileaf.pid, weight=ileaf.pweight, prefix=prefix, checkiout=ileaf.iout, checkid=ileaf.galid)
                    score2 = self.calc_velocity_offset(ileaf, prefix=prefix)
                    score3 = np.exp( -np.abs(np.log10(self.gal_gm['m']/ileaf.gal_gm['m'])) )   # Mass difference
                self.branch.scores[tout][i] += score0+score1+score2+score3
                self.branch.score0s[tout][i] += score0
                self.branch.score1s[tout][i] += score1
                self.branch.score2s[tout][i] += score2
                self.branch.score3s[tout][i] += score3
                ith += 1

        clock.done()

    def calc_matchrate(self, otherleaf, checkpid=None, weight=None, checkiout=0, checkid=0, prefix=""):
        # Subject to `calc_score` BOTTLENECK!!
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger, level='debug')

        if checkiout == 0:
            checkiout = self.iout
        if checkid == 0:
            checkid = self.galid

        calc = False
        ioutkeys = list(otherleaf.saved_matchrates.keys())
        if (checkiout in ioutkeys):
            igalkeys = list(otherleaf.saved_matchrates[checkiout].keys())
            if checkid in igalkeys:
                val = otherleaf.saved_matchrates[checkiout][checkid]
                self.data.debugger.debug(f"{[prefix]} [G{checkid} at {checkiout}] is already saved in [L{otherleaf.galid} at {otherleaf.iout}]")
            else:
                calc = True
        else:
            otherleaf.saved_matchrates[checkiout] = {}
            calc = True

        if calc:
            if checkpid is None:
                checkpid = self.pid
            
            if otherleaf.nparts < len(checkpid)/200:
                return -1

            ind = large_isin(checkpid, otherleaf.pid)
            clock.done(add=f"({len(checkpid)} vs {otherleaf.nparts})")
            if howmany(ind,True)==0:
                val = -1
            else:
                val = np.sum( weight[ind] ) / np.sum( weight )
            otherleaf.saved_matchrates[checkiout][checkid] = val
        return val


    def calc_bulkmotion(self, checkind=None, prefix="", **kwargs):
        # Subject to `calc_velocity_offset`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger, level='debug')


        if checkind is None:
            checkind = np.full(self.nparts, True)

        # try:
        weights = self.pweight[checkind]
        weights /= np.sum(weights)
        vx = np.convolve( self.pvx[checkind], weights[::-1], mode='valid' )[0]
        vy = np.convolve( self.pvy[checkind], weights[::-1], mode='valid' )[0]
        vz = np.convolve( self.pvz[checkind], weights[::-1], mode='valid' )[0]

        # except Warning as e:
        #     print("WARNING!! in calc_bulkmotion")
        #     breakpoint()
        #     self.data.debugger.warning("########## WARNING #########")
        #     self.data.debugger.warning(e)
        #     self.data.debugger.warning(self.summary())
        #     raise ValueError("velocity wrong!")

        clock.done(add=f"({howmany(checkind, True)})")
        return np.array([vx, vy, vz])

    
    def calc_velocity_offset(self, otherleaf, prefix="", **kwargs):
        # Subject to `calc_score`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.data.debugger, level='debug')

        calc = False
        ioutkeys = list(otherleaf.saved_veloffsets.keys())
        if (self.iout in ioutkeys):
            igalkeys = list(otherleaf.saved_veloffsets[self.iout].keys())
            if self.galid in igalkeys:
                val = otherleaf.saved_veloffsets[self.iout][self.galid]
                self.data.debugger.debug(f"{[prefix]} [L{self.galid} at {self.iout}] is already saved in [L{otherleaf.galid}] at {otherleaf.iout}]")
            else:
                calc = True
        else:
            otherleaf.saved_veloffsets[self.iout] = {}
            calc = True
        
        if calc:
            ind = large_isin(otherleaf.pid, self.pid)
            if howmany(ind, True) < 3:
                clock.done(add=f"({self.nparts} vs {howmany(ind, True)})")
                val = 0
            else:
                # refv = self.calc_bulkmotion(useold=True, prefix=prefix)         # selfvel
                refv = np.array([self.gal_gm['vx'], self.gal_gm['vy'], self.gal_gm['vz']])
                inv = otherleaf.calc_bulkmotion(checkind=ind, prefix=prefix) - refv  # invel at self-coo
                # self.data.debugger.debug(f"gal{refv}, ref{inv}")
                totv = np.array([otherleaf.gal_gm['vx'], otherleaf.gal_gm['vy'], otherleaf.gal_gm['vz']]) - refv # totvel at self-coo
                # self.data.debugger.debug(f"gal{otherleaf.gal_gm[['vx','vy','vz']]}, ref{totv}")
                val = 1 - nbnorm(totv - inv)/(nbnorm(inv)+nbnorm(totv))
            otherleaf.saved_veloffsets[self.iout][self.galid] = val

            clock.done(add=f"({self.nparts} vs {howmany(ind, True)})")
        # return 1 - np.sqrt( np.sum((totv - inv)**2) )/(np.linalg.norm(inv)+np.linalg.norm(totv))
        return val