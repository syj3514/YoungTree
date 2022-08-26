import inspect
from numpy.lib.recfunctions import append_fields, drop_fields
from rur import uri

from tree_utool import *

#########################################################
###############         Leaf Class                #######
#########################################################
class Leaf():
    __slots__ = ['debugger', 'verbose', 'branch', 'parents', 'data', 
                'mode', 'galaxy', 'gal_gm', 'galid','iout', 'istep',
                'nextids', 'pruned','interplay',
                'nparts','pid','page','pm','pweight',
                'px','py','pz','pvx','pvy','pvz', 'prog']
    def __init__(self, gal, BranchObj, DataObj, verbose=1, prefix="", debugger=None, interplay=False, prog=True, **kwargs):
        func = f"[__Leaf__]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=verbose, debugger=debugger)
        self.debugger=debugger
        self.verbose = verbose

        self.branch = BranchObj
        self.parents = [self.branch.root['id']]
        self.data = DataObj
        self.mode, self.galaxy = self.branch.mode, self.branch.galaxy
        self.gal_gm = gal
        self.galid, self.iout = self.gal_gm['id'], self.gal_gm['timestep']
        self.istep = out2step(self.iout, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
        
        self.nextids = {}
        self.pruned = False
        self.interplay = interplay
        prefix += f"<ID{self.galid}:iout(istep)={self.iout}({self.istep})>"

        # for fast I/O
        self.nparts = 0
        self.pid = None
        self.px = None; self.py = None; self.pz = None
        self.pvx = None; self.pvy = None; self.pvz = None
        self.page = None; self.pm = None; self.pweight = None
        self.prog = prog

        self.load_parts(prefix=prefix)
        self.importance(prefix=prefix, usevel=True)

        # clock.done()
    
    def __del__(self):
        self.debugger.info(f"[DEL] Leaf (root={self.galid} at {self.iout}) is destroyed")

    def __str__(self):
        if self.branch is None:
            return f"<Leaf object> {self.data.galstr}{self.galid} at iout{self.iout}\n\troot parents: {self.parents}\n\tPruned!"
        text = f"<Leaf object> {self.data.galstr}{self.galid} at iout{self.iout}\n\troot parents: {self.parents}\n\tcurrent branch: {self.branch.root['id']}\n\t{self.nparts} {self.data.partstr}s"
        return text

    def clear(self, msgfrom='self'):
        if len(self.parents)==0:
            self.debugger.info(f"[CLEAR] Leaf (root={self.galid}) [from {msgfrom}]")
            self.pid=None; self.page=None; self.pm=None; self.pweight=None
            self.px=None; self.py=None; self.py=None
            self.pvx=None; self.pvy=None; self.pvy=None
            self.nparts=0
            self.gal_gm = None
            self.branch = None
            self.pruned = True

    def load_parts(self, prefix=""):
        # Subject to `__init__`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        temp = self.data.load_part(self.iout, self.galid, prefix=prefix, galaxy=self.galaxy)
        self.pid = temp['id']
        self.nparts = len(self.pid)
        self.px = temp['x']; self.py = temp['y']; self.pz = temp['z']
        self.pvx = temp['vx', 'km/s']; self.pvy = temp['vy', 'km/s']; self.pvz = temp['vz', 'km/s']
        self.page = temp['age', 'Gyr']; self.pm = temp['m']

        self.debugger.debug(prefix+f" [ID{self.galid} iout{self.iout}] Nparts={self.gal_gm['nparts']}({self.nparts})")

        clock.done()
    

    def importance(self, prefix="", usevel=True):
        # Subject to `__init__`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        
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
            #     self.debugger.warning("########## WARNING #########")
            #     self.debugger.warning(e)
            #     self.debugger.warning(f"gal velocity {cvx}, {cvy}, {cvz}")
            #     self.debugger.warning(f"len parts {self.nparts}")
            #     self.debugger.warning(f"vels first tens {vels[:10]}")
            #     self.debugger.warning(f"vels std {np.std(vels)}")
            #     self.debugger.warning(self.summary())
            #     breakpoint()
            #     raise ValueError("velocity wrong!")
        self.pweight = self.pm/dist

        clock.done()
    

    def load_nextids(self, igals, njump=0, masscut_percent=1, nnext=5, prefix="", **kwargs): # MAIN BOTTLENECK!!
        # Subject to `find_candidates`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        iout, istep = ioutistep(igals[0], galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
        isnap = self.data.load_snap(iout, prefix=prefix)
        jstep = istep-1-njump if self.prog else istep+1+njump
        jout = step2out(jstep, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
        jsnap = self.data.load_snap(jout, prefix=prefix)
        jgals = self.data.load_gal(jout, 'all', return_part=False, prefix=prefix) # BOTTLENECK!!
        dt = np.abs( isnap.params['age'] - jsnap.params['age'] )
        dt *= 1e9 * 365 * 24 * 60 * 60 # Gyr to sec
        nexts = np.zeros(3).astype(int)
        
        for igal in igals: # tester galaxies at iout
            ivel = rms(igal['vx'], igal['vy'], igal['vz'])
            radii = 5*max(igal['r'],1e-4) + 5*dt*ivel*jsnap.unit['km']
            neighbors = cut_sphere(jgals, igal['x'], igal['y'], igal['z'], radii, both_sphere=True)
            self.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} in radii")
            
            
            if len(neighbors)>0: # candidates at jout
                neighbors = neighbors[(neighbors['m'] >= igal['m']*masscut_percent/100) & (~np.isin(neighbors['id'], nexts))]
                self.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} after masscut {masscut_percent} percent")
                if len(neighbors) > nnext or len(igals) > 2*nnext:
                    _, checkpid = self.data.load_gal(iout, igal['id'], return_part=True, prefix=prefix)

                    rate = np.zeros(len(neighbors))
                    gals, gmpids = self.data.load_gal(jout, neighbors['id'], return_part=True, prefix=prefix)
                    # iout, istep = ioutistep(igal, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
                    ith = 0
                    for _, gmpid in zip(gals, gmpids):
                        if atleast_numba(checkpid, gmpid):
                            ind = large_isin(checkpid, gmpid)
                            rate[ith] = howmany(ind, True)/len(checkpid)
                        ith += 1
                    ind = rate>0

                    neighbors, rate = neighbors[ind], rate[ind]
                    self.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} after crossmatch")
                    if len(neighbors) > 0:
                        if len(neighbors) > nnext or len(igals) > 2*nnext:
                            arg = np.argsort(rate)
                            neighbors = neighbors[arg][-nnext:]
                            self.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} after score sorting")
                        nexts = np.concatenate((nexts, neighbors['id']))
                elif len(neighbors) > 0:
                    nexts = np.concatenate((nexts, neighbors['id']))
                else:
                    pass
            
        nexts = np.concatenate((nexts, np.zeros(3).astype(int)))
        
        clock.done()
        return np.unique( nexts[nexts>0] ), jout
    

    def find_candidates(self, masscut_percent=1, nstep=5, nnext=5, prefix="", **kwargs):
        ############################################################
        ########    ADD case when no nexther, jump beyond     ####### Maybe done?
        ############################################################
        # if not self.nextids:
        keys = list(self.nextids.keys())
        if len(keys) < 5:
            func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
            clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

            igals = np.atleast_1d(self.gal_gm)
            njump=0
            for i in range(nstep):
                jstep = self.istep-1-i if self.prog else self.istep+1+i
                if jstep > 0 and jstep <= np.max(self.data.nstep):
                    jout = step2out(jstep, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
                    if jout in keys:
                        nextids = self.nextids[jout]
                    else:
                        nextids, jout = self.load_nextids(igals, njump=njump, masscut_percent=masscut_percent, nnext=nnext, prefix=prefix, **kwargs)
                        self.nextids[jout] = nextids
                    self.debugger.info(f"*** jout(jstep)={jout}({jstep}), nexts={nextids}")
                    if len(nextids) == 0:
                        self.debugger.info("JUMP!!")
                        njump += 1
                    else:
                        nextids = self.branch.update_cands(jout, nextids, checkids=self.pid, prefix=prefix) # -> update self.branch.candidates & self.branch.scores
                        if len(nextids) == 0:
                            self.debugger.info("JUMP!!")
                            njump += 1
                        else:
                            njump = 0
                            igals = self.data.load_gal(jout, nextids, return_part=False, prefix=prefix)

            clock.done()
        keys = list(self.nextids.keys())
        if len(keys)==0:
            return False
        temp = np.sum([len(self.nextids[key]) for key in keys])
        return temp > 0
    

    def calc_score(self, prefix=""):    # MAIN BOTTLENECK!!
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        timekeys = self.branch.candidates.keys()
        checkpid = self.pid
        checkwei = self.pweight
        if self.prog:
            igyr = self.data.load_snap(self.iout, prefix=prefix).params['age']
            for tth, tout in enumerate(timekeys):
                candidates = self.branch.candidates[tout]
                if self.galaxy:
                    tgyr = self.data.load_snap(tout, prefix=prefix).params['age']
                    ageind = self.page >= (igyr-tgyr)
                    checkpid = self.pid[ ageind ]
                    checkwei = self.pweight[ ageind ]
            
            ith = 0
            for i, ileaf in candidates.items():
                # try:
                score0 = self.calc_matchrate(ileaf, checkpid=self.branch.inipid, weight=self.branch.inipwei, prefix=prefix)
                score1 = self.calc_matchrate(ileaf, checkpid=checkpid, weight=checkwei, prefix=prefix) # importance weighted matchrate
                score2 = 0  # Velocity offset
                score3 = 0
                if score1 > 0 or score0 > 0:
                    if self.interplay:
                        score1 *= self.calc_matchrate(self, checkpid=ileaf.pid, weight=ileaf.pweight, prefix=prefix)
                    score2 = self.calc_velocity_offset(ileaf, prefix=prefix)
                    score3 = np.exp( -np.abs(np.log10(self.gal_gm['m']/ileaf.gal_gm['m'])) )   # Mass difference
                # except Warning as e:
                #     print("WARNING!! in calc_score")
                #     breakpoint()
                #     self.debugger.warning("########## WARNING #########")
                #     self.debugger.warning(e)
                #     self.debugger.warning(self.summary())
                #     raise ValueError("velocity wrong!")
                self.branch.scores[tout][i] += score0+score1+score2+score3
                self.branch.score0s[tout][i] += score0
                self.branch.score1s[tout][i] += score1
                self.branch.score2s[tout][i] += score2
                self.branch.score3s[tout][i] += score3
                ith += 1

        clock.done()

    # def atleast_leaf(self, otherleaves, checkpid, prefix=""):
    #     func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
    #     clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger, level='debug')

    #     ids = tuple(otherleaf.pid for otherleaf in otherleaves)
    #     val = atleast_numba_para(ids, checkpid)

    #     clock.done()
    #     return val # True or False

    def calc_matchrate(self, otherleaf, checkpid=None, weight=None, prefix=""):
        # Subject to `calc_score` BOTTLENECK!!
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger, level='debug')

        if checkpid is None:
            checkpid = self.pid
            if self.galaxy and self.prog:
                tgyr = self.data.load_snap(otherleaf.iout, prefix=prefix).params['age']
                igyr = self.data.load_snap(self.iout, prefix=prefix).params['age']
                checkpid = self.pid[ self.page >= (igyr-tgyr) ]
        
        if otherleaf.nparts < len(checkpid)/200:
            return -1

        # if atleast_numba(checkpid, otherleaf.pid):
        #     ind = large_isin(checkpid, otherleaf.pid)
        # else:
        #     clock.done(add=f"({len(checkpid)} vs {otherleaf.nparts})")
        #     return -1
        ind = large_isin(checkpid, otherleaf.pid)
        clock.done(add=f"({len(checkpid)} vs {otherleaf.nparts})")
        if howmany(ind,True)==0:
            return -1
        return np.sum( weight[ind] ) / np.sum( weight )


    def calc_bulkmotion(self, checkind=None, prefix="", **kwargs):
        # Subject to `calc_velocity_offset`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger, level='debug')


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
        #     self.debugger.warning("########## WARNING #########")
        #     self.debugger.warning(e)
        #     self.debugger.warning(self.summary())
        #     raise ValueError("velocity wrong!")

        clock.done(add=f"({howmany(checkind, True)})")
        return np.array([vx, vy, vz])

    
    def calc_velocity_offset(self, otherleaf, prefix="", **kwargs):
        # Subject to `calc_score`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger, level='debug')

        ind = large_isin(otherleaf.pid, self.pid)
        if howmany(ind, True) < 3:
            clock.done(add=f"({self.nparts} vs {howmany(ind, True)})")
            return 0
        else:
            # refv = self.calc_bulkmotion(useold=True, prefix=prefix)         # selfvel
            refv = np.array([self.gal_gm['vx'], self.gal_gm['vy'], self.gal_gm['vz']])
            inv = otherleaf.calc_bulkmotion(checkind=ind, prefix=prefix) - refv  # invel at self-coo
            # self.debugger.debug(f"gal{refv}, ref{inv}")
            totv = np.array([otherleaf.gal_gm['vx'], otherleaf.gal_gm['vy'], otherleaf.gal_gm['vz']]) - refv # totvel at self-coo
            # self.debugger.debug(f"gal{otherleaf.gal_gm[['vx','vy','vz']]}, ref{totv}")

        clock.done(add=f"({self.nparts} vs {howmany(ind, True)})")
        # return 1 - np.sqrt( np.sum((totv - inv)**2) )/(np.linalg.norm(inv)+np.linalg.norm(totv))
        return 1 - nbnorm(totv - inv)/(nbnorm(inv)+nbnorm(totv))