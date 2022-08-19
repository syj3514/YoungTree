import inspect
from numpy.lib.recfunctions import append_fields, drop_fields
from rur import uri

from tree_utool import *

#########################################################
###############         Leaf Class                #######
#########################################################
class Leaf():
    def __init__(self, gal, BranchObj, DataObj, verbose=1, prefix="", debugger=None, interplay=False, **kwargs):
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
        
        self.fatids = {}
        self.part = None
        self.pruned = False
        self.interplay = interplay
        prefix += f"<ID{self.galid}:iout(istep)={self.iout}({self.istep})>"
        self.refv = None
        self.load_parts(prefix=prefix)
        self.importance(prefix=prefix, usevel=True)

        # clock.done()

    def __str__(self):
        if self.branch is None:
            return f"<Leaf object> {self.data.galstr}{self.galid} at iout{self.iout}\n\troot parents: {self.parents}\n\tPruned!"
        text = f"<Leaf object> {self.data.galstr}{self.galid} at iout{self.iout}\n\troot parents: {self.parents}\n\tcurrent branch: {self.branch.root['id']}\n\t{len(self.part['id'])} {self.data.partstr}s"
        return text

    def clear(self):
        if len(self.parents)==0:
            self.gal_gm = None
            self.branch = None
            self.part = []
            self.pruned = True

    def load_parts(self, prefix=""):
        # Subject to `__init__`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        self.part = self.data.load_part(self.iout, self.galid, prefix=prefix, galaxy=self.galaxy)
        self.debugger.debug(prefix+f" [ID{self.galid} iout{self.iout}] Nparts={self.gal_gm['nparts']}({len(self.part['x'])})")

        clock.done()
    

    def importance(self, prefix="", usevel=True):
        # Subject to `__init__`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
    
        cx, cy, cz = self.gal_gm['x'],self.gal_gm['y'],self.gal_gm['z']
        dist = distance3d(cx,cy,cz, self.part['x'], self.part['y'], self.part['z']) / self.gal_gm['rvir']
        if usevel:
            try:
                cvx, cvy, cvz = self.gal_gm['vx'],self.gal_gm['vy'],self.gal_gm['vz']
                vels = distance3d(cvx,cvy,cvz, self.part['vx'], self.part['vy'], self.part['vz'])
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
        table = append_fields(self.part.table, "dist", np.empty(
            self.part.table.shape[0], dtype="<f8"), dtypes="<f8")
        table["dist"] = dist
        self.part = uri.RamsesSnapshot.Particle(table, self.part.snap)

        # clock.done()
    

    def load_fatids(self, igals, njump=0, masscut_percent=1, nfat=5, prefix="", **kwargs):
        # Subject to `find_candidates`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        iout, istep = ioutistep(igals[0], galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
        jout = step2out(istep-1-njump, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
        jgals = self.data.load_gal(jout, 'all', return_part=False, prefix=prefix)
        dt = self.data.load_snap(iout, prefix=prefix).params['age'] - self.data.load_snap(jout, prefix=prefix).params['age']
        dt *= 1e9 * 365 * 24 * 60 * 60 # Gyr to sec
        fats = np.zeros(3).astype(int)
        
        for igal in igals: # tester galaxies at iout
            ivel = rms(igal['vx'], igal['vy'], igal['vz'])
            isnap = self.data.load_snap(jout, prefix=prefix)
            radii = 5*max(igal['r'],1e-4) + 5*dt*ivel*isnap.unit['km']
            neighbors = cut_sphere(jgals, igal['x'], igal['y'], igal['z'], radii, both_sphere=True)
            self.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} in radii")
            if len(neighbors) > nfat:
                _, checkpart = self.data.load_gal(iout, igal['id'], return_part=True, prefix=prefix)
            
            if len(neighbors)>0: # candidates at jout
                neighbors = neighbors[neighbors['m'] >= igal['m']*masscut_percent/100]
                self.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} after masscut {masscut_percent} percent")
                if len(neighbors) > nfat:
                    rate = np.zeros(len(neighbors))
                    gals, gmpids = self.data.load_gal(jout, neighbors['id'], return_part=True, prefix=prefix)
                    iout, istep = ioutistep(igal, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
                    ith = 0
                    for _, gmpid in zip(gals, gmpids):
                        if atleast_numba(checkpart, gmpid):
                            ind = large_isin(checkpart, gmpid)
                            rate[ith] = howmany(ind, True)/len(checkpart)
                        ith += 1
                    ind = rate>0

                    neighbors, rate = neighbors[ind], rate[ind]
                    self.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} after crossmatch")
                    if len(neighbors) > 0:
                        if len(neighbors) > nfat:
                            arg = np.argsort(rate)
                            neighbors = neighbors[arg][-nfat:]
                            self.debugger.debug(f"igal[{igal['id']}] len={len(neighbors)} after score sorting")
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
        # if not self.fatids:
        keys = list(self.fatids.keys())
        if len(keys) < 5:
            func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
            clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

            igals = np.atleast_1d(self.gal_gm)
            njump=0
            for i in range(nstep):
                if self.istep-1-i > 0:
                    iout = step2out(self.istep-1-i, galaxy=self.galaxy, mode=self.mode, nout=self.data.nout, nstep=self.data.nstep)
                    if iout in keys:
                        fatids = self.fatids[iout]
                    else:
                        fatids, jout = self.load_fatids(igals, njump=njump, masscut_percent=masscut_percent, nfat=nfat, prefix=prefix, **kwargs)
                        self.fatids[iout] = fatids
                    self.debugger.info(f"*** iout(istep)={iout}({self.istep-1-i}), fats={fatids}")
                    if len(fatids) == 0:
                        self.debugger.info("JUMP!!")
                        njump += 1

                        # if not self.fatids:
                        #     self.fatids=False
                    else:
                        njump = 0
                        # self.fatids=True
                        self.branch.update_cands(iout, fatids, checkids=self.part['id'], prefix=prefix) # -> update self.branch.candidates & self.branch.scores
                        igals = self.data.load_gal(iout, fatids, return_part=False, prefix=prefix)

            clock.done()
        keys = list(self.fatids.keys())
        if len(keys)==0:
            return False
        temp = np.sum([len(self.fatids[key]) for key in keys])
        return temp > 0
    

    def calc_score(self, prefix=""):
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        timekeys = self.branch.candidates.keys()
        for tout in timekeys:
            checkpart = self.part
            if self.galaxy:
                tgyr = self.data.load_snap(tout, prefix=prefix).params['age']
                igyr = self.data.load_snap(self.iout, prefix=prefix).params['age']
                candidates = self.branch.candidates[tout]
                checkpart = self.part[ self.part['age','Gyr'] >= (igyr-tgyr) ]
            for i, ileaf in candidates.items():
                try:
                    score0 = self.calc_matchrate(ileaf, checkpart=self.branch.inipart, prefix=prefix)
                    score1 = self.calc_matchrate(ileaf, checkpart=checkpart, prefix=prefix) # importance weighted matchrate
                    score2 = 0  # Velocity offset
                    score3 = 0
                    if score1 > 0 or score0 > 0:
                        if self.interplay:
                            score1 *= self.calc_matchrate(self, checkpart=ileaf.part, prefix=prefix)
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
                self.branch.scores[tout][i] += score0+score1+score2+score3
                self.branch.score0s[tout][i] += score0
                self.branch.score1s[tout][i] += score1
                self.branch.score2s[tout][i] += score2
                self.branch.score3s[tout][i] += score3

        clock.done()


    def calc_matchrate(self, otherleaf, checkpart=None, prefix=""):
        # Subject to `calc_score`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        if checkpart is None:
            checkpart = self.part
            if self.galaxy:
                tgyr = self.data.load_snap(otherleaf.iout, prefix=prefix).params['age']
                igyr = self.data.load_snap(self.iout, prefix=prefix).params['age']
                checkpart = self.part[ self.part['age','Gyr'] >= (igyr-tgyr) ]
        if atleast_numba(checkpart['id'], otherleaf.part['id']):
            ind = large_isin(checkpart['id'], otherleaf.part['id'])
        else:
            # clock.done(add=f"({len(checkpart['id'])} vs {len(otherleaf.part['id'])})")
            return -1
        # clock.done(add=f"({len(checkpart['id'])} vs {len(otherleaf.part['id'])})")
        return np.sum( checkpart['m'][ind]/checkpart['dist'][ind] ) / np.sum( checkpart['m']/checkpart['dist'] )


    def calc_bulkmotion(self, checkpart=None, useold=False, prefix="", **kwargs):
        # Subject to `calc_velocity_offset`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)
        
        if (useold) and (self.refv is not None):
            # clock.done()
            return self.refv

        switch = False
        if checkpart is None:
            switch = True
            checkpart = self.part

        try:
            vx = np.average(checkpart['vx', 'km/s'], weights=checkpart['m']/checkpart['dist'])
            vy = np.average(checkpart['vy', 'km/s'], weights=checkpart['m']/checkpart['dist'])
            vz = np.average(checkpart['vz', 'km/s'], weights=checkpart['m']/checkpart['dist'])
            if switch:
                self.refv = np.array([vx,vy,vz])
        except Warning as e:
            print("WARNING!! in calc_bulkmotion")
            breakpoint()
            self.debugger.warning("########## WARNING #########")
            self.debugger.warning(e)
            self.debugger.warning(f"len parts {len(checkpart['m'])}")
            self.debugger.warning(f"dtype {checkpart.dtype}")
            self.debugger.warning(self.summary())
            raise ValueError("velocity wrong!")

        # clock.done(add=f"({len(checkpart['id'])})")
        return np.array([vx, vy, vz])

    
    def calc_velocity_offset(self, otherleaf, prefix="", **kwargs):
        # Subject to `calc_score`
        func = f"[{inspect.stack()[0][3]}]"; prefix = f"{prefix}{func}"
        # clock = timer(text=prefix, verbose=self.verbose, debugger=self.debugger)

        ind = large_isin(otherleaf.part['id'], self.part['id'])
        if howmany(ind, True) < 3:
            # clock.done(add=f"({len(self.part['id'])} vs {howmany(ind, True)})")
            return 0
        else:
            refv = self.calc_bulkmotion(useold=True, prefix=prefix)
            instar = otherleaf.part[ind]
            inv = self.calc_bulkmotion(checkpart=instar, prefix=prefix) - refv
            totv = np.array([otherleaf.gal_gm['vx'], otherleaf.gal_gm['vy'], otherleaf.gal_gm['vz']]) - refv

        # clock.done(add=f"({len(self.part['id'])} vs {howmany(ind, True)})")
        # return 1 - np.sqrt( np.sum((totv - inv)**2) )/(np.linalg.norm(inv)+np.linalg.norm(totv))
        return 1 - nbnorm(totv - inv) / (nbnorm(inv)+nbnorm(totv))