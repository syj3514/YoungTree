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
from collections.abc import Iterable
import inspect
import gc
from copy import deepcopy
gc.collect()
import psutil

from datetime import datetime

#################################################
#################################################
#################################################
#################################################
#################################################
def current_time():
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

def inside(a,b):
    '''
    xa1 <= xb1 <= xb2 <= xa2
    '''
    ind1 = (a[:,0] <= b[:,0]).all()
    ind2 = (a[:,1] <= b[:,2]).all()
    return ind1*ind2

def mergebox(a,b):
    shape = a.shape
    newbox = np.zeros(shape)
    for idim in shape[0]:
        newbox[idim, 0] = min(a[idim, 0], b[idim, 0])
        newbox[idim, 1] = max(a[idim, 1], b[idim, 1])
    return newbox



#################################################
#################################################
#################################################
#################################################
#################################################
class youngtree():
    def __init__(self, treemaker, inigals, lengstep=4, mode='hagn', galaxy=True, verbose=3, prefix="", debug=False, **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        self.verbose=verbose
        self.debug = debug
        if self.verbose > 2 or self.debug:
            print(f"{prefix}{func} START")
        ref = time.time()
        self.galaxy=galaxy
        self.mode=mode
        if mode[0] == 'h':
            self.repo = f'/storage4/Horizon_AGN'
        elif mode[0] == 'y':
            self.repo = f'/storage3/Clusters/{mode[1:]}'
        self.gmpids = {}
        self.galaxymakers = {}
        # self.cumpartss = {}
        self.treemakers = treemaker
        if galaxy:
            self.nout, self.nstep, self.zred, self.aexp, self.gyr = jn.pklload(f'/storage6/jeon/data/{mode}/{mode}_nout_nstep_zred_aexp_gyr.pickle')
        else:
            self.nout, self.nstep, self.zred, self.aexp, self.gyr = jn.pklload(f'/storage6/jeon/data/{mode}/dm_{mode}_nout_nstep_zred_aexp_gyr.pickle')
        self.preserve_iout = []
        self.stars = {}
        self.snaps = {}
        self.cache_parts = {}
        self.cache_refers = {}
        self.cache_count = 0
        self.memory = 0

        self.inigals = inigals
        self.sandbox = {}

        self.lengstep = lengstep
        for inigal in inigals:
            iout, _ = jn.ioutistep(inigal, galaxy=self.galaxy, mode=self.mode)
            if not iout in self.stars.keys():
                self.stars[iout] = self.load_star(iout, prefix=f"\t{prefix}{func}")
                self.preserve_iout.append(iout)
            self.sandbox[inigal['id']] = self.Sandbox(self, inigal, lengstep=self.lengstep, verbose=self.verbose-1, prefix=f"\t{func}", debug=self.debug)

        if self.verbose > 1 or self.debug:
            jn.printtime(ref, f"{prefix}{func} Done")
    
    def __str__(self, **kwargs):
        t1 = f"SY's merger tree builder. (Use galaxy={self.galaxy})\n"
        t2 = f"Sim: {self.mode}\n"
        t3 = f"{len(self.sandbox.keys())} galaxies are saved in sandboxes\n"
        t4 = f"Example of Sandbox object:\n"
        key = list(self.sandbox.keys())[0]
        st1 = f"\tID={self.sandbox[key].root['id']}\n"
        st2 = f"\tNow saved parking lots: {self.sandbox[key].parking.keys()}\n"
        # st3 = f"\tCurrent branch:\n"
        # st4 = f"\t{self.sandbox[key].results_gids}"
        return t1+t2+t3+t4+st1+st2#+st3+st4


    def load_galaxymaker(self, dict_done=None, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        if self.verbose > 2 or self.debug:
            print(f"{prefix}{func} START")
        ref = time.time()     

        if dict_done is None:
            last50 = np.zeros(50)*np.nan
            for i, iout in enumerate(self.nout):
                ref2 = time.time()
                if iout not in self.galaxymakers.keys():
                    snap = self.load_snap(iout)
                    if dict_done is None:
                        gm, gmpid = uhmi.HaloMaker.load(snap, galaxy=self.galaxy, load_parts=True)
                        cumparts = np.insert(np.cumsum(gm["nparts"]), 0, 0)
                        inst = np.copy(gmpid)
                        gmpid = [inst[ cumparts[i]:cumparts[i+1] ] for i in range(len(gm))]
                        self.galaxymakers[iout] = np.copy(gm)
                        self.gmpids[iout] = gmpid
                        time.sleep(1)
                        # self.cumpartss[iout] = cumparts
                if self.verbose > 2 and not self.debug:
                    last50 = jn.timeestimator_inst(ref2, i+1, len(self.nout), last100=last50)
                    if i%10 == 9:
                        jn.clear_output(wait=True)
            del gm
            del gmpid
            del cumparts
            del inst
        else:
            self.galaxymakers = dict_done["galaxymakers"]
            self.gmpids = dict_done["gmpids"]
            del dict_done
            # self.cumpartss = dict_done["cumpartss"]

        if self.verbose > 1 or self.debug:
            if self.verbose > 2 and not self.debug:
                time.sleep(1)
                jn.clear_output(wait=True)
            jn.printtime(ref, f"{prefix}{func} Done")

    ##########################################################
    class Sandbox():
        def __init__(self, treeobj, gal, lengstep=4, verbose=3, prefix="", debug=False, **kwargs):
            func = f"[{inspect.stack()[0][3]}]"
            self.debug=debug
            self.verbose = verbose
            ###### It takes very short time!
            # if self.verbose > 2 or self.debug:
            #     print(f"{prefix}{func} START")
            ref = time.time()
            self.tree = treeobj
            self.root = gal
            self.nowtarget = treeobj.gm2tm(gal)
            iout, istep = jn.ioutistep(gal, galaxy=self.tree.galaxy, mode=self.tree.mode)
            self.simmaxstep = np.max(self.tree.nstep)
            self.startout = iout
            self.startstep = istep
            self.done = False

            self.parking = {}
            self.scores = {}
            self.lengstep = lengstep

            self.results_gids = np.zeros(len(self.tree.nstep)).astype(int) - 1

            self.parking[istep] = np.array([self.nowtarget])
            self.scores[istep] = np.array([0])
            self.results_gids[self.simmaxstep - istep] = gal['id']
            # if self.verbose > 2 or self.debug:
            #     jn.printtime(ref, f"{prefix}{func} Done")
        
        # SANDBOX
        def choose_winner(self,targetstep, prefix="", **kwargs):
            ###### It takes very short time!
            # func = f"[{inspect.stack()[0][3]}]"
            # ref = time.time()
            # if self.verbose > 2 or self.debug:
            #     print(f"{prefix}{func} START")
            argmin = np.argmin(self.scores[targetstep])
            winner = self.parking[targetstep][argmin]

            self.results_gids[self.simmaxstep - targetstep] = winner['id']
            self.nowtarget = self.tree.gm2tm(winner)

            del self.parking[targetstep]
            del self.scores[targetstep]
            # if self.verbose > 2 or self.debug:
            #     jn.printtime(ref, f"{prefix}{func} Done")

        # SANDBOX
        def receive_candidates(self, cands, prefix="", **kwargs):
            ###### It takes very short time!
            # func = f"[{inspect.stack()[0][3]}]"
            # if self.verbose > 2 or self.debug:
            #     print(f"{prefix}{func} START")
            ref = time.time()
            if len(np.unique(cands['timestep'])) == 1:
                iout, istep = jn.ioutistep(cands[0], galaxy=self.tree.galaxy, mode=self.tree.mode)
                if istep in self.parking.keys():
                    oldid = self.parking[istep]['id']
                    newid = cands['id']
                    issame = (set(oldid) == set(newid))
                    if not issame:
                        new = ~np.isin(newid, oldid)
                        cands = cands[new]
                        self.parking[istep] = np.append(self.parking[istep], cands)
                else:
                    self.parking[istep] = cands
            else: # If candidates have different timesteps for each other.
                for cand in cands:
                    iout, istep = jn.ioutistep(cand, galaxy=self.tree.galaxy, mode=self.tree.mode)
                    if istep in self.parking.keys():
                        if not cand['id'] in self.parking[istep]['id']:
                            self.parking[istep] = np.append(self.parking[istep], cand)
                    else:
                        self.parking[istep] = cand
            # if self.verbose > 2 or self.debug:
            #     jn.printtime(ref, f"{prefix}{func} Done")
        
        # SANDBOX
        def calc_score(self, agecut=True, mass_weight=True, prefix="", **kwargs):
            func = f"[{inspect.stack()[0][3]}]"
            keys = list(self.parking.keys())
            if self.verbose > 2 or self.debug:
                print(f"{prefix}{func} START ({len(keys)} steps)")
            ref = time.time()
            
            for key in keys:
                if key != self.nowtarget['timestep']:
                    cands = self.parking[key]
                    if self.nowtarget['timestep'] in keys:
                        scores = np.zeros(len(cands)).astype(int)
                    else:
                        scores = self.tree.scoring(self.nowtarget, cands, agecut=agecut, mass_weight=mass_weight, prefix=f"\t{prefix}{func}")
                    
                    if key in self.scores.keys():
                        self.scores[key] += scores
                    else:
                        self.scores[key] = scores
                scores=None
            if self.verbose > 2 or self.debug:
                jn.printtime(ref, f"{prefix}{func} Done ({len(keys)} steps)")
        
        # SANDBOX
        def prune_gals(self, maxngal=20, prefix="", **kwargs):
            ###### It takes very short time!
            # func = f"[{inspect.stack()[0][3]}]"
            # if self.verbose > 2 or self.debug:
            #     print(f"{prefix}{func} START")
            ref = time.time()
            targetstep = np.max(list(self.parking.keys()))
            scores = self.scores[targetstep]
            if len(scores) > maxngal:
                arg = np.argpartition(scores, maxngal)[:maxngal]
                self.scores[targetstep] = self.scores[targetstep][arg]
                self.parking[targetstep] = self.parking[targetstep][arg]
            # if self.verbose > 2 or self.debug:
            #     jn.printtime(ref, f"{prefix}{func} Done")
        
        # SANDBOX
        def extract_branch(self, prefix="", **kwargs):
            func = f"[{inspect.stack()[0][3]}]"
            if self.verbose > 2 or self.debug:
                print(f"{prefix}{func} START")
            ref = time.time()
            timesteps = self.simmaxstep - np.arange(len(self.results_gids))
            ind = self.results_gids > 0
            gids, timesteps = self.results_gids[ind], timesteps[ind]
            i=0
            for gid, timestep in zip(gids, timesteps):
                gal = self.tree.treemakers.loadgals(timestep, gid)
                if i==0:
                    gals = gal
                    i+=1
                else:
                    gals = np.append(gals, gal)
            if self.verbose > 2 or self.debug:
                jn.printtime(ref, f"{prefix}{func} Done")
            return gals

    ##########################################################
    def _do_onestep(self, nfat=5, maxngal=20, agecut=True, mass_weight=False, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        if self.verbose > 2 or self.debug:
            print(f"\n{prefix}{func} START ({len(self.inigals['id'])} gals)")
        ref = time.time()
        notyet = False
        last50 = np.zeros(50)*np.nan
        targetstepmin = -np.inf
        for j, ith in enumerate(self.inigals['id']):
            ref2 = time.time()
            ibox = self.sandbox[ith]
            if not ibox.done:
                notyet = True
                already_keys = list(ibox.parking.keys())
                targetstep = np.max(already_keys)
                targetstepmin = max(targetstepmin,targetstep)

                ibox.prune_gals(maxngal=maxngal, prefix=f"\t{prefix}{func}")
                progs = ibox.parking[targetstep]

                # Find progenitor candidates of gals in (T) step
                for i in range(self.lengstep):
                    if (targetstep - i) <= 1:
                        break
                    # i) If (T-1) gals already saved
                    if (targetstep-i-1) in already_keys:
                        # --> Just load saved progs
                        progs = ibox.parking[targetstep-i-1]
                    # ii) (T-1) gals not saved yet
                    else:
                        progs = self.load_fats_of_gals(progs, masscut_percent=0, nfat=nfat, prefix=f"\t{prefix}{func}")
                        # ii-1) No progs in (T-1) step
                        if progs is None:
                            ibox.done = True
                            break
                        # ii-2) Find progenitors in (T-1) step
                        else:
                            # ii-2-a) progs = [None]
                            if (len(progs) == 1) and (progs[0] is None):
                                ibox.done = True
                                break
                            elif len(progs) == 0:
                                ibox.done = True
                                break
                            else:
                                ibox.receive_candidates(progs, prefix=f"\t{prefix}{func}")
                progs = None
                del progs
                ibox.calc_score(agecut=agecut, mass_weight=mass_weight, prefix=f"\t{prefix}{func}")
                ibox.choose_winner(targetstep, prefix=f"\t{prefix}{func}")
                if len(list(ibox.parking.keys())) == 0:
                    ibox.done = True
            # if self.verbose > 0 and not self.debug:
            if self.verbose > 0 or self.debug:
                last50 = jn.timeestimator_inst(ref2, j+1, len(self.inigals['id']),message=f"\t[At {targetstepmin}, ID{ith} (mem {self.memory/1024:.2f} GB)]", last100=last50)
                if j%10==9 and not self.debug:
                    jn.clear_output(wait=True)
        if notyet:
            if targetstepmin + 1 in self.nstep:
                self.remove_ioutdata(jn.step2out(targetstepmin+1,galaxy=self.galaxy, mode=self.mode), prefix=f"\t{prefix}{func}")
        if not self.debug:
            jn.clear_output(wait=True)
        if self.verbose > 1 or self.debug:
            jn.printtime(ref, f"{prefix}{func} Done ({len(self.inigals['id'])} gals)")
        return notyet

    def Build_Tree(self, nfat=5, maxngal=20, agecut=True, mass_weight=False, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        ref = time.time()
        if self.verbose > 0 or self.debug:
            print(f"{prefix}{func} START")
        notyet = True
        _, maxstep = jn.ioutistep(self.inigals[0], galaxy=self.galaxy, mode=self.mode)
        last50 = np.zeros(50)*np.nan
        i = 0
        while notyet:
            i+=1
            ref2 = time.time()
            # jn.memory_usage(message="Before do_onestep")
            notyet = self._do_onestep(nfat=nfat, maxngal=maxngal, agecut=agecut, mass_weight=mass_weight, prefix=f"\t{prefix}{func}")
            # jn.memory_usage(message="After do_onestep")
            # print()
            # if self.verbose > 0 and not self.debug:
            if True:
                last50 = jn.timeestimator_inst(ref2, i, maxstep,message=f"\t[YT (mem {self.memory/1024:.2f} GB)]", last100=last50)
                if i%10 == 9 and not self.debug:
                    jn.clear_output(wait=True)
        
        if self.verbose > 0 or self.debug:
            jn.printtime(ref, f"{prefix}{func} Done")

    def return_branch(self, gal, **kwargs):
        return self.sandbox[gal['id']].extract_branch()

    ##########################################################
    def remove_ioutdata(self, iout, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        if self.verbose > 2 or self.debug:
            print(f"{prefix}{func} START")
            # jn.memory_usage(message="Before")
        ref = time.time()
        # if iout in self.stars.keys():
        #     del self.stars[iout]
        if iout in self.snaps.keys():
            if not iout in self.preserve_iout:
                # print(f"[###] Remove iout={iout}")
                self.snaps[iout].clear()
                # print(f"[###] self.snaps: refcount={sys.getrefcount(self.snaps[iout])}")
                self.snaps[iout] = None
                del self.snaps[iout]
                # print(f"[###] self.stars: refcount={sys.getrefcount(self.stars[iout])}")
                self.stars[iout] = None
                del self.stars[iout]
                # print(f"[###] self.cache_parts: refcount={sys.getrefcount(self.cache_parts[iout])}")
                self.cache_parts[iout] = None
                del self.cache_parts[iout]
                self.cache_refers[iout] = None
                del self.cache_refers[iout]
                self.cache_count = 10
                # print(f"[###] self.galaxymakers: refcount={sys.getrefcount(self.galaxymakers[iout])}")
                self.galaxymakers[iout] = None
                del self.galaxymakers[iout]
                # del self.gmpids[iout]
        gc.collect()
        if self.verbose > 2 or self.debug:
            # jn.memory_usage(message="After")
            jn.printtime(ref, f"{prefix}{func} Done")

    def gm2tm(self, gal, **kwargs):
        iout, istep = jn.ioutistep(gal, galaxy=self.galaxy, mode=self.mode)
        if 'nparts' in gal.dtype.names:
            return self.treemakers.loadgals(istep, gal['id'])
        else:
            return gal
        # return inst
    
    def tm2gm(self, gal, **kwargs):
        iout, istep = jn.ioutistep(gal, galaxy=self.galaxy, mode=self.mode)
        if 'nparts' in gal.dtype.names:
            return gal
        else:
            return self.galaxymakers[iout][gal['id']-1]
        # return inst

    def load_fats_of_gal(self, gal, masscut_percent=0, nfat=5, prefix="",**kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        ref = time.time()
        # if self.verbose > 2 or self.debug:
        #     print(f"{prefix}{func} START")
        iout, istep = jn.ioutistep(gal, galaxy=self.galaxy, mode=self.mode)
        if 'fat1' in gal.dtype.names:
            igal = gal
        else:
            igal = self.gm2tm(gal)
        fats = np.array([igal['fat1'], igal['fat2'], igal['fat3'], igal['fat4'], igal['fat5']])[:nfat]
        mfats = np.array([igal['mfat1'], igal['mfat2'], igal['mfat3'], igal['mfat4'], igal['mfat5']])[:nfat]
        ind = (mfats > masscut_percent) & (fats > 0)
        # No father
        if jn.howmany(ind, True) == 0:
            # No fathers
            cands = self.treemakers.loadtree(istep-1)
            dt = jn.timeconversion(istep, start='nstep', final='gyr', mode=self.mode, galaxy=self.galaxy) - jn.timeconversion(istep-1, start='nstep', final='gyr', mode=self.mode, galaxy=self.galaxy) # Gyr
            dt = dt*1e9 *365 * 24 * 60 * 60 # sec
            vel = np.array( [gal['vx'], gal['vy'], gal['vz']] ) # km/s
            isnap = self.load_snap(iout)
            ds = vel * dt  *1e5 / isnap.params['unit_l'] # codeunit
            ds = np.sqrt(np.sum(ds**2))
            choose = max(ds, gal['r'])
            progs = jn.cut_sphere(cands, gal['x'], gal['y'],gal['z'], choose*3)
            if len(progs)==0:
                if self.verbose > 2 or self.debug:
                    jn.printtime(ref, f"{prefix}{func} Done")
                return None
            score_match = self.calc_matchrate(gal, progs, agecut=True, mass_weight=True, prefix=f"\t{prefix}{func}")
            progs = progs[score_match > 0]
            if self.verbose > 2 or self.debug:
                jn.printtime(ref, f"{prefix}{func} Done")
            if len(progs)==0:
                return None
            return progs
        # Yes father
        else:
            fats = fats[ind]
            # if self.verbose > 2 or self.debug:
            #     jn.printtime(ref, f"{prefix}{func} Done")
            return self.treemakers.loadgals(istep-1, fats)

    def load_fats(self, gals, masscut_percent=0, nfat=5, prefix="", **kwargs):
        ###### It takes very short time!
        # func = f"[{inspect.stack()[0][3]}]"
        # if self.verbose > 2 or self.debug:
        #     print(f"{prefix}{func} START ({len(gals)} gals)");ref = time.time()
        if 'fat1' in gals[0].dtype.names:
            igals = gals
        else:
            igals = self.gm2tm(gals)
        
        fats = np.vstack((igals['fat1'], igals['fat2'], igals['fat3'], igals['fat4'], igals['fat5']))[:nfat].ravel() # (5, ngal)
        if masscut_percent>0:
            mfats = np.vstack((igals['mfat1'], igals['mfat2'], igals['mfat3'], igals['mfat4'], igals['mfat5']))[:nfat].ravel()
            fats = fats[mfats>masscut_percent]
        # if self.verbose > 2 or self.debug:
        #     jn.printtime(ref, f"{prefix}{func} Done")
        return np.unique( fats[fats>0] )


    def load_fats_of_gals(self, gals, masscut_percent=0, nfat=5, prefix="", **kwargs):
        ###### It takes very short time!
        func = f"[{inspect.stack()[0][3]}]"
        # if self.verbose > 2 or self.debug:
        #     print(f"{prefix}{func} START ({len(gals)} gals)")
        # ref = time.time()
        if isinstance(gals, Iterable):
            # Several galaxies
            if len(gals) > 1:
                # Several galaxies
                assert len(np.unique(len(gals['timestep'])))==1
                iout, istep = jn.ioutistep(gals[0], galaxy=self.galaxy, mode=self.mode)
                fats = self.load_fats(gals, masscut_percent=masscut_percent, nfat=nfat, prefix=f"\t{prefix}{func}")
                # if self.verbose > 2 or self.debug:
                #     jn.printtime(ref, f"{prefix}{func} Done ({len(gals)} gals)")
                return self.treemakers.loadgals(istep-1, fats)
                # ref2 = time.time()
                # for i, igal in enumerate(gals):
                #     jn.printtime(ref2, f"{prefix}{func} --> {i:02d}th");ref2=time.time()
                #     iprogs = self.load_fats_of_gal(igal, masscut_percent=masscut_percent, nfat=nfat, prefix=f"\t{prefix}{func}")
                #     jn.printtime(ref2, f"{prefix}{func} --> {i:02d}th");ref2=time.time()
                #     if progs is None:
                #         progs = np.copy(iprogs)
                #         jn.printtime(ref2, f"{prefix}{func} --> {i:02d}th");ref2=time.time()
                #     else:
                #         progs = np.concatenate((progs, iprogs))
                #         jn.printtime(ref2, f"{prefix}{func} --> {i:02d}th");ref2=time.time()
                #     jn.printtime(ref2, f"{prefix}{func} --> {i:02d}th");ref2=time.time()
                # if progs is not None:
                #     progs = np.unique(progs)
                # if self.verbose > 2 or self.debug:
                #     jn.printtime(ref, f"{prefix}{func} Done ({len(gals)} gals)")
            else:
                # if self.verbose > 2 or self.debug:
                #     jn.printtime(ref, f"{prefix}{func} Done (1 gal)")
                return self.load_fats_of_gal(gals[0], masscut_percent=masscut_percent, nfat=nfat, prefix=f"\t{prefix}{func}")
        else:
            # One galaxy
            # if self.verbose > 2 or self.debug:
            #     jn.printtime(ref, f"{prefix}{func} Done (1 gal)")
            return self.load_fats_of_gal(gals, masscut_percent=masscut_percent, nfat=nfat, prefix=f"\t{prefix}{func}")

    def load_snap(self, iout, prefix="", **kwargs):
        ###### It takes very short time!
        # func = f"[{inspect.stack()[0][3]}]"
        # ref = time.time()
        # if self.debug:
        #     print(f"{prefix}{func} START")
        if iout in self.snaps.keys():
            pass
        else:
            # snap = uri.RamsesSnapshot(self.repo, iout, path_in_repo='snapshots', mode='yzics')
            func = f"[{inspect.stack()[0][3]}]"
            ref = time.time()
            if self.debug:
                print(f"{prefix}{func} START ({iout})")
            self.snaps[iout] = uri.RamsesSnapshot(self.repo, iout, path_in_repo='snapshots', mode='yzics')
            if self.debug:
                jn.printtime(ref, f"{prefix}{func} Done ({iout})")
        return self.snaps[iout]

    def load_star(self, iout, prefix="", all=False, **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        ref = time.time()
        if not iout in self.stars.keys():
            if self.debug:
                print(f"{prefix}{func} START")
            snap = self.load_snap(iout, prefix=f"\t{prefix}{func}")
            if snap.part is None:
                star = snap.get_part()['star']
            else:
                star = snap.part['star']
            dtype = snap.part_dtype
            target_fields = ["vx", "vy", "vz", "m", "epoch", "id"]
            target_idx = np.where(np.isin(np.dtype(dtype).names, target_fields))[0]
            arr = [star.table[field] for field in target_fields]
            dtype = [dtype[idx] for idx in target_idx]
            star.table = np.rec.fromarrays(arr, dtype=dtype)

            arg = np.argsort( np.abs(star["id"]) )
            self.stars[iout] = uri.RamsesSnapshot.Particle(star.table[arg], snap)
            star = None
        self.memory = psutil.Process().memory_info().rss / 2 ** 20 # Bytes to MB
        if self.debug:
            jn.printtime(ref, f"{prefix}{func} Done")
        return self.stars[iout]

    def load_pid_in_gal(self, gal, iout=None, prefix="", **kwargs):
        ###### It takes very short time!
        # func = f"[{inspect.stack()[0][3]}]"
        # ref = time.time()
        # if self.debug:
        #     print(f"{prefix}{func} START")
        if iout is None:
            iout, _ = jn.ioutistep(gal, galaxy=self.galaxy, mode=self.mode)
        # if self.debug:
        #     jn.printtime(ref, f"{prefix}{func} Done")
        return self.gmpids[iout][gal['id']-1]

    def load_allstars(self, iout, prefix="", **kwargs):
        '''NOT USED'''
        func = f"[{inspect.stack()[0][3]}]"
        ref = time.time()
        # if self.verbose > 2 or self.debug:
        #     print(f"{prefix}{func} START")
        snap = self.load_snap(iout, prefix=f"\t{prefix}{func}")
        default_box = np.array([[0, 1], [0, 1], [0, 1]])
        if not np.array_equiv(snap.box, default_box):
            snap.box = default_box
            snap.get_part()
        # if self.debug:
        #     jn.printtime(ref, f"{prefix}{func} Done")
        return self.load_star(iout, prefix=f"\t{prefix}{func}")

    def load_star_boxhalo(self, iout, gal, radii=2, rname='r', prefix="", **kwargs):
        '''NOT USED'''
        func = f"[{inspect.stack()[0][3]}]"
        ref = time.time()
        if self.verbose > 2 or self.debug:
            print(f"{prefix}{func} START")
        snap = self.load_snap(iout, prefix=f"\t{prefix}{func}")
        center = np.stack([gal[key] for key in ["x", "y", "z"]], axis=-1)
        extent = gal[rname]*radii*2
        newbox = snap.get_box(center, extent)
        if inside(snap.box, newbox):
            star = jn.cut_box(self.load_star(iout, prefix=f"\t{prefix}{func}").table,xbox=extent[0], ybox=extent[1], zbox=extent[2])
            if self.verbose > 2 or self.debug:
                jn.printtime(ref, f"{prefix}{func} Done")
            return uri.RamsesSnapshot.Particle(star, snap)
        else:
            # ! WARNING !
            # Below would make cuboid, not cube.
            snap.box = mergebox(snap.box, newbox)
            star = snap.get_part()['star'].table
            arg = np.argsort( np.abs(snap.part["star"]["id"]) )
            if self.verbose > 2 or self.debug:
                jn.printtime(ref, f"{prefix}{func} Done")
            return uri.RamsesSnapshot.Particle(star[arg], snap)

    def append_star(self, iout, star, prefix="", **kwargs):
        '''NOT USED'''
        func = f"[{inspect.stack()[0][3]}]"
        ref = time.time()
        if self.verbose > 2 or self.debug:
            print(f"{prefix}{func} START")
        stars = self.load_star(iout, prefix=f"\t{prefix}{func}")
        table = stars.table
        snap = stars.snap
        ind = ~jn.large_isin(star['id'], table['id'])
        if jn.howmany(ind, True)>0:
            table = np.concatenate((table, star.table[ind]))
            arg = np.argsort( np.abs(table['id']) )
            self.stars[iout] = uri.RamsesSnapshot.Particle(table[arg], snap)
        if self.verbose > 2 or self.debug:
            jn.printtime(ref, f"{prefix}{func} Done")

    def load_parts_in_gal(self, gal, check=False, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        iout, _ = jn.ioutistep(gal, galaxy=self.galaxy, mode=self.mode)
        do = False
        if iout in self.cache_parts.keys():
            if gal['id'] in self.cache_parts[iout]:
                case = 'already'
                self.cache_refers[iout][gal['id']] += 1
                self.cache_count += 1
                return self.cache_parts[iout][gal['id']]
            else:
                do = True
        else:
            self.cache_parts[iout] = {}
            self.cache_refers[iout] = {}
            do = True

        if do:
            ref = time.time()
            if self.verbose > 2 or self.debug:
                print(f"{prefix}{func} START")
            if (self.cache_count == 4000) or (self.memory > 90000):
                self.cache_flush(iout)
            inpid = self.load_pid_in_gal(gal, iout, prefix=f"\t{prefix}{func}")
            star = self.load_star(iout, prefix=f"\t{prefix}{func}")
        
            if len(star['id']) == np.max( np.abs(star['id']) ):
                case = 'all'
                istars = uri.RamsesSnapshot.Particle(star.table[inpid-1], star.snap)
            else:
                case = 'isin'
                isin = jn.large_isin(np.abs(self.stars[iout]['id']), inpid)
                istars = uri.RamsesSnapshot.Particle(star.table[isin], star.snap)
            if check:
                print("[load_parts_in_gal] Let's check")
                print(f"(Gal): nparts={gal['nparts']}   (Part): len={len(istars['id'])}")
                print(f"(Gal): m={gal['m']}   (Part): mtot={np.sum(istars['m','Msol'])}")
                print(f"(Gal): x={gal['x']:.2f}, y={gal['y']:.2f}, z={gal['z']:.2f}, r={gal['r']:.2f}")
                print(f"(Part): minx={np.min(istars['x'])} maxx={np.max(istars['x'])}")
                print(f"(Part): miny={np.min(istars['y'])} maxy={np.max(istars['y'])}")
                print(f"(Part): minz={np.min(istars['z'])} maxz={np.max(istars['z'])}")
            inpid = None
            star = None
            # already_keys = list( self.cache_parts[iout].keys() )
            # if len(already_keys) > 100:
            #     for ikey in already_keys[:10]:
            #         self.cache_parts[iout][ikey] = None
            #         del self.cache_parts[iout][ikey]
            self.cache_count += 1
            self.cache_refers[iout][gal['id']] = 1
            self.cache_parts[iout][gal['id']] = istars
            istars = None
        if self.verbose > 2 or self.debug:
            jn.printtime(ref, f"{prefix}{func}<{case}> Done")
        return self.cache_parts[iout][gal['id']]
    
    def cache_flush(self, iout, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        ref = time.time()
        if self.verbose > 2 or self.debug:
            print(f"{prefix}{func} START")
        keys = list(self.cache_refers[iout].keys())
        for key in keys:
            # print("### refer count: ",key, self.cache_refers[iout][key])
            self.cache_refers[iout][key] -= 1
            if self.cache_refers[iout][key] == 0:
                self.cache_parts[iout][key] = None
                del self.cache_parts[iout][key]
                del self.cache_refers[iout][key]
        self.cache_count = 0
        gc.collect()
        if self.verbose > 2 or self.debug:
            jn.printtime(ref, f"{prefix}{func} Done")


    def calc_matchrate(self, target, gals,agecut=True, mass_weight=False, return_pid_each=False, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        if self.verbose > 2 or self.debug:
            print(f"{prefix}{func} START ({len(gals)} gals)")
        ref = time.time()
        tout, tstep = jn.ioutistep(target, galaxy=self.galaxy, mode=self.mode)
        tgyr = jn.timeconversion(tout, final='gyr', mode=self.mode, galaxy=self.galaxy)
        if agecut or mass_weight:
            stars = self.load_parts_in_gal(target, prefix=f"\t{prefix}{func}")
            ref_page = stars['age', 'Gyr']
            ref_pm = stars['m', 'Msol']
            stars = None

        ref_pid = np.copy(self.load_pid_in_gal(target, tout, prefix=f"\t{prefix}{func}"))
        arr = np.zeros(len(gals))
        pids = [0]*len(gals)
        # indices = [0]*len(gals)
        last50 = np.zeros(50)*np.nan
        minout = 0
        for i, igal in enumerate(gals):
            ref2 = time.time()
            iout, istep = jn.ioutistep(igal, galaxy=self.galaxy, mode=self.mode)
            if iout <= minout:
                pass
            else:
                igyr = jn.timeconversion(iout, final='gyr', mode=self.mode, galaxy=self.galaxy)
                ind = ref_pid >= 0
                if agecut:
                    ind = ref_page >= (tgyr - igyr)
                if mass_weight:
                    pm = ref_pm[ind]
                
                if not True in ind:
                    minout = iout
                    print("No particles at this epoch")
                    pass
                else:
                    ref_pid_at_ith = ref_pid[ind]
                    inst = jn.large_isin(ref_pid_at_ith, self.load_pid_in_gal(igal, iout=iout, prefix=f"\t{prefix}{func}"))

                    if mass_weight:
                        arr[i] = np.sum(pm[inst])/np.sum(pm)
                    else:
                        arr[i] = jn.howmany(inst, True)/len(ref_pid_at_ith)
                    
                    if return_pid_each:
                        pids[i] = np.copy(ref_pid_at_ith[inst])
                        # indices[i] = np.copy(inst)

                if self.verbose > 2 and not self.debug:
                    last50 = jn.timeestimator_inst(ref2, i+1, len(gals), message='[matchrate]', last100=last50)
                    if i%10 == 9:
                        jn.clear_output(wait=True)
        if self.verbose > 1 or self.debug:
            jn.printtime(ref, f"{prefix}{func} Done ({len(gals)} gals)")
        if return_pid_each:
            return arr, pids
        return arr
    
    def calc_bulkmotion(self, gal, star=None, use_pid=None, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        ref = time.time()
        # if self.verbose > 1 or self.debug:
        #     print(f"{prefix}{func} START")
        if star is None:
            star = self.load_parts_in_gal(gal, prefix=f"\t{prefix}{func}")
        ind = star['m']>0
        if use_pid is not None:
            ind = jn.large_isin(np.abs(star['id']), use_pid)
        vx = np.average(star['vx', 'km/s'][ind], weights=star['m'][ind])
        vy = np.average(star['vy', 'km/s'][ind], weights=star['m'][ind])
        vz = np.average(star['vz', 'km/s'][ind], weights=star['m'][ind])
        star = None
        # if self.verbose > 1 or self.debug:
        #     jn.printtime(ref, f"{prefix}{func} Done")
        return np.array([vx, vy, vz])

    def calc_velocity_offset(self, target, gals, pids=None, agecut=True, mass_weight=False, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        if self.verbose > 2 or self.debug:
            print(f"{prefix}{func} START ({len(gals)} gals)")
        ref = time.time()
        if pids is None:
            _, pids = self.calc_matchrate(target, gals, agecut=agecut, mass_weight=mass_weight, return_pid_each=True, prefix=f"\t{prefix}{func}")
        results = np.zeros(len(gals))
        for i, igal in enumerate(gals):
            totv = np.array([igal['vx'], igal['vy'], igal['vz']])
            results[i] = np.inf
            if isinstance(pids[i], Iterable):
                if len(pids[i])>0:
                    inv = self.calc_bulkmotion(igal, star=self.load_parts_in_gal(igal, prefix=f"\t{prefix}{func}"), use_pid=pids[i], prefix=f"\t{prefix}{func}")
                    dv = totv - inv
                    results[i] = np.sqrt( np.sum(dv**2) )
        if self.verbose > 1 or self.debug:
            jn.printtime(ref, f"{prefix}{func} Done ({len(gals)} gals)")
        return results
    
    def scoring(self, target, gals, agecut=True, mass_weight=False, prefix="", **kwargs):
        func = f"[{inspect.stack()[0][3]}]"
        if self.verbose > 2 or self.debug:
            print(f"{prefix}{func} START ({len(gals)} gals)")
        ref = time.time()
        score_match, pids = self.calc_matchrate(target, gals, agecut=agecut, mass_weight=mass_weight, 
        return_pid_each=True, prefix=f"\t{prefix}{func}") ###
        score_veloff = self.calc_velocity_offset(target, gals, pids=pids, agecut=agecut, mass_weight=mass_weight, prefix=f"\t{prefix}{func}")
        score_mvir = np.abs( np.log10(gals['mvir']/target['mvir']) )
        score_rvir = np.abs( np.log10(gals['rvir']/target['rvir']) )

        # 0 is the best
        rank = np.argsort(1-score_match) # High match
        rank_match = np.argsort(rank)
        rank = np.argsort(score_veloff) # Low velocity offset
        rank_veloff = np.argsort(rank)
        rank = np.argsort(score_mvir) # Low mvir difference
        rank_mvir = np.argsort(rank)
        rank = np.argsort(score_rvir) # Low rvir difference
        rank_rvir = np.argsort(rank)
        rank = rank_match + rank_veloff + rank_mvir + rank_rvir
        if self.verbose > 1 or self.debug:
            jn.printtime(ref, f"{prefix}{func} Done ({len(gals)} gals)")
        return rank # 0 is the best



#################################################
#################################################
#################################################
#################################################
#################################################



# modes = ['y07206', 'y10002', 'y35663', 'y36413', 'y36415', 'y39990']
# ngal: [34, 91, 151, 127, 107, 601]
modes = ['y07206', 'y36415', 'y36413', 'y10002']#, 'y39990']

for mode in modes:
    # Configuration
    prefix = f"[{mode}]"
    # dm_nout, dm_nstep, dm_zred, dm_aexp, dm_gyr = jn.pklload(f'/storage6/jeon/data/{mode}/dm_{mode}_nout_nstep_zred_aexp_gyr.pickle')
    nout, nstep, zred, aexp, gyr = jn.pklload(f'/storage6/jeon/data/{mode}/{mode}_nout_nstep_zred_aexp_gyr.pickle')
    repo = f'/storage3/Clusters/{mode[1:]}'
    current_time()
    print(f"{prefix} {nout[-1]} ~ {nout[0]}")



    # TreeMaker
    print(f"\n{prefix} TreeMaker load..."); ref = time.time()
    tm = jn.pklload(f"/storage6/jeon/data/{mode}/tm/{mode}_TreeMaker.pickle")
    galtm = jn.treemaker(galaxy=True, tm=tm, mode=mode)
    tm = None
    jn.printtime(ref, f"{prefix} TreeMaker load done")



    # GalaxyMaker
    print(f"\n{prefix} GalaxyMaker load..."); ref = time.time()
    gmpids = {}
    cumpartss = {}
    galaxymakers = {}
    for i, iout in enumerate(nout):
        snap = uri.RamsesSnapshot(repo, iout, path_in_repo='snapshots', mode='yzics')
        gm, gmpid = uhmi.HaloMaker.load(snap, galaxy=True, load_parts=True)
        cumparts = np.insert(np.cumsum(gm["nparts"]), 0, 0)
        # gmpid = [inst] + [gmpid[ cumparts[i]:cumparts[i+1] ] for i in range(len(gm))]
        inst = np.copy(gmpid)
        gmpid = [inst[ cumparts[i]:cumparts[i+1] ] for i in range(len(gm))]
        galaxymakers[iout] = gm
        gmpids[iout] = gmpid
        cumpartss[iout] = cumparts
        snap.clear()
    dict_done = {"galaxymakers":galaxymakers, "gmpids":gmpids, "cumpartss":cumpartss}
    jn.printtime(ref, f"{prefix} GalaxyMaker load done")



    # Target galaxies
    print(f"\n{prefix} target galaxies load..."); ref = time.time()
    iout = 187
    snap_now = uri.RamsesSnapshot(repo, iout, path_in_repo='snapshots', mode='yzics' )
    gals_now = uhmi.HaloMaker.load(snap_now, galaxy=True)
    # gals_now = gals_now[gals_now['m'] > 1e10]
    jn.printtime(ref, f"{prefix} {len(gals_now)} gals load done")



    # Young Tree setting
    print(f"\n{prefix} YoungTree setting..."); ref = time.time()
    yt = youngtree(galtm, gals_now, lengstep=3, mode=mode, verbose=0, debug=False)
    yt.load_galaxymaker(dict_done=deepcopy(dict_done))
    gc.collect()
    jn.printtime(ref, f"{prefix} YT setting done")
    


    # Young Tree Run
    print(f"\n{prefix} YoungTree Run start..."); ref = time.time()
    yt.Build_Tree(maxngal=10, agecut=True, mass_weight=True)
    jn.printtime(ref, f"{prefix} YT Done")
    


    # Save tree
    print(f"\n{prefix} YoungTree Run start..."); ref = time.time()
    for target in gals_now:
        branch = yt.return_branch(target)
        jn.pklsave(branch, f"youngtree/YT_{mode}_{target['id']:05d}_branch.pickle")
    jn.printtime(ref, f"{prefix} YT save Done")