import time
import psutil
import logging
import pickle
import os
import numpy as np
import numba as nb
from collections.abc import Iterable


class timer():
    __slots__ = ['ref', 'units', 'corr', 'unit', 'text', 'verbose', 'debugger']
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

def dprint_(msg, debugger):
    debugger.debug(msg)

def mode2repo(mode):
    if mode[0] == 'h':
        rurmode = 'hagn'
        repo = f"/storage4/Horizon_AGN"
    elif mode[0] == 'y':
        rurmode = 'yzics'
        repo = f"/storage3/Clusters/{mode[1:]}"
    elif mode == 'nh':
        rurmode = 'nh'
        repo = "/storage6/NewHorizon"
    else:
        raise ValueError(f"{mode} is currently not supported!")
    return repo, rurmode

def load_nout(mode='hagn', galaxy=True):
    repo,_ = mode2repo(mode)
    if galaxy:
        path = f"{repo}/galaxy"
    else:
        path = f"{repo}/halo"

    fnames = np.array(os.listdir(path))
    if mode == 'nh':
        ind = [True if "tree_bricks0" in file else False for file in fnames]
    else:
        ind = [True if "tree_bricks" in file else False for file in fnames]
    fnames = fnames[ind]
    fnames = -np.sort( -np.array([int(file[11:]) for file in fnames]) )
    return fnames

def load_nstep(mode='hagn', galaxy=True, nout=None):
    if nout is None:
        nout = load_nout(mode=mode, galaxy=galaxy)
    nstep = np.arange(len(nout))[::-1]+1
    return nstep

def out2step(iout, galaxy=True, mode='hagn', nout=None, nstep=None):
    if nout is None:
        nout = load_nout(mode=mode, galaxy=galaxy)
    if nstep is None:
        nstep = load_nstep(mode=mode, galaxy=galaxy, nout=nout)
    arg = np.argwhere(iout==nout)[0][0]
    return nstep[arg]

def step2out(istep, galaxy=True, mode='hagn', nout=None, nstep=None):
    if nout is None:
        nout = load_nout(mode=mode, galaxy=galaxy)
    if nstep is None:
        nstep = load_nstep(mode=mode, galaxy=galaxy, nout=nout)
    arg = np.argwhere(istep==nstep)[0][0]
    return nout[arg]

def ioutistep(gal, galaxy=True, mode='hagn', nout=None, nstep=None):
    if 'nparts' in gal.dtype.names:
        iout = gal['timestep']
        istep = out2step(iout, galaxy=galaxy, mode=mode, nout=nout, nstep=nstep)
    else:
        istep = gal['timestep']
        iout = step2out(istep,galaxy=galaxy, mode=mode, nout=nout, nstep=nstep)
    return iout, istep


def printgal(gal, mode='hagn', nout=None, nstep=None, isprint=True):
    if 'nparts' in gal.dtype.names:
        iout = gal['timestep']
        istep = out2step(iout, galaxy=True, mode=mode, nout=nout, nstep=nstep)
        made = 'GalaxyMaker'
    else:
        istep = gal['timestep']
        iout = step2out(istep,galaxy=True, mode=mode, nout=nout, nstep=nstep)
        made = 'TreeMaker'
    if isprint:
        print(f"[{made}: {mode}] ID={gal['id']}, iout(istep)={iout}({istep}), logM={np.log10(gal['m']):.2f}")
    return f"[{made}: {mode}] ID={gal['id']}, iout(istep)={iout}({istep}), logM={np.log10(gal['m']):.2f}"


def MB():
    return psutil.Process().memory_info().rss / 2 ** 20
def GB():
    return psutil.Process().memory_info().rss / 2 ** 30

def pklsave(data,fname, overwrite=False):
    '''
    pklsave(array, 'repo/fname.pickle', overwrite=False)
    '''
    if overwrite == False and os.path.isfile(fname):
        raise ValueError(f"{fname} already exist!!")
    else:
        with open(fname, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pklload(fname):
    '''
    array = pklload('path/fname.pickle')
    '''
    with open(fname, 'rb') as handle:
        try:
            arr = pickle.load(handle)
        except EOFError:
            arr = pickle.load(handle.read())
            # arr = {}
            # unpickler = pickle.Unpickler(handle)
            # # if file is not empty scores will be equal
            # # to the value unpickled
            # arr = unpickler.load()
    return arr

def distance3d(x,y,z, xs,ys,zs):
    return np.sqrt((x-xs)**2 + (y-ys)**2 + (z-zs)**2)

def cut_sphere(targets, cx, cy, cz, radius, return_index=False, both_sphere=False):
    '''
    Return targets which is closer than radius of (cx,cy,cz)\n
    ##################################################\n
    Parameters
    ----------
    targets :       They must have 'x', 'y', 'z'\n
    cx,cy,cz :      xyz position of center\n
    radius :        radius from center\n
    return_index :  Default is False\n
    
    Returns
    -------
    targets[ind]
    
    Examples
    --------
    >>> from jeonpkg import jeon as jn
    >>> cx, cy, cz, cr = halo['x'], halo['y'], halo['z'], halo['rvir']
    >>> gals = jn.cut_sphere(gals, cx,cy,cz,cr)
    '''
    # region
    indx = targets['x'] - cx
    indx[indx>0.5] = 1 - indx[indx>0.5]
    indy = targets['y'] - cy
    indy[indy>0.5] = 1 - indy[indy>0.5]
    indz = targets['z'] - cz
    indz[indz>0.5] = 1 - indz[indz>0.5]
    if both_sphere:
        radius = radius + targets['r']
    ind = indx ** 2 + indy ** 2 + indz ** 2 <= radius ** 2
    if return_index is False:
        return targets[ind]
    else:
        return targets[ind], ind
    # endregion

def howmany(target, vals):
    '''
    How many vals in target
    
    Examples
    --------
    >>> target = [1,1,2,3,7,8,9,9]
    >>> vals = [1,3,5,7,9]
    >>> howmany(target, vals)
    6
    '''
    if isinstance(vals, Iterable):
        return np.count_nonzero(np.isin(target, vals))
    else:
        if isinstance(vals, bool) and vals:
            return np.count_nonzero(target)
        else:
            return np.count_nonzero(target==vals)

def printtime(ref, msg, unit='sec', return_elapse=False):
    units = {"sec":1, "min":1/60, "hr":1/3600, "ms":1000}
    elapse = time.time()-ref
    print(f"{msg} ({elapse*units[unit]:.3f} {unit} elapsed)")
    if return_elapse:
        return elapse*units[unit]

def rms(*args):
    inst = 0
    for arg in args:
        inst += arg**2
    return np.sqrt(inst)

@nb.jit(parallel=True)
def large_isin(a, b):
    '''
    [numba] Return part of a which is in b
    
    Examples
    --------
    >>> a = [1,2,3,4,5,6]
    >>> b = [2,4,6,8]
    >>> large_isin(a,b)
    [False, True, False, True, False, True]
    '''
    # region
    shape = a.shape
    a = a.ravel()
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            result[i] = True
    return result.reshape(shape)
    # endregion

def atleast_isin(a, b):
    '''
    Return True if any element of a is in b
    '''
    return not set(a).isdisjoint(b)