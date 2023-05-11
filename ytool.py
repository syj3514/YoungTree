import numpy as np
import os
from collections.abc import Iterable
import numba as nb
import importlib
import pickle
import psutil
from numba.core.config import NUMBA_NUM_THREADS
from numba import config
import traceback
config.CPU_VECTORIZE=True
config.CPU_CACHE_MAXSIZE='524288k'
ncpu = NUMBA_NUM_THREADS

yess = ["Y","y","yes", "YES", "Yes"]
####################################################################################################################
# Classes
####################################################################################################################
class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


####################################################################################################################
# File I/O
####################################################################################################################
def mode2repo(mode):
    dp = True
    if mode[0] == 'h':
        rurmode = 'hagn'
        repo = f"/storage4/Horizon_AGN"
        dp = False
    elif mode[0] == 'y':
        rurmode = 'yzics'
        repo = f"/storage3/Clusters/{mode[1:]}"
        dp = True
    elif mode == 'nh':
        rurmode = 'nh'
        repo = "/storage6/NewHorizon"
    elif mode == 'nh2':
        rurmode = 'y4'
        repo = "/storage7/NH2"
    elif mode == 'nc':
        rurmode = 'nc'
        repo = "/storage7/NewCluster2"
    elif mode == 'fornax':
        rurmode = 'fornax'
        repo = '/storage5/FORNAX/KISTI_OUTPUT/l10006'
    elif mode == 'custom':
        from custom_mode import repo, rurmode, dp
    else:
        raise ValueError(f"{mode} is currently not supported!")
    return repo, rurmode, dp

def make_params_dict(fname:str) -> None:
    arg = fname.find('.')
    param_name = fname[:arg]
    params = importlib.import_module(param_name)
    p = {}
    for key in params.__dict__.keys():
        p[key] = params.__dict__[key]
    mode = p['mode']
    dp = True if p['galaxy'] else False
    if mode[0] == 'h':
        rurmode = 'hagn'; repo = f"/storage4/Horizon_AGN"; dp = False
    elif mode[0] == 'y':
        rurmode = 'yzics'; repo = f"/storage3/Clusters/{mode[1:]}"; dp = True
    elif mode == 'nh':
        rurmode = 'nh'; repo = "/storage6/NewHorizon"; dp = True
    elif mode == 'nh2':
        rurmode = 'y4'; repo = "/storage7/NH2"
    elif mode == 'nc':
        rurmode = 'nc'; repo = "/storage7/NewCluster2"; dp = True
    elif mode == 'fornax':
        rurmode = 'fornax'; repo = '/storage5/FORNAX/KISTI_OUTPUT/l10006'
    else:
        raise ValueError(f"{mode} is currently not supported!")
    p['rurmode'] = rurmode; p['repo'] = repo; p['dp'] = dp
    nout = load_nout(p['repo'], galaxy=p['galaxy'])
    nstep = load_nstep(nout)
    p['nout'] = nout; p['nstep'] = nstep

    p = DotDict(p)
    return p

def load_nout(repo, galaxy=True, path_in_repo=None):
    if path_in_repo is None:
        path_in_repo='galaxy' if galaxy else 'halo'
    path = f"{repo}/{path_in_repo}"
    fnames = os.listdir(path)
    fnames = [file for file in fnames if file.startswith("tree_bricks")]
    nout = np.array([int(file[-5:]) for file in fnames])
    return np.sort(nout)[::-1]

def load_nstep(nout):
    nstep = np.arange(len(nout))[::-1]+1
    return nstep

def pklsave(data,fname, overwrite=False):
    '''
    pklsave(array, 'repo/fname.pickle', overwrite=False)
    '''
    if os.path.isfile(fname):
        if overwrite == False:
            raise FileExistsError(f"{fname} already exist!!")
        else:
            with open(f"{fname}.pkl", 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.remove(fname)
            os.rename(f"{fname}.pkl", fname)
    else:
        with open(f"{fname}", 'wb') as handle:
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
####################################################################################################################
# Connection
####################################################################################################################
def gal2id(gal):
    return gal['timestep']*100000 + gal['id']
def id2gal(iid):
    return iid//100000, iid%100000

def gethalo(*args, halos=None, complete=False):
    if isinstance(args, tuple):
        if len(args)==2:
            iout, iid = args
        elif len(args)==1:
            iout = args[0]//100000; iid = args[0]%100000
        else:
            raise TypeError(f"{type(args)} is not understood")
    else:
        raise TypeError(f"{type(args)} is not understood")
    if(complete):
        return halos[iout][iid-1]
    else:
        arg = np.argwhere(halos[iout]['id']==iid)[0][0]
        return halos[iout][arg]


def out2step(iout, nout, nstep):
    try:
        arg = np.argwhere(iout==nout)[0][0]
        return nstep[arg]
    except IndexError:
        print(f"\n!!! {iout} is not in nout({np.min(nout)}~{np.max(nout)}) !!!\n")
        traceback.print_stack()

def step2out(istep, nout, nstep):
    try:
        arg = np.argwhere(istep==nstep)[0][0]
        return nout[arg]
    except IndexError:
        print(f"\n!!! {istep} is not in nstep({np.min(nstep)}~{np.max(nstep)}) !!!\n")
        traceback.print_stack()

def maxdesc(halo, all=True, offset=1, nout=None, nstep=None):
    if len(halo['desc_score'])<1:
        return 0, -1
    arg = np.argmax(halo['desc_score'])
    if not all:
        istep = out2step(halo['timestep'], nout, nstep)
        iout = step2out(istep+offset, nout, nstep)
        ind = (halo['desc']//100000 == iout)
        if not True in ind:
            return 0, -1
        arg = np.argmax(halo['desc_score'] * ind)
    return halo['desc'][arg], halo['desc_score'][arg]
def maxprog(halo, all=True, offset=1, nout=None, nstep=None):
    if len(halo['prog_score'])<1:
        return 0, -1
    arg = np.argmax(halo['prog_score'])
    if not all:
        istep = out2step(halo['timestep'], nout, nstep)
        iout = step2out(istep-offset, nout, nstep)
        ind = (halo['prog']//100000 == iout)
        if not True in ind:
            return 0, -1
        arg = np.argmax(halo['prog_score'] * ind)
    return halo['prog'][arg], halo['prog_score'][arg]



####################################################################################################################
# Calculation
####################################################################################################################
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

def distance(*args):
    if len(args)==2:
        return distance3d(args[0]['x'],args[0]['y'],args[0]['z'],args[1]['x'],args[1]['y'],args[1]['z'])
    elif len(args)==4:
        if(hasattr(args[0], 'x')):
            return distance3d(args[1], args[2], args[3], args[0]['x'],args[0]['y'],args[0]['z'])
        elif(hasattr(args[-1], 'x')):
            return distance3d(args[0], args[1], args[2], args[-1]['x'],args[-1]['y'],args[-1]['z'])
        else:
            raise ValueError("Check arguments!")
    elif len(args)==6:
        return distance3d(*args)
    else:
        raise ValueError("Check arguments!")
def distance3d(x,y,z, xs,ys,zs):
    return np.sqrt((x-xs)**2 + (y-ys)**2 + (z-zs)**2)

def rms(*args):
    inst = 0
    for arg in args:
        inst += arg**2
    return np.sqrt(inst)

def atleast_isin(a, b):
    '''
    Return True if any element of a is in b
    '''
    return not set(a).isdisjoint(b)

def MB()->float:
    return psutil.Process().memory_info().rss / 2 ** 20
def GB()->float:
    return psutil.Process().memory_info().rss / 2 ** 30

def timeconv(sec):
    units = {'ms':1/1000, 'sec':1, 'min':60, 'hr':3600, 'day':24*3600}
    key = 'sec'
    if sec < 1:
        key = 'ms'
    if sec>60:
        key = 'min'
        if sec > 3600:
            key = 'hr'
            if sec > 24*3600:
                key = 'day'
    return key, sec / units[key]

####################################################################################################################
# Numba
####################################################################################################################
@nb.njit(fastmath=True)
def nbsum(a:np.ndarray,b:np.ndarray)->float:
    n:int = len(a)
    s:float = 0.
    for i in nb.prange(n):
        s += a[i]*b[i]
    return s

@nb.njit(fastmath=True)
def nbnorm(l:np.ndarray)->float:
    s = 0.
    for i in range(l.shape[0]):
        s += l[i]**2
    return np.sqrt(s)

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
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            result[i] = True
    return result

@nb.jit(parallel=True)
def large_isind(a, b):
    '''
    [numba] Return part of a which is in b
    
    Examples
    --------
    >>> a = [1,2,3,4,5,6]
    >>> b = [2,4,6,8]
    >>> large_isin(a,b)
    [False, True, False, True, False, True]
    '''
    n = len(a)
    result = np.full(n, False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            result[i] = True
    return np.where(result)[0]

@nb.jit(fastmath=True)
def atleast_numba(a, b):
    '''
    Return True if any element of a is in b
    '''
    n = len(a)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            return True
    # return result.reshape(shape)

@nb.jit(fastmath=True, parallel=True, nopython=True)
def atleast_numba_para(aa, b):
    '''
    Return True if any element of a is in b
    '''
    # nn = len(aa) # <- Fall-back from the nopython compilation path to the object mode compilation path has been detected, this is deprecated behaviour.
    nn = len(aa) # <- Function "atleast_numba_para" was compiled in object mode without forceobj=True, but has lifted loops.
    results = np.full(nn, False)
    
    # for j in nb.prange(nn): <--- Fall-back from the nopython compilation path to the object mode compilation path has been detected, this is deprecated behaviour.
    # for j in nb.prange(nn): <--- Function "atleast_numba_para" was compiled in object mode without forceobj=True.
    # for j in nb.prange(nn): <--- Compilation is falling back to object mode WITHOUT looplifting enabled because Function "atleast_numba_para" failed type inference due to: non-precise type pyobject
    for j in nb.prange(nn): # <--- Compilation is falling back to object mode WITHOUT looplifting enabled because Function "atleast_numba_para" failed type inference due to: Cannot determine Numba type of <class 'numba.core.dispatcher.LiftedLoop'>
        a = aa[j]
        n = len(a)
        set_b = set(b)
        for i in nb.prange(n):
            if a[i] in set_b:
                results[j] = True
                break
    return results