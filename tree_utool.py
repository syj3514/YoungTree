import time
import psutil
import logging
import pickle
import os
import numpy as np
import numba as nb
from collections.abc import Iterable

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

class timer():
    __slots__ = ['ref', 'units', 'corr', 'unit', 'text', 'verbose', 'debugger', 'level']
    def __init__(self, unit="sec",text="", verbose=2, debugger=None, level='info'):
        self.ref = time.time()
        self.units = {"ms":1/1000, "sec":1, "min":60, "hr":3600}
        self.corr = self.units[unit]
        self.unit = unit
        self.text = text
        self.verbose=verbose
        self.debugger=debugger
        self.level = level
        
        if self.verbose>0:
            print(f"{text} START")
        if self.debugger is not None:
            if self.level == 'info':
                self.debugger.info(f"{text} START")
            else:
                self.debugger.debug(f"{text} START")
    
    def done(self, add=None):
        elapse = time.time()-self.ref
        if add is not None:
            self.text = f"{self.text} {add}"
        if self.verbose>0:
            print(f"{self.text} Done ({elapse/self.corr:.3f} {self.unit})")
        if self.debugger is not None:
            if self.level == 'info':
                self.debugger.info(f"{self.text} Done ({elapse/self.corr:.3f} {self.unit})")
            else:
                self.debugger.debug(f"{self.text} Done ({elapse/self.corr:.3f} {self.unit})")

def make_logname(mode, iout, dirname='./log', logprefix=None, overwrite=False):
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    if not os.path.isdir(dirname+f"/{mode}"):
        os.mkdir(dirname+f"/{mode}")
    if logprefix is None:
        logprefix = f"output_"
    if logprefix[-1] != "_":
        logprefix = f"{logprefix}_"

    if iout<0:
        fname = f"{dirname}/{mode}/{logprefix}{mode}_ini.log"
    else:
        fname = f"{dirname}/{mode}/{logprefix}{mode}_{iout:05d}.log"
    if os.path.isfile(fname) and (not overwrite):
        num = 1
        while os.path.isfile(fname):
            if iout<0:
                fname = f"{dirname}/{mode}/{logprefix}{mode}_ini_{num}.log"
            else:
                fname = f"{dirname}/{mode}/{logprefix}{mode}_{iout:05d}_{num}.log"
            num += 1
    return fname


from logging.handlers import RotatingFileHandler
def custom_debugger(fname, detail=True):
    logger_file_handler = RotatingFileHandler(fname, mode='a')
    
    if detail:
        logger_file_handler.setLevel(logging.DEBUG)
    else:
        logger_file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter(u'%(asctime)s [%(levelname)8s] %(message)s')
    logger_file_handler.setFormatter(formatter)

    logging.captureWarnings(True)

    root_logger = logging.getLogger(fname)
    warnings_logger = logging.getLogger("py.warnings")
    root_logger.handlers = []
    warnings_logger.handlers = []
    root_logger.addHandler(logger_file_handler)
    warnings_logger.addHandler(logger_file_handler)
    root_logger.setLevel(logging.DEBUG)
    root_logger.info("Debug Start")
    root_logger.propagate=False
    return root_logger

def dprint_(msg, debugger):
    debugger.debug(msg)

def mode2repo(mode):
    dp = True
    if mode[0] == 'h':
        rurmode = 'hagn'
        repo = f"/storage4/Horizon_AGN"
        dp = False
    elif mode[0] == 'y':
        rurmode = 'yzics'
        repo = f"/storage3/Clusters/{mode[1:]}"
        dp = False
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
    else:
        raise ValueError(f"{mode} is currently not supported!")
    return repo, rurmode, dp

def load_nout(mode='hagn', galaxy=True, double_check=True, useptree=False):
    repo,_,_ = mode2repo(mode)
    if galaxy:
        path = f"{repo}/galaxy"
    else:
        path = f"{repo}/halo/DM"

    # From GalaxyMaker
    fnames = np.array(os.listdir(path))
    ind = [True if "tree_bricks" in file else False for file in fnames]
    fnames = fnames[ind]
    lengs = np.array([len(file)-11 for file in fnames])
    maxleng = np.max(lengs)
    ind = lengs >= maxleng
    fnames = fnames[ind]
    fnames = -np.sort( -np.array([int(file[11:]) for file in fnames]) )

    if double_check:
        # From output
        f2 = np.array(os.listdir(repo+"/snapshots"))
        ind = [True if ("output_" in file)&(len(file)<13) else False for file in f2]
        f2 = f2[ind]
        f2 = np.array([int(file[-5:]) for file in f2])
        ind = np.isin(fnames, f2)
        fnames = fnames[ind]

        if (os.path.isdir(f"{repo}/ptree")) & (useptree):
            # From ptree
            fout = os.listdir(f"{repo}/ptree")
            ind = [True if f"ptree_{fnum:05d}.pkl" in fout else False for fnum in fnames]
            return fnames[ind]
        else:
            return fnames
    else:
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

@nb.njit(fastmath=True)
def nbnorm(l):
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
    # region
    # shape = a.shape
    # a = a.ravel()
    n = len(a)
    # result = np.zeros((n,), dtype=bool)
    result = np.full(n, False)
    # result = np.full((n,), False)
    set_b = set(b)
    for i in nb.prange(n):
        if a[i] in set_b:
            result[i] = True
    return result
    # return result.reshape(shape)
    # endregion

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

def atleast_isin(a, b):
    '''
    Return True if any element of a is in b
    '''
    return not set(a).isdisjoint(b)

from scipy.io import FortranFile
def read_galaxymaker(repo, brick,boxsize_physical, ids=None, dp=False):
    int_ = np.int32
    real_ = np.float32
    if dp:
        real_ = np.float64
    full_path = repo+brick

    dtype = [("nparts", int_), ("id", int_), ("timestep", int_), ("m", real_), ("x", real_), ("y", real_), ("z", real_), ("vx", real_), ("vy", real_), ("vz", real_), ("r", real_), ("rvir", real_), ("mvir", real_), ("member", object)]

    with FortranFile(full_path, mode='r') as f:
        skipread_i(f, 5, dtype=int_)
        nhalos, nsubs = f.read_ints(int_)
        nhalos = nhalos + nsubs
        if ids is None:
            ids = np.arange(nhalos)+1

        count = 0
        for i in range(nhalos):
            nparts, =  f.read_ints(int_) #
            gmpids = f.read_ints(int_) #
            galid, = f.read_ints(int_) #
            if galid in ids:
                timestep, = f.read_ints(int_) #
                skipread_i(f, 1, dtype=int_)
                m, = f.read_reals(real_) #
                x,y,z = f.read_reals(real_) #
                vx,vy,vz = f.read_reals(real_) #
                skipread_f(f, 1, dtype=real_)
                r,_,_,_ = f.read_reals(real_) #
                skipread_f(f, 3, dtype=real_)
                rvir, mvir, _, _ = f.read_reals(real_)
                skipread_f(f, 1, dtype=real_)
                skipread_i(f, 1, dtype=int_)
                skipread_f(f, 2, dtype=real_)

                arr = np.array((nparts, galid, timestep, m, x, y, z, vx, vy, vz, r, rvir, mvir, gmpids), dtype=dtype)
                if count==0:
                    data = arr
                    count += 1
                else:
                    data = np.hstack((data, arr))
            else:
                skipread_i(f, 2, dtype=int_)
                skipread_f(f, 10, dtype=real_)
                skipread_i(f, 1, dtype=int_)
                skipread_i(f, 2, dtype=int_)
        
        mass_unit = 1E11
        data['m'] *= mass_unit
        data['mvir'] *= mass_unit
        data['x'] = data['x'] / boxsize_physical + 0.5
        data['y'] = data['y'] / boxsize_physical + 0.5
        data['z'] = data['z'] / boxsize_physical + 0.5
        data['rvir'] /= boxsize_physical
        data['r'] /= boxsize_physical
        return data



def skipread_i(f, n, dtype=np.int32):
    for _ in range(n):
        f.read_ints(dtype)
def skipread_f(f, n, dtype=np.float64):
    for _ in range(n):
        f.read_reals(dtype)