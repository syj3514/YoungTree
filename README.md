# YoungTree

Seyoung Jeon (syj3514@yonsei.ac.kr)
Department of Astronomy and Yonsei University Observatory, Yonsei University


**YoungTree** is a Python toolkit for building particle–based merger trees for galaxies or dark matter halos in RAMSES cosmological simulations.  
It is designed to work with *GalaxyMaker* / *HaloMaker* catalogues and supports multiple projects such as Horizon-AGN, NewHorizon, NewCluster, and related zoom-in runs.
Technically, it needs the RUR library (https://github.com/sanhancluster/rur), and simulations run with RAMSES-yOMP (https://github.com/sanhancluster/RAMSES-yOMP.git).

The code links objects across snapshots using shared particle IDs, phase–space–weighted scores, and a configurable time window, and then post-processes these links into a consistent set of main-branch (father/son) relations and long-lived branches.

---

## Main features

- **Particle-based linking**
  - Uses member particle IDs from GalaxyMaker/HaloMaker catalogues.
  - Supports both galaxy-based (`galaxy=True`, star particles) and halo-based (`galaxy=False`, dark matter particles) trees.

- **Multiple simulation backends**
  - Built for RAMSES simulations read via the RUR interface.
  - Current `mode` options include:
    - `h*` – Horizon-AGN family
    - `y*` – YZiCS / cluster runs
    - `nh` – NewHorizon
    - `nh2` – NewHorizon2
    - `nc` – NewCluster
    - `fornax` – Fornax boxes
    - `custom` – user-defined repository and RUR mode

- **Configurable time window**
  - For each reference snapshot, progenitor and descendant candidates are searched within `nsnap` snapshots before and after.
  - A minimum shared-particle fraction (`mcut`) controls how aggressive the candidate selection is.

- **Memory-aware, restartable workflow**
  - Large catalogues and particle data are streamed from disk and periodically flushed.
  - Intermediate “leaf” data are stored as per-snapshot, per-object pickle files and can be resumed from a temporary `treebase` state.
  - Logging is handled via rotating log files for each snapshot and for the global run.

- **Post-processing of links**
  - Converts many-to-many candidate links into:
    - direct **father/son** relations,
    - optional “indirect” or secondary links,
    - forward/backward propagation of branches to define long-lived objects.
  - Produces both a full “object-rich” catalogue and a “light” version without Python objects for fast analysis.

---

## Repository structure

The core logic is implemented in a small number of modules:

- **`ytool.py`**
  - Helper utilities:
    - `DotDict` parameter container
    - I/O helpers (`pklsave`, `pklload`)
    - Numba-accelerated numerical routines
    - Snapshot indexing (`out2step`, `step2out`)
    - Simple logging and timing tools

- **`yroot.py`**
  - Core data structures:
    - `TreeBase`: manages snapshots, catalogues, particle–halo matching, and “leaf” objects.
    - `Leaf`: per-galaxy (or per-halo) container with member particle IDs, velocities, weights, and candidate links.
  - Handles:
    - loading RAMSES snapshots via RUR,
    - loading GalaxyMaker/HaloMaker catalogues,
    - computing candidate links between snapshots,
    - writing/reading temporary per-snapshot backup files,
    - memory flushing and summary reports.

- **`yrun.py`**
  - High-level orchestration:
    - `do_onestep(...)`: runs the full matching procedure for a single snapshot (`iout`).
    - `gather(...)`: merges all per-snapshot “bricks” into a single `*all.pickle` catalogue.
    - `connect(...)`: builds direct father/son relations (`*fatson.pickle`).
    - `build_branch(...)`: builds long-lived branches and saves `*stable.pickle` and `*light.pickle`.
  - Also contains logging helpers (`make_log`, `follow_log`) and time/memory reporting.

- **`ysub.py`**
  - Small command-line wrapper for processing a single snapshot.
  - Intended to be called from a batch script or job scheduler (e.g. one `ysub.py` invocation per `iout`).

---

## Typical workflow

A typical workflow consists of three main stages:

1. **Prepare a parameter file**

   Create a Python parameter file (e.g. `params_nc.py`) that defines a `DotDict` of configuration values, including:

   - Simulation selection and repository:
     - `mode` (e.g. `"nc"`, `"nh"`, `"hagn"`, `"y..."`, `"custom"`)
     - `galaxy` (`True` for galaxy trees, `False` for halo trees)
     - `resultdir` (output directory)
     - `fileprefix` (prefix for all output files)
   - Matching configuration:
     - `nsnap` (number of snapshots to search forward/backward)
     - `mcut` (minimum shared-mass / shared-particle fraction)
     - `fcontam`, `strict` (contamination and filtering options)
   - Runtime configuration:
     - `ncpu` (number of OpenMP / Numba threads)
     - `flushGB` (memory threshold for flushing temporary data)
     - `ontime`, `onmem`, `oncpu`, `verbose` (debugging and profiling options)

   You can use the existing example parameter file as a template.

2. **Run the per-snapshot matching**

   The actual parallelisation strategy is up to you (e.g. MPI, job array, or simple loop).  
   Conceptually, each snapshot is processed by something like:

   ```bash
   python ysub.py \
       <iout> <fout> <reftot> \
       <resultdir> <logprefix> <mainlogname> <maxout>
