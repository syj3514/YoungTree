#   [mode]
#   Simulation name (should be supported by rur)
#       "hagn":     Horizon-AGN
#       "y01605":   YZiCS 01605
#       "y04466":   YZiCS 04466
#       ...
#       "y49096":   YZiCS 49096
#       "fornax":   FORNAX
#       "nh":       NewHorizon
#       "nh2":      NewHorizon2
#       "nc":       NewCluster
#       None
mode = "y49096"

#   [ncpu]
#   Set nthread in numba
#   If ncpu <=0, skip
ncpu = 24
nice = 1

#   [galaxy]
#   Type of data
#       True:     Use galaxy and star
#       False:    Use halo and DM
galaxy = False
fullpath=None
loadall = True
usefortran = False
fcontam = 0.1 # Ignore halos with contaminated greater than fcontam

#   [nsnap]
#   How many snapshots to use for each galaxy
nsnap = 5

#   [overwrite]
#   If tree results already exist, overwrite or not?
#       True: overwrite
#       False: skip
overwrite = True

#   [logprefix]
#   file name of ytree log (./logprefix_iout.log)
# logprefix = f"ytree_"
logprefix = f"ytree_dm_"

#   [detail]
#   Detail debugging in log
#       True:   DEBUG level
#       False:  INFO level
detail = False
ontime = True # Time check
onmem = False # Memory check
oncpu = False # CPU check
verbose = 1

#   [flushGB]
#   Memory threshold for auto-flush in Gigabytes
flushGB = 200


