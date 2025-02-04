#---------------------------------------------------------------------
#   Main Parameters
#       These parameters influence the output of YoungTree.
#---------------------------------------------------------------------

# ! This is overwritten by the command line arguments (-m or --mode) !
mode = "nc" #   Simulation name (should be supported by rur)
#       "hagn":     Horizon-AGN
#       "y01605":   YZiCS 01605
#       "y04466":   YZiCS 04466
#       ...
#       "y49096":   YZiCS 49096
#       "fornax":   FORNAX
#       "nh":       NewHorizon
#       "nh2":      NewHorizon2
#       "nc":       NewCluster
#       "custom":   Custom simulation

galaxy = True # True: galaxy, False: halo
fullpath=None # Full path to the HaloMaker output directory
nsnap = 4 # The number of snapshots to be considered when making trees
mcut = 0.01# Number fraction threshold for the prog/desc candidates
finalout = 623 # Maximum output number to be considered




#---------------------------------------------------------------------
#   Running Parameters
#       These parameters do not influence the output of YoungTree.
#       However, they can affect the performance of YoungTree.
#---------------------------------------------------------------------

# ! This is overwritten by the command line arguments (-n or --ncpu) !
ncpu = 48 # Set nthread in numba and OpenMP
loadall = False # Load all galaxies and particles at once
usefortran = False # Use Fortran for particle loading
takeover = True # If True, the treebase is taken over from the previous run
flushGB = 400 # Memory threshold for auto-flush in Gigabytes
nice = 1 # set os nice value






#---------------------------------------------------------------------
#   Logging Parameters
#       These parameters decide the logging format
#---------------------------------------------------------------------

path_in_repo = "YoungTree" # Where output and log files are saved
fileprefix = "ytree_"
logprefix = f"ytree_623_" # file name of ytree log (./logprefix_iout.log)
detail = False # Detail debugging in log (True: DEBUG level, False: INFO level)
ontime = True # Time check
onmem = False # Memory check
oncpu = False # CPU check
verbose = 1 # Verbosity level (0: quiet, 5: verbose)
mint = 0 # Minimum time for logging (`Done` prints only if t > mint)





#---------------------------------------------------------------------
# Galaxy Filtering Function
#     This function is applied to the galaxy list.
#     The function should return a boolean array.
#     This function SHOULD NOT include any arguments.
#     The default function returns True for all galaxies.
#
#     If strict,
#         The function is applied to the all snapshots.
#     else,
#         The function is applied to the initial snapshot only
#         so even if `unsatisfied` progenitors are found, those progenitors
#         are not removed from the tree.
#         However, in other snapshots, if the galaxy is not selected as a progenitor and
#         if the galaxy is not satisfied, the galaxy is removed from the tree.
#---------------------------------------------------------------------
light = False
default = True
if(default):
    def filtering(gals):
        import numpy as np
        return np.ones(len(gals), dtype=bool)

else:
    strict = False
    # Example 1) Mass cut
    def filtering(gals):
        import numpy as np
        return gals["m"] > 10**10.5

    # Example 2) ID list
    # ids = [1, 10, 12, 20]
    # def filtering(gals):
    #     import numpy as np
    #     return np.isin(gals["id"], ids, assume_unique=True)

    # Example 3) Box region
    # def filtering(gals):
    #     import numpy as np
    #     x1, x2 = 0.45, 0.51
    #     y1, y2 = 0.41, 0.45
    #     z1, z2 = 0.46, 0.51
    #     return (gals["x"] > x1) & (gals["x"] < x2) & \
    #            (gals["y"] > y1) & (gals["y"] < y2) & \
    #            (gals["z"] > z1) & (gals["z"] < z2)
