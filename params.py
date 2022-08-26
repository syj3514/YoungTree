#   [mode]
#   Simulation name (should be supported by rur)
#       hagn:     Horizon-AGN
#       y01605:   YZiCS 01605
#       y04466:   YZiCS 04466
#       ...
#       y49096:   YZiCS 49096
#       nh:       NewHorizon
mode = 'nh'

#   [galaxy]
#   Type of data
#       True:     Use galaxy and star
#       False:    Use halo and DM
galaxy = True

#   [iout]
#   Starting output number
#       Specific iout or -1 (last snapshot)
iout = -1

#   [prog]
#   Finding ascending or descending order of output
#       True:   Find progenitor to past (father)
#       False:  Find descendant to future (son)
prog = True

#   [usegals]
#   Root galaxies ID
#       'all':    All galaxies
#       int ID:   Specific one galaxy
#       ID list:  Specific several galaxies
#       tuple:    Use galaxies mass range (minmass, maxmass)
#usegals = 'all'
usegals = 1

#   [overwrite]
#   If tree results already exist, overwrite or not?
#       True: overwrite
#       False: skip
overwrite = True

#   [logname]
#   file name of output log (./logname.log)
logname = f"output_{mode}"

#   [detail]
#   Detail debugging in log
#       True:   DEBUG level
#       False:  INFO level
detail = True

#   [flush_GB]
#   Memory threshold for auto-flush in Gigabytes
flush_GB = 200


