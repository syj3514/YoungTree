import argparse, subprocess
import os
from ytool import *

parser = argparse.ArgumentParser(description='YoungTree (syj3514@yonsei.ac.kr)')
parser.add_argument("params", metavar='params', help='params.py', type=str)
parser.add_argument("path", metavar="path", help="/path/to/the/ytree_treebase.temp.pickle", type=str)
parser.add_argument("-m", "--mode", required=False, help='Simulation mode', type=str)
args = parser.parse_args()
params = make_params_dict(args.params, mode=args.mode)

if __name__=='__main__':
    if(os.path.exists(args.path)):
        print(f"Load `{args.path}`...")
        treebase = pklload(args.path)
        print(f"Done")

        # oldp = treebase.p
        # treebase.p = params
        for key in params.keys():
            if key in treebase.p.keys():
                if(not isinstance(params[key], np.ndarray)):
                    if params[key] != treebase.p[key]:
                        print(f"Update {key}: {treebase.p[key]} -> {params[key]}")
                        treebase.p[key] = params[key]
                else:
                    if not np.all(params[key] == treebase.p[key]):
                        print(f"Update {key}: {treebase.p[key]} -> {params[key]}")
                        treebase.p[key] = params[key]
            else:
                print(f"Add {key}: {params[key]}")
        confirm = input("Do you want to update? [y/n]: ")
        if(confirm in ['y', 'Y', 'yes', 'Yes']):
            pklsave(treebase, args.path, overwrite=True)
            print(f"Save `{args.path}`...")
        else:
            print("Canceled")
            exit()