"""
    find_the_most_likely.py

"""
import sys
sys.path += [
    r"D:\PyTools\pytools-2_9_5_develop\lib",
    r"D:\PyTools\pytools-2_9_5_develop\lib\KekLib",
    r"D:\PyTools\pytools-2_9_5_develop\lib\DataStructure",
    r"D:\PyTools\pytools-2_9_5_develop\lib\SerialAnalyzer",
    ]

import os
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
from glob import glob
import subprocess
from CrysolUtils import np_loadtxt_crysol
from molass_legacy.DENSS.DenssUtils import fit_data

debug = False

this_dir = os.path.dirname(os.path.abspath(__file__))
crysol_dir = os.path.join(this_dir, "crysol_files")

component_file = r"D:\TODO\20231031\CRYSOL\component-1.dat"
crysol_exe = r"C:\Program Files\ATSAS-3.1.3\bin\crysol.exe"

target = np.loadtxt(component_file)
print(target.shape)
qc, Ic, Icerr, D = fit_data(*target.T)

qv = None
ty = None
target_spline = UnivariateSpline(qc, Ic, s=0, ext=3)

def normalize(y):
    i = len(y)//4
    return y/np.max(y[0:i])

def compute_rmsd(y1, y2):
    return np.sqrt(np.mean((y1 - y2)**2))

min_rmsd = None
min_code = None

fh = open("rmsd.csv", "w")

for k, file in enumerate(glob(r"D:\TODO\20231031\PDB\aldolase_non_mutants\*.*")):
    print([k], file)
    if file[-3:] == "pdb":
        pass
    else:
        new_name = file[:-5] + "-ba" + file[-1] + ".pdb"
        os.rename(file, new_name)
        file = new_name

    crysol_file = file[-12:-3] + "int"
    crysol_path = os.path.join(crysol_dir, crysol_file)
    if os.path.exists(crysol_path):
        pass
    else:
        # uncomment this to stop adding crysol results
        # continue

        if os.path.exists(crysol_file):
            os.rename(crysol_file, crysol_path)
        else:
            cmd = [crysol_exe, file, component_file, '--constant']
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0:
                os.rename(crysol_file, crysol_path)
            else:
                print(result.stderr)
                continue

    # cleanup
    print(crysol_file)
    pdbcode = crysol_file[:-4]
    for file in glob(pdbcode + "*.*"):
        print("delete", file)
        os.remove(file)

    crysol_data = np_loadtxt_crysol(crysol_path)[0]

    if qv is None:
        qv = crysol_data[:,0]
        ty = normalize(target_spline(qv))

    cy = normalize(crysol_data[:,1])
    rmsd = compute_rmsd(cy, ty)
    fh.write(",".join([pdbcode, "%g" % rmsd]) + "\n")
    fh.flush()
    if min_rmsd is None or rmsd < min_rmsd:
        min_rmsd = rmsd
        min_code = pdbcode
    if debug:
        print("rmsd=", rmsd)
        fig, ax = plt.subplots()
        ax.set_yscale('log')
        ax.plot(qv, ty)
        ax.plot(qv, cy)
        fig.tight_layout()
        plt.show()

    if k > 1:
        # break
        pass

print("min_code=%s, min_rmsd=%g" % (min_code, min_rmsd))

fh.close()
