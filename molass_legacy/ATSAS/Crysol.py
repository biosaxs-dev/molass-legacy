"""
    ATSAS.Crysol.py

    Copyright (c) 2016-2023, SAXS Team, KEK-PF
"""
import os
import re
import subprocess
from molass_legacy.KekLib.BasicUtils import Struct
from .AutoRg import autorg_exe_array

class CrysolExecutor:
    def __init__(self):
        self.exe_path   = None
        if len( autorg_exe_array ) > 0:
            crysol_path = autorg_exe_array[0].replace( 'autorg.exe', 'crysol_30.exe' )
            if os.path.exists( crysol_path ):
                self.exe_path   = crysol_path

    def execute(self, pdb_file, exp_file=None):
        cmd = [self.exe_path, pdb_file]
        if exp_file is not None:
            cmd += [exp_file]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            print(result.stderr.decode())
        result.check_returncode()

    def move(self, pdb_file, tgt_folder):
        from shutil import move
        pattern = pdb_file.replace('.pdb', '*00*.*')
        for path in glob.glob(pattern):
            print(path)
            move(path, tgt_folder)

rg_re = re.compile(r"\s+(\S+)$")

def get_info_from_crysol_log(path):
    rg = None
    with open(path) as fh:
        for line in fh:
            if line[0:15].find("Rg (Atoms") >= 0:
                m = rg_re.search(line[:-1])
                if m:
                    rg = float(m.group(1))
    return Struct(Rg=rg)

chi_square_re = re.compile(r"\s+(\S+)$")

def get_info_from_crysol_fit_log(path):
    chi_square = None
    with open(path) as fh:
        for line in fh:
            if line[0:15].find("Chi-square") >= 0:
                m = chi_square_re.search(line[:-1])
                if m:
                    chi_square = float(m.group(1))
    return Struct(Chi_square=chi_square)
