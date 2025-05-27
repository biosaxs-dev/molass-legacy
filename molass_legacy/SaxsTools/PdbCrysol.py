"""
    PdbCrysol.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import sys
import os
import numpy as np

def pdb2crysol_impl(pdbid, out_folder):
    from Pdb.PypdbLite import get_pdb_file
    from molass_legacy.ATSAS.Crysol import CrysolExecutor

    ret = get_pdb_file(pdbid, filetype='pdb')
    pdb_file = pdbid + '.pdb'
    pdb_path = os.path.join(out_folder, pdb_file)
    with open(pdb_path, "w") as fh:
                        fh.write(ret)

    crysol = CrysolExecutor()
    print('crysol.execute')
    crysol.execute(pdb_path)
    # crysol.move(path, sas_folder)

if __name__ == '__main__':
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')
    import molass_legacy.KekLib, SerialAnalyzer

    # pdb2crysol_impl("1AON", "temp")
    # pdb2crysol_impl("1FFK", "temp")
    # pdb2crysol_impl("1GQ2", "temp")
    # pdb2crysol_impl("1HTO", "temp")
    # pdb2crysol_impl("1VQ9", "temp")
    # pdb2crysol_impl("1XI4", "temp")
    # pdb2crysol_impl("2J28", "temp")
    pdb2crysol_impl("2RDO", "temp")
