"""
    Protein.WypMassRatios.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser, MMCIFParser, CaPPBuilder
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.Data.IUPACData import protein_weights

pdb_parser = PDBParser()
cif_parser = MMCIFParser()
polypeptide_builder = CaPPBuilder()

def get_amino_acids_props(pdb_file):
    parser = cif_parser if pdb_file[-4:] == '.cif' else pdb_parser
    structure = parser.get_structure("XXXX", pdb_file)

    result_dict = {}
    for polypeptide in polypeptide_builder.build_peptides(structure):
        seq = polypeptide.get_sequence()
        analyzed_seq = ProteinAnalysis(str(seq))
        count_dict = analyzed_seq.count_amino_acids()
        for k, n in count_dict.items():
            v = result_dict.get(k, 0)
            v += n*protein_weights[k]
            result_dict[k] = v

    # print(result_dict)
    total = np.sum(list(result_dict.values()))
    props_dict = {}
    for k, v in result_dict.items():
        props_dict[k] = v/total

    return props_dict

def compute_mass_ratios(pdb_file):
    props_dict = get_amino_acids_props(pdb_file)
    w = props_dict['W']
    y = props_dict['Y']
    p = props_dict['P']
    return y/w, p/w

def create_mass_ratios_file(pdb_folder):
    from glob import glob

    ratio_file = os.path.join(pdb_folder, "wyp_ratios.csv")
    fh = open(ratio_file, "w")

    for path in glob(pdb_folder + r"\*.pdb"):
        print(path)
        ratios = compute_mass_ratios(path)
        folder, file = os.path.split(path)
        fh.write(",".join([file, *["%.3g" % r for r in ratios]]) + "\n")

    fh.close()

def estimate_coefficients():
    """
             
    1ova.pdb,2.96,2.63  0.981,0.552
    1ald.pdb,3.84,3.57  0.960,0.433
    2glk.pdb,1.33,1.79  0.965,0.566
    3v03.pdb,8.87,7.89  1.01, 0.611
    """
    X = [
          [0.981, 0.552],
          [0.960, 0.433],
          [0.965, 0.566],
          [1.01,  0.611],
          ]

    Y = [
          [2.96, 2.63],
          [3.84, 3.57],
          [1.33, 1.79],
          [8.87, 7.89],
          ]

if __name__ == '__main__':
    pdb_folder = r"D:\TODO\20231218\wyf-ratios\PDB"
    create_mass_ratios_file(pdb_folder)
