"""
    Protein.UvSensitivity.py

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

def get_amino_acids_percent(pdb_file):
    parser = cif_parser if pdb_file[-4:] == '.cif' else pdb_parser
    structure = parser.get_structure("XXXX", pdb_file)

    result_dict = None
    for polypeptide in polypeptide_builder.build_peptides(structure):
        seq = polypeptide.get_sequence()
        analyzed_seq = ProteinAnalysis(str(seq))
        count_dict = analyzed_seq.count_amino_acids()
        if result_dict is None:
            result_dict = count_dict
        else:
            for k, v in count_dict.items():
                result_dict[k] += v

    # print(result_dict)
    total = np.sum(list(result_dict.values()))
    percent_dict = {}
    for k, v in result_dict.items():
        percent_dict[k] = v/total
    # print(percent_dict)

    return percent_dict

def get_ratio(proportion_dict):
    a280 = proportion_dict['W']
    a275 = proportion_dict['Y']
    a258 = proportion_dict['F']
    ratio = a280/a275 + a280/a258
    print("ratio=", ratio)
    return ratio

def plot_amino_acids_proportions(pdb_files, use_mass=False):
    fig, ax = plt.subplots()

    pdbids = []
    ratios = []
    for path in pdb_files:
        folder, file = os.path.split(path)
        pdbids.append(file.split("-")[0].upper())
        percent_dict = get_amino_acids_percent(path)
        if use_mass:
            method = "Mass"
            proportion_dict = {}
            total_mass = 0
            for k, v in percent_dict.items():
                mass = v*protein_weights[k]
                proportion_dict[k] = mass
                total_mass += mass
            for k, v in proportion_dict.items():
                proportion_dict[k] = v/total_mass
        else:
            method = "Count"
            proportion_dict = percent_dict
        ratio = get_ratio(proportion_dict)
        ratios.append(ratio)
        ax.bar(proportion_dict.keys(), proportion_dict.values(), alpha=0.3, label=file)

    ax.set_title("Amino Acid Proportions by %s: " % method + " vs ".join(pdbids))

    labels = ax.get_xticklabels()
    for i in [4, 18, 19]:
        labels[i].set_color("red")

    ax.legend()
    fig.tight_layout()

    print("FnR ratio = %.2g" % (ratios[1]/ratios[0]))

    plt.show()

def demo():
    import seaborn
    seaborn.set()

    DATA_FOLDER = r"D:\TODO\20231129\TDS\Data"
    paths = []
    # for file in ["2eph-ba1.pdb", "5f4x-ba1.pdb"]:
    for file in ["2eph-ba1.pdb", "3dfn-ba1.pdb"]:
        paths.append(os.path.join(DATA_FOLDER, file))

    plot_amino_acids_proportions(paths, use_mass=True)

if __name__ == '__main__':
    demo()
