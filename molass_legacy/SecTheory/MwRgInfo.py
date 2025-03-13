"""
    SecTheory.MwRgInfo.py

    Copyright (c) 2022-2023, SAXS Team, KEK-PF
"""
import os
import numpy as np
from scipy.stats import linregress
import io
import re

TABLE1 = """
Table 1
Light scattering results for standard globular proteins.

Protein             Oligomeric state  Mw(kDa)   Rh(nm)  Rg(nm)
------------------------------------------------------------
Aprotinin               Monomer         6.8     1.35    1.04
Cytochrome C            Monomer         12      1.77    1.37
α-Lactalbumin          Monomer         14.3    1.91    1.49
Myoglobin               Monomer         14.2    2.12    1.64
Carbonic anhydrase      Monomer         29.2    2.35    1.82
Trypsin inhibitor       Monomer         20.5    2.47    1.91
β-Lactoglobulin        Monomer         20.1    2.64    2.04
Ovalbumin               Monomer         42.5    2.98    2.31
Bovine serum albumine   Monomer         67.1    3.56    2.76
Enolase (yeast)         Dimer           79.5    3.57    2.76
Enolase (rabbit)        Dimer           86.4    3.65    2.83
Transferrin             Monomer         76.9    4.02    3.11
Alcohol dehydrogenase   Tetramer        144     4.5     3.48
BSA dimer               Dimer           137.1   3.68    3.62
Aldolase (rabbit)       Tetramer        155     4.77    3.69
------------------------------------------------------------

from
https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4603275/pdf/j-48-01604.pdf

Molecular weight - gyration radius relation of globular proteins:
a comparison of light scattering, small-angle X-ray scattering
and structure-based data

by Detlef-M. Smilgies and Ewa Folta-Stogniew


Table 1
Experimental and computed structural parameters of the studied proteins.

Protein                PDB used       MM(kDa)            Rg(nm)
-------------------------------------------------------------
Ribonuclease A          1FS3            13.7      _      1.58
Lysozyme                1LYZ            14.3      _      1.43
Chymotrypsinogen A      2CGA            25.0      _      1.85
Carbonic anhydrase      1V9E            29.0      _      2.08
Ovalbumin               1OVA            45.0      _      2.66
BSA                     1N5U            66.0      _      2.99
Alcohol dehydrogenase   1JVB           150        _      3.27
Aldolase                1ZAH           158        _      3.51
Glucose isomerase       1OAD           173        _      3.25
β-Amylase              1FA2           200        _      4.22
Catalase                4BLC           232        _      3.84
Apoferritin             1IER           440        _      7.05
Thyroglobulin           _              669        _      7.56
-------------------------------------------------------------

from
https://journals.iucr.org/j/issues/2007/s1/00/sm6016/sm6016.pdf
Accuracy of molecular mass determination of
proteins in solution by small-angle X-ray scattering

by Efstratios Mylonasa and Dmitri I. Svergun


QUERY: Molecular Weight per Deposited Model >= 800, etc.

Protein                PDB used       MM(kDa)            Rg(nm)
-------------------------------------------------------------
GROEL                  1AON           841.75      _     6.972
RIBOSOME               1FFK          1321.3       _     6.928
OXIDOREDUCTASE         1GQ2          1015.3       _     8.430
LIGASE                 1HTO          1296.2       _     8.260
-------------------------------------------------------------

ENDOCYTOSIS/EXOCYTOSIS 1XI4          1683.0       _    14.80
RIBOSOME               2J28          1411.1       _     7.647

by PDB and CRYSOL3.0
"""

MWRG_CSV_PATH = os.path.join(os.path.dirname(__file__), "mwrg.csv")

class MwRgInfo:
    def __init__(self):
        table = io.StringIO(TABLE1)
        is_data_line = 0
        names = []
        data = []
        for line in table:
            if line.find("----") >= 0:
                is_data_line = 1 - is_data_line
                continue
            if is_data_line:
                words = re.split(r"\s+", line[:-1])
                mw = float(words[-3])
                rg = float(words[-1])*10
                data.append((mw, rg))
                names.append(words[-4])

        data = np.array(data)

        if os.path.exists(MWRG_CSV_PATH):
            extra_data = []
            with open(MWRG_CSV_PATH) as fh:
                for line in fh:
                    words = line[:-1].split(',')
                    values = [float(s) for s in words[1:]]
                    if values[1] > 0:   # Rg > 0
                        extra_data.append(values)
                        names.append(words[0])
            extra_data = np.array(extra_data)
        else:
            extra_data = np.array([]).reshape((0,2))
        self.mws = np.concatenate([data[:,0], extra_data[:,0]/1000])
        self.rgs = np.concatenate([data[:,1], extra_data[:,1]])
        self.names = names

        self.result = linregress(np.log(self.mws), np.log(self.rgs))

    def compute_rg(self, mw, n_sigma=0):
        slope, intercept, rvalue, pvalue, stderr = self.result
        intercept_stder = self.result.intercept_stderr

        a = intercept + n_sigma*intercept_stder
        b = slope + n_sigma*stderr
        return np.exp(a + b*np.log(mw))

    def compute_mw(self, rg):
        slope, intercept = self.result[0:2]
        return np.exp((np.log(rg) - intercept)/slope)

mwrg_info = None

def get_mwrg_info(devel=False):
    global mwrg_info
    if mwrg_info is None or devel:
        mwrg_info = MwRgInfo()
    return mwrg_info
