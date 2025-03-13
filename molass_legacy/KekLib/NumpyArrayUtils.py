"""
    NumpyArrayUtils.py

    Copyright (c) 2021-2024, Masatsuyo Takahashi, KEK-PF
"""

import re
import numpy as np
from numpy import nan, inf      # these may be included in the list_str

"""
    convert string like below formmated by numpy back to array

    [[  2.23716837 145.93528911  17.91118771  -7.89019142  93.37566492]
     [  3.77747848 183.18824456  13.02506409  -6.50163151  57.1449466 ]
     [ 16.89832734 220.08621531  21.76543691  10.8143736   40.3889722 ]
     [  2.06556164 306.82469767  nan -14.11287484  inf ]]
"""
def from_space_separated_list_string(ssl):
    # insert commas between values
    sep_values_re = re.compile(r"([^\[\s])\s+([^\[\s])")
    list_str = re.sub(sep_values_re, lambda m: m.group(1) + ',' + m.group(2), ssl)

    # insert commas between lists
    sep_parens_re = re.compile(r"(\])\s+(\[)")
    list_str = re.sub(sep_parens_re, lambda m: m.group(1) + ',' + m.group(2), list_str)
    return np.array(eval(list_str))

if __name__ == '__main__':
    ssl = """
        [[  2.23716837 145.93528911  17.91118771  -7.89019142  93.37566492]
         [  3.77747848 183.18824456  13.02506409  -6.50163151  57.1449466 ]
         [ 16.89832734 220.08621531  21.76543691  10.8143736   40.3889722 ]
         [  2.06556164 306.82469767  nan -14.11287484  inf ]]
     """
    print(from_space_separated_list_string(ssl.replace("\n", "")))
