"""
    Alsaker.Bridge.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
import numpy as np
from testfixtures import replace_in_module
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import STAP
from rpy2.robjects import default_converter, numpy2ri

R_read_table = robjects.r['read.table']
R_matrix = robjects.r['matrix']
R_rep = robjects.r['rep']

"""
    These are patch functions to suppress decode errors which are ignorable.
"""
import rpy2.rinterface_lib.conversion as conversion
from rpy2.rinterface_lib.conversion import ffi

def modified_cchar_to_str(c, encoding: str) -> str:
    # TODO: use isString and installTrChar
    s = ffi.string(c).decode(encoding, errors='ignore')
    return s

def modified_cchar_to_str_with_maxlen(c, maxlen: int, encoding: str) -> str:
    # TODO: use isString and installTrChar
    s = ffi.string(c, maxlen).decode(encoding, errors='ignore')
    return s

# end of patch functions

def do_it_in_the_context(closure):
    """
    workaround for multiple patching of rpy2
    i.e., generator tuple or list are not supported in with statement
    """
    with (  replace_in_module(conversion._cchar_to_str, modified_cchar_to_str),
            replace_in_module(conversion._cchar_to_str_with_maxlen, modified_cchar_to_str_with_maxlen)
    ):
        return closure()

class RgcodeBrige:
    def __init__(self):
        this_dir = os.path.dirname(__file__)
        self.rg_code_dir = os.path.join(this_dir, "Rg code")
        rg_code_file = os.path.join(self.rg_code_dir, "file1-modified-for-molass.R")
        print("R_HONE=", os.environ['R_HOME'])
        print(rpy2.__version__)
        with open(rg_code_file) as f:
            rg_code_string = f.read()

        self.rg_code = STAP(rg_code_string, "estimate_Rg")
 
    def get_sample_data_path(self, name):
        return os.path.join(self.rg_code_dir, name)

    def estimate_Rg(self, data, num_reps, starting_value=None, return_y3=False):
        if starting_value is None:
            starting_value = R_rep(1, num_reps)
        else:
            starting_value = R_rep(starting_value, num_reps)

        print("type(starting_value)=", type(starting_value))    # <class 'rpy2.robjects.vectors.IntVector'>
        print("starting_value=", starting_value)

        if type(data) is np.ndarray:
            numpy2ri.activate()

            nr, nc = data.shape
            M = R_matrix(data, nrow=nr, ncol=nc)
            # this call must take place before numpy2ri.deactivate()
            output = self.rg_code.estimate_Rg(M, num_reps, starting_value=starting_value, return_y3=return_y3)

            numpy2ri.deactivate()
        else:
            M = data
            output= self.rg_code.estimate_Rg(M, num_reps, starting_value=starting_value)
        return output

def bridge_test_impl(caller):
    print("Bridge trial")

    def closure():
        rb = RgcodeBrige()
        filepath = rb.get_sample_data_path("oval_01C_S008_0_01.dat")

        # basic call test
        data =  R_read_table(filepath, header=False)
        print(type(data))
        output = rb.estimate_Rg(data, 1)
        print(type(output), output)

        # numpy interface test
        data = np.loadtxt(filepath)
        print("data.shape=", data.shape)
        output = rb.estimate_Rg(data, 1)
        print(type(output), output)
    
    do_it_in_the_context(closure)