"""
    StateSequence.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import os
import logging
import numpy as np
from datetime import datetime

NUM_FUNC_CALLS_LOCALOPT = 5000

class StateSequence:
    def __init__(self, work_folder=None, niter=100, seed=None):
        if work_folder is None:
            self.fv_array = np.array([])
            self.x_array = np.array([])
            self.xmax = niter * NUM_FUNC_CALLS_LOCALOPT
            if seed is None:
                seed = np.random.randint(100000, 999999)
            self.seed = seed
        else:
            self.fv_array, self.x_array, self.xmax, self.seed = self.get_info_from_folder(work_folder)

    def get_info(self):
        return self.fv_array, self.x_array, self.xmax, self.seed

    def get_nc(self):
        return self.logfile.nc

    def insert(self, fv, x):
        self.fv_array = np.concatenate([self.fv_array, (None, fv, None, None)])     # [fv] must be consistent with (counter, fv, accept, time)
        self.x_array = np.concatenate([self.x_array, [x]])

    def __repr__(self):
        return "StateSequence(%d, %d, %s, %s)" % (len(self.fv_array), len(self.x_array), str(self.xmax), str(self.seed))

    def get_info_from_folder(self, work_folder, demo_index=0):
        from .OptLogFile import OptLogFile
        self.logfile = OptLogFile(os.path.join(work_folder, "optimizer.log"))
        seed_txt = os.path.join(work_folder, 'seed.txt')
        seed = read_seed_txt(seed_txt)
        cb_file = os.path.join(work_folder, 'callback.txt')
        fv_list, x_list = read_callback_txt_impl(cb_file)
        if demo_index > 0:
            n = demo_index + 1
        else:
            n = len(fv_list)
        return np.array(fv_list[0:n]), np.array(x_list[0:n]), fv_list[-1][0], seed

    def get_in_folder(self):
        return self.logfile.optdict['-f']

def save_opt_params(fh, x, f, accept, eval_counter):
    precision_save = np.get_printoptions()["precision"]
    np.set_printoptions(precision=17)

    fh.write("t=%s\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    fh.write("x=\n%s\n" % str(x))
    fh.write("f=%g\n" % f)
    fh.write("a=%s\n" % str(accept))
    fh.write("c=%d\n" % eval_counter)
    fh.flush()

    np.set_printoptions(precision=precision_save)

def read_callback_txt_impl(cb_file):
    import os
    import re
    from datetime import datetime
    from molass_legacy.KekLib.NumpyArrayUtils import from_space_separated_list_string

    time_re = re.compile(r't=(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})')
    space_re = re.compile(r'\s+')
    fv_re = re.compile(r'f=(\S+)')   # f=-0.375527 as well as f=138548, f=-8.52599e-05
    accept_re = re.compile(r'a=(\w+)')
    counter_re = re.compile(r'c=(\d+)')

    fv_list = []
    x_list = []
    accept = None
    time = None

    if os.path.exists(cb_file):
        fh = open(cb_file)
        for k, line in enumerate(fh):
            # print([k], line)
            if line[0:2] == "t=":
                m = time_re.search(line)
                if m:
                    year = int(m.group(1))
                    month = int(m.group(2))
                    day = int(m.group(3))
                    hour = int(m.group(4))
                    minute = int(m.group(5))
                    second = int(m.group(6))
                    time = datetime(year, month, day, hour, minute, second)
            elif line[0:2] == "x=":
                pass
            elif line[0] == "[":
                x_str = line
            elif line[0] == " ":
                x_str += line
            elif line[0:2] == "f=":
                m = fv_re.search(line)
                if m:
                    fv = float(m.group(1))
            elif line[0:2] == "a=":
                m = accept_re.search(line)
                if m:
                    accept = m.group(1) == 'True'
            elif line[0:2] == "c=":
                m = counter_re.search(line)
                if m:
                    counter = int(m.group(1))
                    fv_list.append((counter, fv, accept, time))
                    x = from_space_separated_list_string(x_str)
                    x_list.append(x)
        fh.close()
    else:
        # not yet written
        pass

    return fv_list, x_list

def read_seed_txt(seed_txt):
    import re

    seed_re = re.compile(r'seed=(\d+)')
    try:
        with open(seed_txt) as fh:
            for line in fh:
                m = seed_re.search(line)
                if m:
                    seed = int(m.group(1))
    except:
        seed = None
    return seed
