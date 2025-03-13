"""

    AutoRunTestTools.py

    Copyright (c) 2020, SAXS Team, KEK-PF

"""
import os
import glob
from time import sleep
from datetime import datetime, timedelta, timezone
import numpy as np
from bisect import bisect_left
from shutil import copy

JST = timezone(timedelta(hours=+9), 'JST')

def my_strftime(t):
    return t.strftime('%Y-%m-%d %H:%M:%S')

class RscSimulator:
    def __init__(self, in_folder, out_folder):
        self.in_folder = in_folder
        self.out_folder = out_folder
        assert os.path.exists(self.out_folder) and os.path.isdir(self.out_folder)
        self.get_in_folder_info()

    def get_in_folder_info(self):
        measurement_log = None
        for k, path in enumerate(glob.glob(self.in_folder + r'\*.log')):
            if path.find('measurement') > 0:
                print([k], path)
                measurement_log = path

        mask_file = None
        for k, path in enumerate(glob.glob(self.in_folder + r'\*.mask')):
            print([k], path)
            mask_file = path

        pilatus_counters = []
        for k, path in enumerate(glob.glob(self.in_folder + r'\*.txt')):
            if path.find('PilatusCounter') > 0:
                print([k], path)
                pilatus_counters.append(path)

        self.measurement_log = measurement_log
        self.mask_file = mask_file
        self.pilatus_counters = pilatus_counters

        time_list = []
        file_list = []
        for ext in ['tif', 'cbf']:
            for k, path in enumerate(glob.glob(self.in_folder + r'\*.' + ext)):
                # print([k], path)
                stat_result = os.stat(path)
                if False:
                    for t in [stat_result.st_atime, stat_result.st_mtime, stat_result.st_ctime]:
                        print(datetime.fromtimestamp(t, JST).strftime('%Y-%m-%d %H:%M:%S.%f'))
                m_time = datetime.fromtimestamp(stat_result.st_mtime, JST)
                time_list.append(m_time)
                file_list.append(path)
                print([k], my_strftime(m_time), path)

        print(len(file_list))
        self.time_list = np.array(sorted(time_list))
        self.file_list = file_list

    def run(self, interval=10, timescale=6):
        for path in [self.measurement_log, self.mask_file] + self.pilatus_counters:
            if path is None:
                print("ERROR: One of required path is None!")
                print("Make sure the existence of measurement.log, mask_file, or pilatus_counters.")                
                assert False, "See the measage above."
            print(path)
            copy(path, self.out_folder)

        self.start_time = self.time_list[0] - timedelta(seconds=1)
        print(my_strftime(self.time_list[0]))
        print(my_strftime(self.start_time))

        copy_start = 0
        for step in range(1000):
            sleep(interval)
            current_time = self.start_time + timedelta(seconds=step*interval*timescale)
            print('step %d' % step, my_strftime(current_time))
            k = bisect_left(self.time_list, current_time)
            for j in range(copy_start, k):
                print('copy', [j], my_strftime(self.time_list[j]))
                path  = self.file_list[j]
                copy(path, self.out_folder)
            copy_start = k
