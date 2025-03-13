# coding: utf-8
"""
    InputDataUtils.py

    Migration tool aimed to be compatible with SerialDataUtils.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import os
import re
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting, set_path_length
from SerialDataUtils import get_uv_filename, get_mtd_filename
from InputData import InputData

teller = None
current_data = None

def initialize(debug=False):
    global teller
    if teller is None:
        from InputProcess.InputTeller import InputTeller
        this_dir = os.path.dirname( os.path.abspath( __file__ ) )
        log_folder = os.path.abspath( this_dir + '/../../log' )
        teller = InputTeller(log_folder=log_folder, debug=debug)

def terminate():
    # without this, this parent process won't stop.
    global current_data
    global teller
    del current_data
    del teller

def _get_data(in_folder, debug=True):
    global current_data
    initialize(debug=debug)

    if current_data is None:
        current_data = InputData(in_folder, teller, debug=debug)
        current_data.prepare()

    return current_data

def clear_current_data():
    global current_data
    current_data = None

def update_path_length(comments):
    measurement_date = None
    if comments is not None:
        for line in comments:
            if line.find('Date') > 0:
                date_re = re.compile(r'(\d+/\d+/\d+)')
                m = date_re.search(line)
                if m:
                    measurement_date = int(m.group(1).replace('/', ''))
                    break
    set_setting('measurement_date', measurement_date)
    set_path_length(measurement_date)

def load_intensity_files( in_folder, logger=None, debug=True ):
    data = _get_data(in_folder)
    data_array = data.make_redundant_xray_data()
    update_path_length(data.comments)
    return data_array, data.files

def load_uv_array( conc_folder, conc_file=None, column_header=False ):
    disable_uv_data = get_setting('disable_uv_data')

    if disable_uv_data:
        uv_data = None
    else:
        data = _get_data(conc_folder)
        uv_data = data.uv_data

    if uv_data is None:
        data_array = None
        lvector = None
        conc_file = None
        col_header = None
    else:
        data_array = uv_data.data
        lvector = uv_data.vector
        folder_info = data.folder_info
        conc_file = folder_info.conc_file
        col_header = [None]*(data_array.shape[1]+1)     # dummies; assuming that col_header is not used

    if column_header:
        return data_array, lvector, conc_file, col_header
    else:
        return data_array, lvector, conc_file
