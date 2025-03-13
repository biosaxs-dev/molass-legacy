"""
    _MOLASS.WorkUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
from .SerialSettings import get_setting

def get_temp_folder(make_folder=True):
    temp_folder = get_setting('temp_folder')

    if temp_folder is None:
        analysis_folder = get_setting('analysis_folder')
        assert analysis_folder is not None, 'analysis_folder is not set'
        temp_folder = os.path.join(analysis_folder, '.temp')

    if make_folder:
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
    else:
        assert os.path.exists(temp_folder), 'temp_folder does not exist'

    return temp_folder
