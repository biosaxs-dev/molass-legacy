# coding: utf-8
"""

    ファイル名：   PilatusUtils.py

    処理内容：

       Pilatus 画像データ関連の共通処理

    Copyright (c) 2020, 2025, SAXS Team, KEK-PF
"""
import os
import re
import glob
import PilatusCounter
from PilatusUtilsOldStyle import get_data_info as get_data_info_old_style
from PilatusUtilsNewStyle import get_data_info as get_data_info_new_style, file_name_re

DEBUG = False
NEW_STYLE_ONLY = True

def get_in_folder_info( in_folder ):
    orig_folder = os.getcwd()
    os.chdir( in_folder )
    log_files = glob.glob( 'measure*.log' )
    if len( log_files ) == 0:
        log_files = glob.glob( '*.log' )
    mask_files = glob.glob( '*.mask' )
    os.chdir( orig_folder )

    log_file, mask_file = (None, None )
    if len( log_files ) >= 1:
        log_file = log_files[0]

    mask_file = None
    if len( mask_files ) >= 1:
        mask_file = mask_files[0]

    return log_file, mask_file

old_file_name_re = re.compile(r'(\w+)_(\d)_(\d{5})\.(\w{3})')

class FolderInfo:
    def __init__(self, path):
        self.path = path
        self.single_image = True
        self.new_style = None
        self.get_image_file_names(path)

    def get_image_file_names(self, folder):
        for ext in ['tif', 'cbf']:
            self.image_files = []
            old_type_files = []
            for k, path in enumerate(glob.glob(folder + r'\*.' + ext)):
                # print([k], path)
                folder, file= os.path.split(path)
                m = file_name_re.match(file)
                if m:
                    # print('match', [k], path)
                    self.image_files.append(path)
                    if int(m.group(3)) > 0:
                        self.single_image = False
                else:
                    if old_file_name_re.match(file):
                        old_type_files.append(path)

            if len(self.image_files) > 0 and len(old_type_files) == 0:
                if self.new_style is None:
                    self.new_style = True
                break
            else:
                if len(old_type_files) > 0:
                    self.new_style = False
                    break

        if DEBUG:
            print('-------- old_type_files=', old_type_files)
            print('-------- new_style=', self.new_style)

    def is_new_style(self):
        return self.new_style

def get_data_info( in_folder,
                    adj_folder, syn_folder, pilatus_counter, counter_id,
                    log_file_path=None,
                    sample_complete=False, for_test_data=False,
                    logger=None ):

    log_file        = None
    mask_file       = None

    if not in_folder or not os.path.exists( in_folder ):
        return log_file, mask_file, [], pilatus_counter

    log_file, mask_file = get_in_folder_info( in_folder )

    # TODO: refactor to move PilatusCounter out.
    if pilatus_counter is None:
        pilatus_counter = PilatusCounter.Counter( in_folder )

    if log_file_path is None:
        log_file_path = in_folder + '/' + log_file 

    folder_info = FolderInfo(in_folder)

    if NEW_STYLE_ONLY or folder_info.is_new_style():
        get_data_info_impl = get_data_info_new_style
    else:
        get_data_info_impl = get_data_info_old_style

    return get_data_info_impl(folder_info, log_file, mask_file,
                    adj_folder, syn_folder, pilatus_counter, counter_id,
                    log_file_path, sample_complete, for_test_data,
                    logger=logger)
