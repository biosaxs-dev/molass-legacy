# coding: utf-8
"""
    DataDebugger.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import os
import re
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry

foler_re = re.compile(r'(\d+)$')

class DataDebugger:
    def __init__(self, folder_name="debugger-00"):
        folder = os.path.join(get_setting('analysis_folder'), folder_name)
        folder = self.get_new_folder(folder)

        mkdirs_with_retry(folder)
        self.folder = folder

    def get_new_folder(self, folder):
        while os.path.exists(folder):
            folder = re.sub(foler_re, lambda m: '%02d' % (int(m.group(1))+1), folder)
            # print('next folder=', folder)

        # print('final folder=', folder)
        return folder

    def get_sub_folder(self, subfolder_name="00"):
        sub_folder = os.path.join(self.folder, subfolder_name)
        sub_folder = self.get_new_folder(sub_folder)
        mkdirs_with_retry(sub_folder)
        return sub_folder
