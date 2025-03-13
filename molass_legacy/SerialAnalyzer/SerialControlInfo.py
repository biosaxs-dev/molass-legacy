# coding: utf-8
"""

    SerialControlInfo.py

    Copyright (c) 2019-2020, SAXS Team, KEK-PF

"""
import os

class SerialControlInfo:
    def __init__( self,
                    data_folder,
                    conc_folder,
                    work_folder,
                    temp_folder,
                    conc_file=None,
                    serial_data=None,
                    guinier_folder=None,
                    stamp_file=None,
                    serial_file='serial_result.csv',
                    book_file='serial_analysis_report.xlsx',
                    min_analysis_value=0.5,
                    range_type=0,
                    analysis_ranges=None,
                    preview_params=None,
                    zx=False,
                    mapped_info=None,
                    maintenance_log=None,
                    env_info=None,
                    known_info_list=None,
                    cleaner=None,
                    parent=None ):

        self.data_folder    = data_folder
        self.conc_folder    = conc_folder
        self.work_folder    = work_folder
        self.temp_folder    = temp_folder
        self.conc_file      = conc_file
        self.serial_data    = serial_data
        self.guinier_folder = guinier_folder
        self.stamp_file     = stamp_file
        self.serial_file    = serial_file
        self.book_file      = book_file
        self.min_analysis_value    = min_analysis_value
        self.range_type     = range_type
        self.analysis_ranges = analysis_ranges
        self.preview_params = preview_params
        self.zx             = zx
        self.mapped_info    = mapped_info
        self.maintenance_log    = maintenance_log
        self.env_info       = env_info
        self.known_info_list = known_info_list
        self.cleaner        = cleaner
        self.parent         = parent
        self.more_multicore = os.cpu_count() > 4
        # self.more_multicore = False
