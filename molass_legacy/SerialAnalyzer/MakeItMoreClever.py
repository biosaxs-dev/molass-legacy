# coding: utf-8
"""
    MakeItMoreClever.py

    Copyright (c) 2019-2021, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import is_empty_val
from molass_legacy._MOLASS.SerialSettings import get_setting

def increase_menu_availability(parent):
    an_folder = parent.an_folder.get()
    analysis_name = parent.analysis_name.get()
    analysis_folder = get_setting('analysis_folder')

    if not is_empty_val(an_folder) and not is_empty_val(analysis_name) and analysis_folder is None:
        parent.new_analysis()
        parent.menu4.update_states()
