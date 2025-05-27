"""
    Global.V2Init.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
from molass_legacy.Experiment.DataUtils import get_columntype, update_sec_settings_impl

def update_sec_settings(el_option=0, column_name=None, exclusion_limit=None, num_plates_pm=None):

    if column_name is None:
        columntype = get_columntype()
        column_name = columntype.name
        if num_plates_pm is None:
            num_plates_pm = columntype.num_plates

    update_sec_settings_impl(el_option, column_name, exclusion_limit, num_plates_pm)
