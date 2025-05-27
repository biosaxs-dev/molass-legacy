"""
    Experiment.DataUtils.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
import logging
import numpy as np
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting
from molass_legacy.Experiment.ColumnTypes import get_columntype_from_id, get_columntype_from_name, PORESIZE_ALLOWANCE
from molass_legacy.SecTheory.MwRgFigure import get_mwrg_info

DEFAULT_NUM_PLATES = 48000

UNKNOWN_NAMES = [
    '20160227', 'HIF',
]

def get_columntype(in_folder=None):
    if in_folder is None:
        in_folder = get_setting('in_folder')

    test_columtype_id = get_setting('test_columtype_id')
    if test_columtype_id is not None:
        return get_columntype_from_id(test_columtype_id)

    columtype_id = None
    for column_name in UNKNOWN_NAMES:
        if in_folder.find(column_name) >= 0:
            columtype_id = 'unknown'
            return get_columntype_from_id(columtype_id)

    if in_folder.find('20230706') >= 0:
        columtype_id = 'agilentw'
    else:
        columtype_id = get_setting('default_columtype_id')

    return get_columntype_from_id(columtype_id)

def get_default_num_plates():
    columntype = get_columntype()
    num_plates_pm = columntype.num_plates
    if num_plates_pm is None or np.isnan(num_plates_pm):
        num_plates_pm = DEFAULT_NUM_PLATES
    return num_plates_pm

def update_sec_settings_impl(el_option, column_name, exclusion_limit, num_plates_pm):
    logger = logging.getLogger(__name__)

    if False:
        # currently suppressed due to the following error when an "Agilent" column is selected
        # UnicodeEncodeError: 'cp932' codec can't encode character '\xc5' in position 10964: illegal multibyte sequence
        logger.info("SEC column name is %s", column_name)     

    columntype, columntype_id = get_columntype_from_name(column_name, with_id=True)

    if columntype is None:
        exclusion_limit = get_setting('exclusion_limit')
        mwrg_info = get_mwrg_info()
        poresize = mwrg_info.compute_rg(exclusion_limit)
    else:
        poresize = columntype.poresize
        if el_option == 2:
            exclusion_limit = exclusion_limit
        else:
            exclusion_limit = columntype.excl_limit

    set_setting('columntype_id', columntype_id)
    set_setting('exclusion_limit', exclusion_limit)
    if poresize is None or np.isnan(poresize):
        unknown = 'unknown '
        poresize = 200
        set_setting('poresize_bounds', (35, 300))
    else:
        unknown = ''
        set_setting('poresize_bounds', (poresize - PORESIZE_ALLOWANCE, poresize + PORESIZE_ALLOWANCE))
    set_setting('poresize', poresize)

    if num_plates_pm is None or np.isnan(num_plates_pm):
        num_plates_pm = DEFAULT_NUM_PLATES
    num_plates_pc = int(round(num_plates_pm * 0.3))
    set_setting('num_plates_pc', num_plates_pc)

    logger.info("%spore size is set to %.1f", unknown, poresize)