"""
    Experiment.ColumnTypes.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import numpy as np

class ColumnType:
    def __init__(self, **entries): 
        self.__dict__.update(entries)

    def __str__(self):
        return str(self.__dict__)

COLUMN_TYPE_DICT = {
    "ad75w"   : ColumnType( name="Superdex 75 increase 10/300", shortname="Superdex 75-10",
                        size=(10,  300), excl_limit=100, frac_range=(3, 70), num_plates=43000, poresize=35.6),

    "ad75n"   : ColumnType( name="Superdex 75 increase 3.2/300", shortname="Superdex 75-3.2",
                        size=(3.2, 300), excl_limit=100, frac_range=(3, 70), num_plates=43000, poresize=35.6),

    "ad200w"  : ColumnType( name="Superdex 200 increase 10/300", shortname="Superdex 200-10",
                        size=(10,  300), excl_limit=1300, frac_range=(10, 600), num_plates=48000, poresize=76.0),

    "ad200n"  : ColumnType( name="Superdex 200 increase 3.2/300", shortname="Superdex 200-3.2",
                        size=(3.2, 300), excl_limit=1300, frac_range=(10, 600), num_plates=48000, poresize=76.0),

    "ad6w"    : ColumnType( name="Superdex 6 increase 10/300", shortname="Superdex 6-10",
                        size=(3.2, 300), excl_limit=40000, frac_range=(5, 5000), num_plates=48000, poresize=209),

    "nacalai" : ColumnType( name="COSMOSIL 5Diol-300-II (Nacalai)", shortname="COSMOSIL",
                        size=(7.5, 300), excl_limit=135000, frac_range=(10, 700), num_plates=np.nan, poresize=300),

    "agilentn": ColumnType( name="Bio SEC-3 (Φ4.6×300, 300Å) (Agilent)", shortname="Bio SEC-3 (4.6)",
                        size=(4.6, 300), excl_limit=135000, frac_range=(5, 1250), num_plates=np.nan, poresize=300),

    "agilentw": ColumnType( name="Bio SEC-3 (Φ7.8×300, 300Å) (Agilent)", shortname="Bio SEC-3 (7.8)",
                        size=(7.8, 300), excl_limit=135000, frac_range=(5, 1250), num_plates=np.nan, poresize=300),

    "shodex"  : ColumnType( name="KW403-4F（Shodex）", shortname="KW403-4F",
                        size=(4.6, 300), excl_limit=600, frac_range=(10, 500), num_plates=35000, poresize=60.4),

    "unknown" : ColumnType( name="Unknown", shortname="Unknown",
                        size=(np.nan, np.nan), excl_limit=np.nan, frac_range=(np.nan, np.nan), num_plates=np.nan, poresize=np.nan),
}

from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

PORESIZE_ALLOWANCE = 5

names = []
def get_productnames():
    if len(names) == 0:
        for k, v in COLUMN_TYPE_DICT.items():
            names.append(v.name)
    return names

def get_columntype_from_id(id_):
    return COLUMN_TYPE_DICT.get(id_)

def get_columntype_from_name(name, with_id=False):
    coltype = None
    for k, v in COLUMN_TYPE_DICT.items():
        if v.name == name:
            coltype = v
            break
    if with_id:
        return coltype, k
    else:
        return coltype

def get_columntype_id_from_name(name):
    coltype_id = None
    for k, v in COLUMN_TYPE_DICT.items():
        if v.name == name:
            coltype_id = k
            break
    return coltype_id

def update_sec_settings_by_id(columntype_id):
    columntype = COLUMN_TYPE_DICT[columntype_id]

    set_setting('columntype_id', columntype_id)
    set_setting('exclusion_limit', columntype.excl_limit)
    poresize = columntype.poresize
    set_setting('poresize', poresize)
    set_setting('poresize_bounds', (poresize - PORESIZE_ALLOWANCE, poresize + PORESIZE_ALLOWANCE))

def get_columnname():
    columntype_id = get_setting('columntype_id')
    return COLUMN_TYPE_DICT[columntype_id].shortname

def get_all_poresizes():
    poresize_list = []
    for k, v in COLUMN_TYPE_DICT.items():
        poresize_list.append(v.poresize)
    return sorted(list(set(poresize_list)))     # remove duplicates
