"""

    TrimmingInfo.py

    Copyright (c) 2016-2024, SAXS Team, KEK-PF
"""

TRIMMING_ITEMS = ["uv_restrict_list", "xr_restrict_list"]

class TrimmingInfo:
    def __init__(self, flag, start_, stop_, size_, extra=None):
        start = int(start_)     # int to avoid overspecification with numpy.int64
        stop = int(stop_)       # int to avoid overspecification with numpy.int64
        size = int(size_)       # int to avoid overspecification with numpy.int64
        self.flag   = flag
        self.start  = start
        self.end    = size - 1 if stop is None else stop - 1    # for plot (to be used)
        self.stop   = stop      # for slice
        self.size   = size
        self.extra  = extra
        self.items  = [flag, start, stop, size]

    def get_slice(self):
        return slice(self.start, self.stop)

    def get_safe_object(self):
        start_ = 0 if self.start is None else self.start
        stop_ = self.size if self.stop is None else self.stop
        return TrimmingInfo(self.flag, start_, stop_, self.size)

    def __repr__(self):
        return 'TrimmingInfo' + str( (self.flag, self.start, self.stop, self.size) )

    def __iter__(self):
        return iter([self.flag, self.start, self.stop, self.size])

    def __getitem__(self, item):
        return self.items[item]

def save_trimming_txt(path, info_list=None):
    if info_list is None:
        from molass_legacy._MOLASS.SerialSettings import get_setting
        info_list = [get_setting(item) for item in TRIMMING_ITEMS]
    fh = open(path, "w")
    for ti in info_list:
        fh.write(str(ti) + "\n")
    fh.close()

def load_trimming_txt(path):
    """
    trimming.txt example

    [TrimmingInfo(1, 656, 1107, 1600), None]
    [TrimmingInfo(1, 650, 1101, 1600), TrimmingInfo(1, 8, 1018, 1178)]

    """
    import numpy as np  # for eval
    fh = open(path)
    ret_list = []
    for line in fh:
        # print(line)
        line_list = eval(line, globals(), locals())
        ret_list.append(line_list)
    fh.close()
    return ret_list

def restore_trimming_info_impl(trimming_txt, logger=None):
    from molass_legacy._MOLASS.SerialSettings import set_setting

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    ret_list = load_trimming_txt(trimming_txt)
    for item, info in zip(TRIMMING_ITEMS, ret_list):
        set_setting(item, info)
        logger.info("%s restored as %s", item, str(info))

def get_trimming_info_list():
    from molass_legacy._MOLASS.SerialSettings import get_setting
    info_list = []
    for item in TRIMMING_ITEMS:
        info_list.append(get_setting(item))
    return info_list

def set_trimming_info_list(info_list, logger=None):
    from molass_legacy._MOLASS.SerialSettings import set_setting

    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    for item, info in zip(TRIMMING_ITEMS, info_list):
        set_setting(item, info)
        logger.info("%s restored as %s", item, str(info))

def get_trimming_slices_from_list(info_list):
    ret_slices = []
    for list_ in info_list:
        if list_ is None:
            start, stop = None, None
        else:
            start = list_.start
            stop = list_.stop
        ret_slices.append(slice(start, stop))
    return ret_slices

def get_trimming_ends_from_list(info_list):
    if info_list is None:
        return [(0, -1), (0, -1)]

    ret_ends = []
    for list_ in info_list:
        if list_ is None:
            start, end = 0, -1
        else:
            start = list_.start
            end = list_.end
        ret_ends.append((start, end))
    return ret_ends

def seem_to_be_manually_trimmed():
    from molass_legacy._MOLASS.SerialSettings import get_setting

    manually_trimmed = get_setting("manually_trimmed")
    return manually_trimmed

def get_mapped_info(mapping, size, tinfo):
    A, B = mapping
    start = max(0, int(round(tinfo.start*A + B)))
    stop = min(size, int(round(tinfo.stop*A + B)))
    return TrimmingInfo(1, start, stop, size)

def get_wider_info(tinfo1, tinfo2):
    assert tinfo1.size == tinfo2.size
    start = min(tinfo1.start, tinfo2.start)
    stop = max(tinfo1.stop, tinfo2.stop)
    return TrimmingInfo(1, start, stop, tinfo1.size)
