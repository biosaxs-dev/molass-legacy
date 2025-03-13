"""

    ファイル名：   PilatusUtilsNewStyle.py

    処理内容：

       Pilatus 画像データ関連の共通処理

    Copyright (c) 2015-2025, SAXS Team, KEK-PF
"""
import os
import re
from PilatusUtilsOldStyle import get_prefix_info_list, get_dict_info

file_name_re = re.compile(r'(\w+)_(d\d)_(\d{5})\.(\w{3})')

def get_data_info( folder_info, log_file, mask_file,
                    adj_folder, syn_folder, pilatus_counter, counter_id,
                    log_file_path,
                    sample_complete=False, for_test_data=False,
                    logger=None, debug=False ):

    in_folder = folder_info.path
    counter_dict, org_file_dict, adj_file_dict, syn_file_dict = get_dict_info( in_folder, adj_folder, syn_folder, pilatus_counter, counter_id )
    # print('counter_dict=', counter_dict)

    image_files = folder_info.image_files

    # print('len(image_files)=', len(image_files))
    prefix_info_list, text_dict = get_prefix_info_list( in_folder, log_file_path )
    prefix_dict = {}
    for info in prefix_info_list:
        # print(info[0])
        prefix_dict[info[0]] = info

    single_image = folder_info.single_image

    exec_dict = {}
    num_chages_dict = {}

    def get_prefix_info(f):
        parts = f.split('_')
        info = None
        while info is None:
            f_ = '_'.join(parts)
            info = prefix_dict.get(f_)
            if info is not None or len(parts) == 1:
                break
            parts.pop(-1)
        return info

    for k, path in enumerate(sorted(image_files)):
        folder, file= os.path.split(path)
        m = file_name_re.match(file)
        if not m:
            continue

        # print([k], file)
        f1 = m.group(1)
        f2 = m.group(2)
        f3 = m.group(3)

        prefix_info = get_prefix_info(f1)
        if prefix_info is None:
            continue

        num_changes = prefix_info[2]
        exec_date = prefix_info[-1]
        data_dict = exec_dict.get(exec_date)
        if data_dict is None:
            exec_dict[exec_date] = data_dict = {}

        if single_image:
            key = f1
        else:
            key = '_'.join([f1,f3])
        num_chages_dict[key] = num_changes
        rec = data_dict.get(key)
        if rec is None:
            data_dict[key] = rec = []

        pos_info = prefix_info[3]
        if len(pos_info) == 0:
            print("skipped %s, which has no position change info." % file)
            continue

        fkey = '_'.join([f1,f2,f3])
        count = counter_dict.get(fkey)
        if count is None:
            if logger is None:
                import logging
                logger = logging.getLogger(__name__)
            logger.error("No counter info for %s.", fkey)
            continue

        orig_key = re.sub(r'_d\d_', '_d0_', fkey)
        orig_count = counter_dict.get(orig_key)
        ratio = count/orig_count

        if debug:
            if k < 12:
                print([k], fkey, orig_count, count, ratio)

        adj_file = adj_file_dict.get( key )
        syn_file = syn_file_dict.get( key )
        i = len(rec)
        rec.append([file, pos_info[i], ratio, adj_file, syn_file])

    def alphanum_key(pair):
        ret_key = [int(c) if c.isdecimal() else c for c in re.split('([0-9]+)', pair[0])]
        return ret_key, pair[1]

    data_array = []
    for edate, data_dict in sorted(exec_dict.items()):
        for key, value in sorted(data_dict.items(), key=alphanum_key):
            data_array.append([key, value, num_chages_dict[key]])

    if for_test_data:
        assert False
    else:
        return log_file, mask_file, data_array, pilatus_counter
