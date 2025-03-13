"""
    Optimizer.FullOptUtils.py

    Copyright (c) 2021-2024, SAXS Team, KEK-PF
"""
import re

def get_class_code_from_logfile(path):
    from .OptLogFile import OptLogFile
    log_file = OptLogFile(path)
    return log_file.class_code

def get_in_data_folder(in_data_info_txt):
    data_folder = None
    with open(in_data_info_txt) as fh:
        in_folder_re = re.compile("in_folder=(\S+)\s*")
        for line in fh:
            m = in_folder_re.match(line)
            if m:
                data_folder = m.group(1)
                break
    return data_folder

def in_folder_consistency_ok(in_folder, in_data_info_txt, parent):
    data_folder = get_in_data_folder(in_data_info_txt)

    folder_a = "/".join(in_folder.replace("\\", "/").split("/")[-2:])
    folder_b = "/".join(data_folder.replace("\\", "/").split("/")[-2:])
    ok = folder_a == folder_b
    if not ok:
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        yn = MessageBox.askyesno("Inconsistency Notification",
            "Input folders do not seem consistent as shown below.\n"
            "   Main Window Input: %s\n"
            "   Job  Result Input: %s\n"
            "The process can be unreliable.\n"
            "Would you like to proceed anyway?" % (folder_a, folder_b),
            parent=parent)
        if yn:
            ok = True
    return ok
