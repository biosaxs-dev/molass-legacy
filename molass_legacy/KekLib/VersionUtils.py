"""
    VersionUtils.py

    Copyright (c) 2024, Masatsuyo Takahashi, KEK-PF
"""
import re
import os
from subprocess import run
from datetime import datetime

def get_exe_version(exe_path):
    ret = run([exe_path, '-V'], capture_output=True)
    return [int(v) for v in ret.stdout.decode()[:-1].split(" ")[-1].split('.')]

def update_molass_version_file(version_file, with_time=False):
    dist_folder = version_file.split('\\')[-4]
    version  = dist_folder.split('-')[-2].replace('_', '.')
    version_re = re.compile(r"_MOLASS .+ python")
    temp_file = version_file + ".temp"
    out_fh = open(temp_file, "w")
    with open(version_file) as in_fh:
        for line in in_fh:
            if line.find('_MOLASS') > 0:
                format = "%Y-%m-%d"
                if with_time:
                    format += " %H:%M:%S"
                line = re.sub(version_re, "_MOLASS %s (%s python" % (version, datetime.today().strftime(format)), line)
            out_fh.write(line)
    out_fh.close()
    os.remove(version_file)
    os.rename(temp_file, version_file)

if __name__ == '__main__':
    import sys
    print(get_exe_version(sys.executable))