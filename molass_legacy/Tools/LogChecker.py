"""
    LogChecker.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import sys
import glob
import re

started_log_re = re.compile(r"started loading (\S+)")
yyyymmdd_re = re.compile(r"^(\d{8}).*")

def check_logfiles(log_root):
    num_files = 0
    num_normals = 0
    num_errors = 0
    first_yyyymmdd = None
    last_yyyymmdd = None
    for file in glob.glob(log_root + "/*/*.log"):
        num_files += 1

        # print(file)
        found_started = False
        found_finished = False
        with open(file) as fh:
            for n, line in enumerate(fh):
                if not found_started:
                    m = started_log_re.search(line)
                    if m:
                        for node in m.group(1).split("/"):
                            m_ = yyyymmdd_re.match(node)
                            if m_:
                                yyyymmdd = m_.group(1)
                                if first_yyyymmdd is None:
                                    first_yyyymmdd = yyyymmdd
                                last_yyyymmdd = yyyymmdd

                if line.find("Finished.") > 0:
                    found_finished = True

        if found_finished:
            print(file, "ok")
            num_normals += 1
        else:
            print(file, "something wrong")
            num_errors += 1

    print("num_files     %4d     %s ～ %s" % (num_files, first_yyyymmdd, last_yyyymmdd))
    print("num_normals   %4d" % num_normals)
    print("num_errors    %4d" % num_errors)

if __name__ == '__main__':
    assert len(sys.argv) > 1
    check_logfiles(sys.argv[1])
