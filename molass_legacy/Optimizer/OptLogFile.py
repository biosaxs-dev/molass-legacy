"""
    Optimizer.OptLogFile.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import re

class OptLogFile:
    def __init__(self, path):
        from molass_legacy.Optimizer.StrategicOptimizer import LOG_IDENTIFICATION

        version_re = re.compile(r"\((\d+-\d+-\d+)")
        optlist_re = re.compile(r"optlist=(\[[^\]]+\])")
        active_indices_re = re.compile(r"indeces=(\[[^\]]+\])")

        self.active_indeces = None
        optlist = None
        version_date = None
        
        with open(path) as fh:
            for k, line in enumerate(fh.readlines()):
                if line.find("_MOLASS") > 0:
                    m = version_re.search(line)
                    if m:
                        version_date = m.group(1)
                    continue

                if line.find("started with") > 0:
                    m = optlist_re.search(line)
                    if m:
                        optlist = eval(m.group(1))
                    continue

                if line.find(LOG_IDENTIFICATION) > 0:
                    # note that use of Strategic Optimizer is optional
                    m = active_indices_re.search(line)
                    if m:
                        self.active_indeces = eval(m.group(1))
                    break

        assert optlist is not None

        self.version_date = "2022-09-16" if version_date is None else version_date
        self.optdict = dict(optlist)
        self.class_code = self.optdict['-c']
        self.nc = int(self.optdict['-n'])

    def get_active_indeces(self):
        if self.active_indeces is None:
            self.active_indeces = []
        return self.active_indeces