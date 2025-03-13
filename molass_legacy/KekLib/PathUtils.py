"""
    PathUtils.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import os
import re

class NonExistingPath:
    def __init__(self, path):
        while os.path.exists(path):
            path = re.sub(r"^([^\(\)]+\S)(\s+)?(\(\d+\))?$", self.new_name, path)

        self._path = path

    def new_name(self, m):
        m3 = m.group(3)
        if m3 is None:
            m3 = "0"
        else:
            m3 = m3[1:-1]
        return m.group(1) + " " + "(%d)" % (int(m3)+1)

    def __str__(self):
        return self._path
