# coding: utf-8
"""
    ArgParser.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""

class ArgParser:
    def __init__(self, argv):
        self.argv = argv

    def get_dict(self):
        ret = {}
        for v in self.argv[1:]:
            k, p = v.split('=')
            key = k.replace('--', '')
            val = int(p) if p.isnumeric() else p
            ret[key] = val

        return ret
