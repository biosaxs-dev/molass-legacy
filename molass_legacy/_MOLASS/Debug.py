# coding: utf-8
"""
    Debug.py

    Copyright (c) 2020-2022, SAXS Team, KEK-PF
"""

import os

def debug_log_path():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(this_dir + '/../../log/debug.log')

class DebugInfo:
    def __init__(self, debug=0):
        path = debug_log_path()
        self.debug = debug
        if self.debug:
            import logging
            self.logger = logging.getLogger()
            self.formatter = logging.Formatter('%(asctime)s %(message)s', '%Y-%m-%d %H:%M:%S')
            self.fileh = logging.FileHandler(path, 'w')
            self.fileh.setFormatter(self.formatter)
            self.logger.addHandler(self.fileh)
        else:
            self.logger = None

MOLASS_DEBUG_ARGS = "MOLASS_DEBUG_ARGS"

def set_debug_args(args):
    os.environ[MOLASS_DEBUG_ARGS] = args

class DebugArgs:
    def __init__(self):
        args = os.environ.get(MOLASS_DEBUG_ARGS)
        if args is not None:
            self.parse(args)
        self.args = args

    def is_active(self):
        return self.args is not None

    def parse(self, args):
        import re
        arg_re = re.compile(r"\s*(\w+)\s*=\s*(\S+)\s*")
        for term in args.split(','):
            m = arg_re.match(term)
            if m:
                name = m.group(1)
                value = eval(m.group(2))
                # print(name, value)
                self.__setattr__(name, value)
