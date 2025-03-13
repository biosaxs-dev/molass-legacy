# coding: utf-8
"""
    CallStack.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import re
import os
import traceback

class CallStack:
    def __init__(self, terse=True):
        stack = traceback.format_stack()[:-2]
        if terse:
            c_seq = []
            file_re = re.compile(r'File "(.+)"')
            for line in stack:
                c_seq.append( re.sub(file_re, lambda m: 'File "' + os.path.split(m.group(1))[1] + '"', line) )
            self.c_seq = c_seq
        else:
            self.c_seq = stack

    def __str__(self):
        return ''.join(self.c_seq)

    def __repr__(self):
        return self.__str__()
