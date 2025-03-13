# coding: utf-8
"""
    NoseUtils.py

    Copyright (c) 2019-2022, Masatsuyo Takahashi, KEK-PF
"""

import sys
import re
import inspect
import traceback
import logging
import numpy as np

filepath_re = re.compile(r'File "(.+)"')
eq_call_re = re.compile(r'^(\s+(eq_|gt_)\(\s*[^,]+\s*,\s*)(.*\S)(\s*\)\s*)$')
assert_almost_equal_re = re.compile(r"^(\s+assert_almost_equal\([^,]+,\s*)([\d\.e\+\-]+)(.+)$")

success_result_dict = {}

def eq_(lhs, rhs):
    lineno = traceback.extract_stack()[2].lineno
    # print(lineno)
    # print(lhs, rhs)
    success_result_dict[lineno] = ('eq_', lhs, rhs)

def gt_(lhs, rhs):
    lineno = traceback.extract_stack()[2].lineno
    # print(lineno)
    # print(lhs, rhs)
    success_result_dict[lineno] = ('gt_', lhs, rhs)

def assert_almost_equal(lhs, rhs, n=None):
    lineno = traceback.extract_stack()[2].lineno
    success_result_dict[lineno] = ('assert_almost_equal', lhs, rhs)

def str_for_source_code(value):
    if isinstance(value, (str, Exception)):
        str_value = str(value)
        if str_value.find("'") >= 0:
            q = '"'
        else:
            q = "'"

        return q + str_value + q
    else:
        return str(value)

def write_the_successful_script(newline='\n', kwargs={}):
    # get the caller module
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])

    func_list = []
    for name, func in inspect.getmembers(module):
        if name.find('test') == 0:
            print(name)
            num = inspect.getsourcelines(func)[-1]
            # print(first_line_info)
            # num = first_line_info[1]
            func_list.append( (num, name, func) )

    for num, name, func in sorted(func_list):
        print([num], "testing", name)
        try:
            func(**kwargs)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            print('%d %s call failed:\n' % (num, name), etb)

    # print('success_result_dict=', success_result_dict)

    in_filepath = module.__file__
    out_filepath = in_filepath.replace('.py', '-success.py')

    out_lines = []
    with open(in_filepath) as fh:
        for n, line in enumerate(fh.readlines(), start=1):
            success_rec = success_result_dict.get(n)
            if success_rec is None:
                out_line = line
            else:
                method, lhs, rhs = success_rec
                # assert method in ['eq_', 'gt_']
                if method == 'eq_':
                    out_line = re.sub(eq_call_re, lambda m: m.group(1) + str_for_source_code(lhs) + m.group(4), line)
                elif method == 'gt_':
                    rhs_ = min(rhs, lhs)
                    delta = np.power(10, np.floor(np.log10(rhs_)-2))
                    less_valuse = '%.3g' % (rhs_ - delta)
                    out_line = re.sub(eq_call_re, lambda m: m.group(1) + less_valuse + m.group(4), line)
                elif method == 'assert_almost_equal':
                    lhs_str = "%g" % lhs
                    out_line = re.sub(assert_almost_equal_re, lambda m: m.group(1) + lhs_str + m.group(3), line)
                else:
                    assert False
            out_lines.append(out_line)

    with open(out_filepath, 'w', newline=newline) as fh:
        fh.write( ''.join(out_lines) )

"""
    The following are dummy objects to replace ChangeableLogger instances
"""

class Logger(logging.Logger):
    def __init__(self, path):
        logging.Logger.__init__(self, __name__)

def create_our_logger(path):
    return Logger(path)
