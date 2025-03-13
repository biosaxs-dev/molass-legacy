# coding: utf-8
"""
    OurPipApi.py

    Copyright (c) 2018-2019, Masatsuyo Takahashi, KEK-PF
"""

import re
import fnmatch
import os
import subprocess
from packaging import version
from pip_api import installed_distributions
# from pip_api._call import call
from Console import reset_text_color

def decode_robust(b):
    try:
        s = b.decode('utf-8', errors='ignore')
    except Exception as exc:
        s = str(exc)
    return s

def call(*args):
    python_location = os.environ.get('PIPAPI_PYTHON_LOCATION', 'python')
    result = subprocess.check_output(
        [python_location, '-m', 'pip'] + list(args)
    )
    return decode_robust(result)

shown_line_re = re.compile(r'^(\S+):\s*(.*)$')
searched_line_re = re.compile(r'^(\S+)')
installed_line_re = re.compile(r'^\s+INSTALLED:\s+(\S+)\s*(\(latest\))?')
latest_line_re = re.compile(r'^\s+LATEST:\s+(.+)')

def my_print(*args):
    global _logger
    if _logger is None:
        print(*args)
    else:
        msg = ' '.join( [str(v) for v in args] )
        _logger.info(msg)

def update_all_packages(restrict=None, exec_install=False, logger=None):
    global _logger
    _logger = logger
    if restrict is None:
        restrict_re = None
    else:
        restrict_re = re.compile(fnmatch.translate(restrict))

    for k, v in installed_distributions().items():
        if k[0] == '-':
            my_print(k, "seems to have been erroneously installed.")
            continue

        if restrict_re is None:
            pass
        else:
            if restrict_re.match(k):
                pass
            else:
                continue
        reset_text_color()
        my_print(k)
        shown = call('show', k)
        # my_print(shown)
        for line in shown.split('\r\n'):
            m = shown_line_re.match(line)
            if m:
                key = m.group(1)
                value = m.group(2)
                if key == 'Name':
                    assert k == value
                    # my_print("'%s'" % value)
                elif key == 'Version':
                    my_print(k, value)
        try:
            searched = call('search', k)
        except Exception as exc:
            my_print(exc)
            my_print('pip search failed for', k)
            searched = ''
        # my_print(searched)
        this_line_no = None
        verified = False
        for n, line in enumerate(searched.split('\r\n')):
            m = searched_line_re.search(line)
            if m:
                name = m.group(1)
                if name == k:
                    # my_print(line)
                    this_line_no = n
            if this_line_no is not None and n > this_line_no and n < this_line_no + 3:
                my_print(line)
                m = installed_line_re.match(line)
                if m:
                    installed_version = m.group(1)
                    installed_latest = m.group(2)
                    # my_print(installed_version, installed_latest)
                    if installed_latest is not None:
                        verified = True
                        break
                m = latest_line_re.match(line)
                if m:
                    latest_version = m.group(1)
                    my_print(installed_version, latest_version)
                    if version.parse(installed_version) < version.parse(latest_version):
                        my_print('need update')
                        if exec_install:
                            my_print('pip install %s -U' % k)
                            try:
                                install_log = call('install', k, '-U')
                                my_print(install_log)
                                if install_log.lower().find('error') > 0:
                                    exit()
                            except Exception as exc:
                                my_print(exc)
                                my_print('pip install failed for', k)
                    else:
                        my_print('no need')
                    verified = True
                    break

        # assert verified
        if not verified:
            my_print( k, 'is not verified. need to update manually.' )
