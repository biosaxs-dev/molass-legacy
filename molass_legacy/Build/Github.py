# coding: utf-8
"""
    Github.py

    Copyright (c) 2020-2021, Masatsuyo Takahashi, KEK-PF
"""
import os
import re
import subprocess

def where(name):
    result = subprocess.run(
            [r'C:\Windows\System32\where.exe', name],
            capture_output=True,
            )
    return result.stdout[:-2].decode()  # [:-2] is to remove \r\n

class Github:
    def __init__(self):
        self.git_exe = where('git')

    def git(self, *args):
        result = subprocess.run(
                [self.git_exe] + list(args),
                capture_output=True,
                )
        return result

    def clone(self, url_git):
        self.git('clone', url_git)

gh = None

def get_gh():
    global gh
    if gh is None:
        gh = Github()
    return gh

def download_pytweening(folder):
    if os.path.exists(folder):
        print('already exists')
    else:
        gh = get_gh()
        url = 'https://github.com/asweigart/pytweening.git'
        print('git clone ' + url)
        gh.clone(url)
        if os.path.exists(folder):
            print('ok')

def edit_pytweening():
    src = open('setup.py').read()
    new_src = re.sub(r"version=.+,", "version='1.0.3',", src)

    new_file = 'setup-modified.py'
    with open(new_file, 'w') as fh:
        fh.write(new_src)
    return new_file

def run_setup_install(python_exe, setup_py):
    print('%s setup.py install' % python_exe)
    result = subprocess.run(
            [python_exe, setup_py, 'install'],
            capture_output=True,
            )
    return result

EMBEDDABLES_PYTHON_EXE = r'..\embeddables\python.exe'

def install_pytweening():
    folder = 'pytweening'
    download_pytweening(folder)
    cwd = os.getcwd()
    os.chdir(folder)
    new_file = edit_pytweening()
    result = run_setup_install(EMBEDDABLES_PYTHON_EXE, new_file)
    # print(result)
    assert result.returncode == 0
    print('successfully installed pytweening')
    os.chdir(cwd)
    print(os.getcwd())

# METHODS_DICT = {'pytweening':install_pytweening}  # no longer required as of 2022-01-18
METHODS_DICT = {}

def get_method(k):
    return METHODS_DICT.get(k)
