# coding: utf-8
"""
    DevelEnv.py

    Copyright (c) 2020-2022, Masatsuyo Takahashi, KEK-PF
"""
import ctypes
assert ctypes.windll.shell32.IsUserAnAdmin()
import sys
import os
import platform
from StdoutLogger import StdoutLogger
from .Embeddables import Embeddables

class DevelEnv(Embeddables):
    def __init__(self):
        self.logger = StdoutLogger("build.log")
        self.python_exe = sys.executable
        self.python00 = 'python' + ''.join(platform.python_version().split('.')[0:2])
        self.emb_folder, _ = os.path.split(self.python_exe)
        self.site_folder = os.path.join(self.emb_folder, r'Lib\site-packages')
        self.install_required_minimum()

    def build(self, stable=True):
        self.prepare_unofficial_site()
        self.install_requirements("requirements.txt")
