# -*- coding: utf-8 -*-
"""

    ファイル名：   CommandLineOptions.py

    処理内容：

        コマンドライン・オプションの取得

"""
from __future__ import division, print_function, unicode_literals
import sys

from optparse import OptionParser

p = OptionParser()

# required
p.add_option('-c', '--command', action='store_true',
                help="run in command mode; default is GUI mode")

p.add_option('-i', '--in-folder', action='store', dest='in_folder',
                help="specify input image folder; only required option in command mode")

# optinal
p.add_option('-o', '--out-folder', action='store', dest='out_folder',
                help="specify output image folder; default is IN_FOLDER/Synthesized")

p.add_option('-n', '--autonum-folders', action='store_true', dest='autonum_folders',
                help="create automatically numbered output folder if default output folder exists")

p.add_option('-j', '--adj-folder', action='store', dest='adj_folder',
                help="specify adjusted image folder; default is no adjusted image output")

p.add_option('-m', '--intermediate-results', action='store_true', dest='intermediate_results',
                help="write intermediate results such as *_1_sun.tif")

p.add_option('-p', '--pandastable', action='store_true', 
                help="use pandastable in GUI")

p.add_option('-v', '--version', action='store_true', 
                help="show the appication's version info")

opts, args = p.parse_args()
