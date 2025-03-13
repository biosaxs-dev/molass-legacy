# coding: utf-8
"""

    ファイル名：   CommandLineOptions.py

    処理内容：

        コマンドライン・オプションの取得

"""
import sys

from optparse import OptionParser

usage = "usage: autorg_kek [options] <datafile(s)>"

parser = OptionParser( usage=usage )

parser.add_option('', '--mininterval', action='store', default=3,
                help="Minimum acceptable Guinier interval length in points. (default: 3)")

parser.add_option('', '--smaxrg', action='store', default=1.3,
                help="Maximum acceptable SmaxRg value. (default: 1.3)")

parser.add_option('', '--sminrg', action='store', 
                help="Minimum acceptable SminRg value. (default unlimited)")

parser.add_option('-o', '--output', action='store',
                help="Relative or absolute path to save result" )

parser.add_option('-f', '--format', action='store',
                help="Output format, either of: csv, ssv")

parser.add_option('-A', '--ATSAS', action='store_true', 
                help="output ATSAS autorg columns only")

parser.add_option('-r', '--robust', action='store_true',
                help="output result even if it is quite bad")

parser.add_option('-S', '--server', action='store_true',
                help="run as a server")

parser.add_option('-v', '--version', action='store_true', 
                help="show the appication's version info")
