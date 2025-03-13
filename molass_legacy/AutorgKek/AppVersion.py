# coding: utf-8
"""

    ファイル名：    AppVersion.py

    処理内容：      アプリケーションのバージョン情報


"""
from __future__ import division, print_function, unicode_literals

import platform

def get_com_version_string():
    return 'autorg_kek 0.7.1 (2019-04-18 python %s %s)' % ( platform.python_version(), platform.architecture()[0] )

def autoguinier_version_for_publication():
    import re
    version = get_com_version_string().replace("autorg_kek", "AutoGuinier")
    return re.sub(r"\s+\(.+", "", version)
