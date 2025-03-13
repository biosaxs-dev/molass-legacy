# coding: utf-8
"""

    ファイル名：    AppVersion.py

    処理内容：      アプリケーションのバージョン情報


"""
import platform

def synthesizer_version_string():
    return 'Synthesizer 0.4.6 (2025-03-12 python %s %s)' % ( platform.python_version(), platform.architecture()[0] )
