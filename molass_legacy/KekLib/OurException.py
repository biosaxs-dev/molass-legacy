# -*- coding: utf-8 -*-
"""

    ファイル名：   OurException.py

    処理内容：

        例外処理

"""
from __future__ import division, print_function, unicode_literals

class OurException( Exception ):
    def __init__( self, value ):
        self.value   = value

    def __str__( self ):
        return repr( self.value )
