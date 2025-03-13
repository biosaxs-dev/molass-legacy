# -*- coding: utf-8 -*-
"""

    ファイル名：   SAnglerMask.py

    処理内容：

        SAngler マスクファイルのオブジェクト化。

"""
from __future__ import unicode_literals
import re
import numpy as np

comment_line_re = re.compile( '^#\s+(\w+)=(.+)' )

class SAnglerMask():
    def __init__( self, filename ):

        width, height   = 0, 0
        self.mask_array = []

        # print 'DEBUG: SAnglerMask(%s)' % ( filename )

        fh = open( filename )
        for line in fh.readlines():
            m1 = comment_line_re.match( line )
            if m1:
                key_word = m1.group( 1 )
                value    = m1.group( 2 )
                if key_word == 'Width':
                    width = int( value )
                elif key_word == 'Height':
                    height = int( value )
                continue

            if width == 0 or height == 0:
                raise Exception( 'Wrong format' )
            else:
                if len( self.mask_array ) == 0:
                    self.mask_array = np.zeros( [ height, width ] )

            for pair in line.split("\t"):
                (c, r) = pair.split()
                self.mask_array[ int(r), int(c) ] = 1

        fh.close()
        if width == 0 or height == 0:
            # ここは空ファイルなどが該当する。
            # マスクすべきピクセルが無い場合はありえても、Width と Height は必須とする。
            raise Exception( 'Wrong format' )
