# coding: utf-8
"""
    ファイル名：   OurImageIO.py

    処理内容：

        画像ファイルの種類による差異を吸収するための、画像入出力クラス

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF
"""

import re
from MinimalTiff        import MinimalTiff
import fabio

extension_dict = {
    'tiff'  :   'tif',
    }

extension_re = re.compile( '\.(\w+)$' )

class Image:
    def __init__( self, path ):
        m = extension_re.search( path )
        if m:
            ext_ = m.group( 1 ).lower()
        else:
            assert( False )

        if len( ext_ ) != 3:
            ext_  = extension_dict[ ext_ ]

        self.__ext = ext_

        if ext_ == 'tif':
            self.__image    = MinimalTiff( path )
            self.__header   = self.__image.header
            self.__data     = self.__image.data
        elif ext_ == 'cbf':
            self.__image    = fabio.open( path )
            self.__data     = self.__image.data
        else:
            assert( False )

    def get_header( self )  : return self.__header
    def set_header( self )  : assert( False )
    header  = property( get_header, set_header )

    def get_data( self )    : return self.__data
    def set_data( self, data, force=False ):
        self.__data = data
        self.__image.data = data
    data    = property( get_data, set_data )

    def get_ext( self )     : return self.__ext
    def set_ext( self )     : assert( False )
    ext     = property( get_ext, set_ext )

    def save( self, path ):
        if self.__ext == 'tif':
            self.__image.save( path )
        elif self.__ext  == 'cbf':
            self.__image.save( path )
        else:
            assert( False )
