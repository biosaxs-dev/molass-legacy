"""

    ファイル名：   PilatusCounter.py

    処理内容：

       Pilatus 画像データ関連の共通処理

    Copyright (c) 2015-2024, Masatsuyo Takahashi, KEK-PF
"""
from __future__ import division, print_function, unicode_literals
    # /   を使うと商は float
    # //  を使うと商は int

import os
import sys
import glob
import re
import pandas as pd
from collections import defaultdict

class Counter:
    def __init__( self, in_folder ):
        self.in_folder = in_folder
        self.counter_tables = []

        orig_folder = os.getcwd()
        os.chdir( self.in_folder )
        counter_files   = glob.glob( 'PilatusCounter_*.txt' )

        # print( " counter_files=", counter_files )

        counter_dict = {}
        for file in counter_files:
            counter_table = pd.read_table( file )
            self.counter_tables.append( counter_table )

        os.chdir( orig_folder )

    def get_counter_dict( self, counter_id ):
        counter_dict = {}

        for counter_table in self.counter_tables:
            if counter_id == 'None':
                counter = None
            else:
                counter = counter_table[ counter_id ]       # counter_id の列を選ぶ。
            for i in range( counter_table.File.size ):
                # 'test001_0_d0_00000.tif' などの場合、末尾の拡張子を削除する。
                file_id = re.sub( r'\.\w+$', '', counter_table.File[i] )

                # 'MAG2wk7210_0_00000-0' などの '_00000-0' または '_00000' を削除する。
                # なお、枝番（-0）はある時期に一時的に存在したもので、
                # 最新の状態では存在しないものである。
                # fkey = re.sub( r'_\d+(-\d+)?$', '', file_id )

                fkey = file_id

                if counter_id == 'None':
                    count = 1
                else:
                    count = counter[i]
                counter_dict[ fkey ] = count

        return counter_dict

    def available_counters( self ):
        available_keys = defaultdict( lambda: 0 )
        for counter_table in self.counter_tables:
            for colname, series in counter_table.iteritems():
                if colname[0] != 'C':
                    continue
                # print colname, series
                available = True
                for num in series:
                    if num <= 0:
                        available = False
                        break
                if available:
                    available_keys[ colname ] += 1

        return sorted( available_keys.keys() )
