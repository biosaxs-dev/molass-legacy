# coding: utf-8
"""

    ファイル名：   PilatusImage.py

    処理内容：

        シフトとマスクを考慮してピクセルの値を参照する。

    Copyright (c) 2015-2019, SAXS Team, KEK-PF
"""
from __future__ import unicode_literals
from __future__ import division
    # /   を使うと商は float
    # //  を使うと商は int
import re
from math import sqrt
import numpy as np
import pandas as pd
from OurImageIO             import Image
from SAnglerMask            import SAnglerMask
from Development            import get_devel_info
from PilatusImageProperty   import get_tiffinfo

size_of_pixel       = 172
half_size_of_pixel  = size_of_pixel // 2
comment_line_re = re.compile( '^#\s+(\w+)=(.+)' )

DEBUG = True
INVALID_VALUE = -3

def pixel_rounded_shift( delta ):
    i_ = delta // size_of_pixel
    r_ = delta %  size_of_pixel
    if r_ >= half_size_of_pixel:
        i_ += 1
    return i_

class PilatusImage:
    def __init__( self, file, mask_data='', dy=0, dx=0, i_ratio=1.0, a_ratio=0.5, original_image=None ):

        # print( 'PilatusImage: file=%s, dy=%d, dx=%d' % ( file, dy, dx ) )
        self.image      = None
        self.tiffinfo   = None

        if type(file) != np.ndarray:
            self.image      = Image( file )
            # self.tiffinfo = get_tiffinfo( im )
            self.imarray    = np.array(self.image.data)     # self.imarray.flags.writeable = True
        else:
            assert( original_image is not None )
            self.image      = original_image.image
            self.imarray    = file

        if type( mask_data ) == type( self.imarray ):
            self.mask_array = mask_data
            self.has_mask = True
        elif mask_data != '' and mask_data != None:
            self.set_mask( mask_data )
            self.has_mask = True
        else:
            self.mask_array = None
            self.has_mask = False

        if self.has_mask:
            assert self.imarray.shape == self.mask_array.shape

        self.shifted = not ( dy==0 and dx==0 )
        if self.shifted:
            self.ir = dy // size_of_pixel
            self.dr = dy %  size_of_pixel
            self.ic = dx // size_of_pixel
            self.dc = dx %  size_of_pixel
            # print 'ir=%d, dr=%d, ic=%d, dc=%d' % ( self.ir, self.dr, self.ic, self.dc )
            # pixel 単位に丸める。
            self.ir_rounded = pixel_rounded_shift( dy )
            self.ic_rounded = pixel_rounded_shift( dx )

        self.i_ratio = i_ratio

        if self.has_mask or self.shifted:
            algo = get_devel_info('adj_algorithm')
            # print( 'algo=', algo )
            if algo == 'round':
                self.round_adjust()
            elif algo == 'fast':
                self.fast_adjust( a_ratio )
            else:
                self.adjust( a_ratio )

    def image_array( self ):
        return self.imarray

    def set_mask( self, maskfile ):
        print( 'mask_array=', mask_array )
        mask_object = SAnglerMask( maskfile )
        self.mask_array = mask_object.mask_array

    def compute_cratios( self ):
        self.cratios = []
        for p, q in ( [0,0], [0,1], [1,0], [1,1] ):
            v_prop  = ( (1-p) * (size_of_pixel - self.dr) + p * self.dr ) / size_of_pixel
            h_prop  = ( (1-q) * (size_of_pixel - self.dc) + q * self.dc ) / size_of_pixel
            self.cratios.append( v_prop * h_prop )

    # --------------------------------------------------------------------------
    #   位置調整画像作成（低速版）
    # --------------------------------------------------------------------------
    def adjust( self, a_ratio ):
        if self.shifted:
            self.compute_cratios()

        imarray_ = np.zeros( self.imarray.shape )

        # TODO: マスク設定の高速化
        numrows, numcols = self.imarray.shape
        for i in range(0, numrows):
            for j in range(0, numcols):
                imarray_[i, j] = self.intensityRC( i, j, a_ratio )

        if DEBUG and self.shifted:
            # np.savetxt("slow_adjust_100.csv", imarray_[0:100,0:100], delimiter=",")
            # np.savetxt("slow_adjust_100_.csv", imarray_[0:100,-100:], delimiter=",")
            pass

        self.imarray = imarray_
        self.image.set_data( self.imarray, force=True )

    def intensityRC( self, r, c, a_ratio ):
        if not self.shifted:
            intensity = self.imarray[r,c]
            if self.has_mask and self.mask_array[ r, c ]:
                intensity = -2
        else:
            intensity = 0
            cover_ratio = 0

            k = -1
            for p, q in ( [0,0], [0,1], [1,0], [1,1] ):
                k += 1
                i = r+self.ir+p
                if not ( i >= 0 and i < self.imarray.shape[0] ):
                    continue
                j = c+self.ic+q
                if not ( j >= 0 and j < self.imarray.shape[1] ):
                    continue
                value = self.imarray[ i, j ]
                if self.has_mask and self.mask_array[ i, j ]:
                    value = -2

                if value < 0:
                    continue

                ratio = self.cratios[ k ]
                intensity += value * ratio
                cover_ratio += ratio

            if cover_ratio < a_ratio:
                intensity = -3
            elif cover_ratio < 1.0:
                    # print 'cover_ratio(%d,%d)=%g' % ( r, c, cover_ratio )
                    intensity /= cover_ratio

        if intensity > 0:
            intensity /= self.i_ratio

        return int( intensity )

    # --------------------------------------------------------------------------
    #   位置調整画像作成（移動量 pixel 丸め版、最高速）
    # --------------------------------------------------------------------------
    def round_adjust( self ):
        # マスクをかける。
        if self.has_mask:
            self.imarray[ self.mask_array == 1 ] = -2

        if not self.shifted:
            return

        # print( 'ir_=%d, ic_=%d' % ( self.ir_rounded, ic_ ) )
        # 対応不能領域を含めて全体に無効値（-2）を設定しておく。
        adjusted_array = np.ones( self.imarray.shape ) * ( -2 )

        # 対応可能な矩形領域の両端を把握する。
        row, col = self.imarray.shape
        adj_min_r = max(   0,   0 - self.ir_rounded )
        adj_max_r = min( row, row - self.ir_rounded )
        adj_min_c = max(   0,   0 - self.ic_rounded )
        adj_max_c = min( col, col - self.ic_rounded )
        sft_min_r = max(   0,   0 + self.ir_rounded )
        sft_max_r = min( row, row + self.ir_rounded )
        sft_min_c = max(   0,   0 + self.ic_rounded )
        sft_max_c = min( col, col + self.ic_rounded )

        # 行列（numpy.ndarray）のスライスを使ってデータを対応位置にコピーする。
        adjusted_array[ adj_min_r:adj_max_r, adj_min_c:adj_max_c ] = self.imarray[ sft_min_r:sft_max_r, sft_min_c:sft_max_c ]
        self.imarray = adjusted_array.astype( 'i4' )
        self.image.set_data( self.imarray, force=True )

        return

    # --------------------------------------------------------------------------
    #   位置調整画像作成（高速版）
    # --------------------------------------------------------------------------
    def fast_adjust( self, a_ratio ):
        if self.has_mask:
            self.imarray[ self.mask_array == 1 ] = -2

        if not self.shifted:
            return

        # print 'Doing fast_adjust'

        # ４つの成分行列 (0,0), (0,1), (1,0), (1,1) のカバー比率を計算しておく。
        self.compute_cratios()

        # 周囲の境界部分を上下左右に 1 pixel 幅だけ拡張し、無効値を設定した拡張画像データを作る。
        numrows, numcols = self.imarray.shape
        extended_array = np.ones( [ numrows+2, numcols+2 ] ) * INVALID_VALUE
        extended_array[1:numrows+1,1:numcols+1] = self.imarray

        # シフト量を考慮して、成分行列 (0,0) 相当の参照可能範囲を把握する。
        a00_r_min = max(       0 + self.ir,           0 )
        a00_r_max = min( numrows + self.ir + 1, numrows + 1 )
        a00_c_min = max(       0 + self.ic,           0 )
        a00_c_max = min( numcols + self.ic + 1, numcols + 1 )

        # ４つの成分行列 (0,0), (0,1), (1,0), (1,1) から値行列とウェイト行列を計算する。
        k = 0
        for p, q in ( [0,0], [0,1], [1,0], [1,1] ):
            carray_ = extended_array[ ( a00_r_min+p ):( a00_r_max+p ), ( a00_c_min+q ):( a00_c_max+q ) ] * ( self.cratios[ k ] / self.i_ratio )
            # print 'carray[%d].shape=' % ( k ), carray_.shape

            if k == 0:
                v_array  = np.zeros( carray_.shape, dtype='f8' )    # 作業用の調整値行列
                w_array  = np.zeros( carray_.shape, dtype='f8' )    # ウェイト行列

            valid_cond_ = carray_ >= 0
            v_array[ valid_cond_ ] += carray_[ valid_cond_ ]
            w_array[ valid_cond_ ] += self.cratios[ k ]

            k += 1

        # ウェイトが許容可能なカバー比率未満の pixel の値を無効値にする。
        v_array[ w_array < a_ratio ] = INVALID_VALUE

        # ウェイトが許容可能なカバー比率以上の pixel の値を基準化する。
        valid_cond = w_array >= a_ratio
        v_array[ valid_cond ] /= w_array[ valid_cond ]

        # 結果の調整行列における設定位置を把握する。
        # TODO: check for both signs (+-) of shift values
        b00_r_min = max(       0 - self.ir,       0 )
        if b00_r_min > 0:
            b00_r_min   -= 1
            r_offset = 0
        else:
            r_offset = 1
        b00_r_max = b00_r_min + v_array.shape[0] - r_offset
        b00_c_min = max(       0 - self.ic,       0 )
        if b00_c_min > 0:
            b00_c_min   -= 1
            c_offset = 0
        else:
            c_offset = 1
        b00_c_max = b00_c_min + v_array.shape[1] - c_offset

        # print b00_r_min, b00_r_max, b00_c_min, b00_c_max

        # 結果の調整行列を無効値（int型）で初期化する。
        imarray_ = np.ones( self.imarray.shape, dtype='i4' ) * INVALID_VALUE

        # 上で計算した調整値行列を結果の調整行列の該当位置に代入する。
        imarray_[ b00_r_min:b00_r_max, b00_c_min:b00_c_max ] = v_array[r_offset:v_array.shape[0], c_offset:v_array.shape[1]]

        if DEBUG:
            # np.savetxt("fast_adjust.csv", imarray_, delimiter=",")
            # np.savetxt("fast_adjust_100.csv", imarray_[0:100,0:100], delimiter=",")
            # np.savetxt("fast_adjust_100_.csv", imarray_[0:100,-100:], delimiter=",")
            pass

        self.imarray = imarray_     # dtype='i4' のままでよい。
        self.image.set_data( self.imarray, force=True )

    # --------------------------------------------------------------------------
    #   ２画像の欠損補填（廃棄予定）
    # --------------------------------------------------------------------------
    def make_covered_array( self, other ):

        assert self.imarray.shape == other.imarray.shape
        print( 'Making covered_array' )

        covered_array = self.imarray.copy()

        numrows, numcols = self.imarray.shape
        for i in range(0, numrows):
            for j in range(0, numcols):
                intensity0 = self.imarray[ i, j ]
                intensity1 = other.imarray[ i, j ]
                if intensity0 < 0:
                    if intensity1 >= 0:
                        # other 側に有効値があればそれを設定する。
                        covered_array[ i, j ] = intensity1
                    else:
                        # other 側に有効値がなければ、自身のマスク示唆値（負値）を設定する。
                        covered_array[ i, j ] = intensity0

        return covered_array.astype('i4')

    # --------------------------------------------------------------------------
    #   ２画像の欠損補填
    # --------------------------------------------------------------------------
    def fast_make_covered_array( self, other ):

        assert self.imarray.shape == other.imarray.shape
        # print 'Fast making covered_array'

        covered_array = self.imarray.copy()

        valid_cond = np.logical_and( self.imarray < 0, other.imarray >= 0 )

        covered_array[ valid_cond ] = other.imarray[ valid_cond ]

        return covered_array.astype('i4')

    # --------------------------------------------------------------------------
    #   ２画像の平均化（廃棄予定）
    # --------------------------------------------------------------------------
    def make_average_array( self, other ):

        assert self.imarray.shape == other.imarray.shape
        print( 'Making average_array' )

        average_array = self.imarray.copy()

        numrows, numcols = self.imarray.shape
        for i in range(0, numrows):
            for j in range(0, numcols):
                intensity0 = self.imarray[ i, j ]
                intensity1 = other.imarray[ i, j ]
                if intensity0 < 0:
                    if intensity1 >= 0:
                        # other 側に有効値があればそれを設定する。
                        average_array[ i, j ] = intensity1
                    else:
                        # other 側に有効値がなければ、自身のマスク示唆値（負値）を設定する。
                        average_array[ i, j ] = intensity0
                else:
                    if intensity1 >= 0:
                        average_array[ i, j ] = ( intensity0 + intensity1 ) // 2
                    else:
                        average_array[ i, j ] = intensity0

        return average_array.astype('i4')

    # --------------------------------------------------------------------------
    #   多画像の平均化
    # --------------------------------------------------------------------------
    def fast_make_average_array( self, counter_array, sum_array ):

        # 無効値で初期化する。
        im_array  = np.ones( counter_array.shape, dtype='i4' ) * INVALID_VALUE

        # 有効値が少なくとも１つあった pixel について平均を計算する。
        valid_cond = counter_array > 0
        im_array[ valid_cond ] = sum_array[ valid_cond ] / counter_array[ valid_cond ]

        return im_array

    # --------------------------------------------------------------------------
    #   テスト支援メソッド群
    # --------------------------------------------------------------------------
    def diff( self, other ):
        diff_imarray = np.zeros( self.imarray.shape )
        diff_imarray[ self.imarray != other.imarray ] = 10000
        return PilatusImage( diff_imarray, original_image=self )

    def save( self, path ):
        assert( self.image is not None )
        self.image.save( path )

    def equal( self, other ):
        diff_im_array = self.diff( other ).image_array()
        isequal_ = ( diff_im_array == np.zeros( self.imarray.shape ) ).all()
        if not isequal_:
            print( diff_im_array == np.zeros( self.imarray.shape ) )
            try:
                for i in range( diff_im_array.shape[0] ):
                    for j in range( diff_im_array.shape[1] ):
                        if diff_im_array[ i, j ] == 10000:
                            print( 'self[%d,%d]=%d, other[%d,%d]=%d' % ( i, j, self.imarray[i,j], i, j, other.imarray[i,j]) )
                            assert( False )
            except:
                pass
        return isequal_
