"""
    NumpyUtils.py

    Copyright (c) 2016-2024, Masatsuyo Takahashi, KEK-PF
"""

import re
import numpy as np

def get_valid_index( X ):
    return np.logical_and( np.logical_not( np.isnan( X ) ), np.logical_not( np.isinf( X ) ) )

def _save_names( fh, names, delimiter ):
    line = delimiter.join( names ) + '\n'
    fh.write( bytearray( line, 'cp932' ) )  # utf8 or cp932

def np_savetxt( file, np_array, mode="w", column_names=None, fmt='%.18e'):
    if file.find( '.csv' ) > 0:
        delimiter_ = ','
    elif file.find( '.xlsx' ) > 0:
        assert( mode == "w" )
        np_savexlsx( file, np_array, column_names=column_names )
        return

    else:
        delimiter_ = '\t'

    if mode == 'a':
        with open( file, "ab" ) as fh:
            if column_names is not None:
                _save_names( fh, column_names, delimiter_ )
            np.savetxt( fh, np_array, fmt=fmt, delimiter=delimiter_ )
    else:
        with open( file, "wb" ) as fh:
            if column_names is not None:
                _save_names( fh, column_names, delimiter_ )
            np.savetxt( fh, np_array, fmt=fmt, delimiter=delimiter_ )

def np_savetxt_with_comments(file, np_array, comments):
    with open( file, "wb" ) as fh:
        for line in comments:
            fh.write(bytearray( line, 'cp932' ))

    np_savetxt(file, np_array, mode='a')

def np_savexlsx(file, np_array, column_names=None):
    assert False, "Not implemented"

data_line_re = re.compile(r'^\s*\d+')

default_encoding = 'cp932'

def np_loadtxt_robust( fname, usecols=None, delimiter=None, encoding=None, retry=True ):
    global default_encoding

    encoding_ = default_encoding if encoding is None else encoding

    try:
        return np_loadtxt( fname, usecols, delimiter, encoding_, retry )
    except UnicodeDecodeError as exc:
        if encoding is None:
            try:
                default_encoding = 'utf-8'
                return np_loadtxt( fname, usecols, delimiter, default_encoding, retry )
            except Exception as exc:
                raise exc
        else:
            raise exc
    except Exception as exc:
        raise exc

def np_loadtxt( fname, usecols=None, delimiter=None, encoding='utf-8', retry=True ):
    global first_data_line

    if type( fname ) == str:
        fh = open( fname, 'r', encoding=encoding )
        is_csv =  fname.find( '.csv' ) > 0
    else:
        fh = fname
        is_csv = False

    comment_lines = []
    first_data_line = None

    def generator( fh ):
        global first_data_line

        for line in fh:
            if type( line ) == bytes:
                line = line.decode()

            if data_line_re.match( line ):
                if first_data_line is None:
                    first_data_line = line
                yield line
            else:
                comment_lines.append( line )

    if delimiter is None:
        delimiter = ',' if is_csv else None

    try:
        array = np.loadtxt( generator( fh ), delimiter=delimiter, usecols=usecols )
    except UnicodeDecodeError as e:
        # should be handled by the caller
        raise e
    except Exception as e:
        if retry and delimiter is None:
            delim_ = ',' if first_data_line.find( ',' ) > 0 else None
            array, comment_lines = np_loadtxt( fname, delimiter=delim_,
                usecols=usecols, encoding=encoding,
                retry=False     # to avoid infinite recursive calls
                )
        else:
            raise e

    fh.close()

    return array, comment_lines

def get_safe_approximate_max( array ):
    val95, val99 = np.percentile( array[ np.isfinite( array ) ], [ 95, 99 ] )
    return val99 if abs( ( val99 - val95 ) / val99 ) < 0.1 else val95

def simply_safe_sprintf( fmt, *args ):
    try:
        s = fmt % tuple( args )
    except:
        try:
            s = fmt % tuple( [ np.nan if a is None else a for a in args ] )
        except:
            s = ' '.join( [ fmt ] + [ str( a ) for a in args ] )
            if fmt[-1] == '\n':
                s += '\n'
    return s

def warped_weight_vector( size, m ):
    assert size - 1 > m 

    n = size - m

    dm = 1/(m-1)
    w_L = np.arange(m)*0.5*dm

    dn = 1/(n-1)
    w_R = 0.5 + 0.5*( dn + np.arange(0, n)*(1-dn)/(n-1) )

    return np.hstack( [ w_L, w_R ] )

def more_warped_weight_vector( size, m ):
    assert size - 1 > m 

    n = size - m

    hm = m//2
    hn = n//2
    hsize = size - hm - hn

    w_Z = np.zeros( hm )

    dm = 1/(hm-1)
    w_L = np.arange(hm)*0.5*dm

    dn = 1/(hn-1)
    w_R = 0.5 + 0.5*( dn + np.arange(0, hn)*(1-dn)/(hn-1) )

    w_1 = np.ones( hn )

    return np.hstack( [ w_Z, w_L, w_R, w_1 ] )

"""
    https://stackoverflow.com/questions/14313510/how-to-calculate-moving-average-using-numpy
"""
def moving_average(a, n=3, keepsize=False) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret_ = ret[n - 1:] / n
    if keepsize:
        assert n % 2 == 1
        # TODO: for n % 2 == 0
        hn  = n//2
        ret_ = np.hstack( [ [ret_[0]]*hn, ret_, [ret_[-1]]*hn  ] )

    return ret_

def arg_less_percentile(x, p):
    kth = int(len(x)*p/100)
    kpp = np.argpartition(x, kth)
    return int(np.average(kpp[0:kth+1]))

def arg_more_percentile(x, p):
    kth = int(len(x)*p/100)
    kpp = np.argpartition(x, kth)
    return int(np.average(kpp[kth:]))

def get_proportional_points(x1, x2, proportions):
    return np.array([x1*(1-w) + x2*w for w in proportions])
