# coding: utf-8
"""

    BasicUtils.py

    Copyright (c) 2016-2022, Masatsuyo Takahashi, KEK-PF

"""

from __future__ import division, print_function, unicode_literals

import sys
import os
import re
import getpass
import time
import shutil
import inspect

def mkdirs_with_retry( path, retry=3 ):
    retry_ = 0
    while( not os.path.exists( path ) ):
        try:
            os.makedirs( path )
        except:
            if retry_ < retry:
                retry_ += 1
                time.sleep( 1 )
                print( 'retry os.makedirs( %s )' % ( path ) )
            else:
                assert( False )

def clear_dirs_with_retry( dirs, retry=3  ):
    for dir_ in dirs:
        if os.path.exists( dir_ ):
            shutil.rmtree( dir_ )
        mkdirs_with_retry( dir_, retry=retry )

def rename_with_retry( old_path, new_path, retry=3 ):
    retry_ = 0
    done_ = False
    while( not done_ ):
        try:
            os.rename( old_path, new_path )
            done_ = True
        except:
            if retry_ < retry:
                retry_ += 1
                time.sleep( 1 )
                print( 'retry os.rename( %s, %s )' % ( old_path, new_path ) )
            else:
                assert( False )

def is_empty_dir( path ):
    if not os.path.exists( path ):
        return False

    for dirpath, dirnames, files in os.walk( path ):
        # print( 'files=', files )
        if files:
            return False
        else:
            return True

def num_files_in(path):
    for dirpath, dirnames, files in os.walk( path ):
        if files:
            return len(files)
        else:
            return 0

def is_almost_empty_dir( path ):
    if not os.path.exists( path ):
        return False

    for dirpath, dirnames, files in os.walk( path ):
        if len( files ) == 0:
            return True
        elif len( files ) == 1:
            path = os.path.join( dirpath, files[0] )
            return os.path.getsize( path ) == 0
        else:
            return False

def is_usedfolder(folder):
    return os.path.exists(folder) and len(os.listdir(folder)) > 0

def get_filename_extension( filename ):
    f = filename.split( '.' )
    if len( f ) > 1:
        ext = f[-1]
    else:
        ext = None
    return ext

def exe_name():
    name_ =  os.path.split( sys.argv[0] )[-1]
    for ext in ( '.exe', '.py' ):
        name_ = name_.replace( ext, '' )
    return name_

def home_dir():
    try:
        username = getpass.getuser()
    except:
        # suddenly began to happen on 20191106 on a note PC
        import win32api
        username = win32api.GetUserName()
    return os.path.expanduser( '~' + username )

"""
    URL: http://stackoverflow.com/questions/1305532/convert-python-dict-to-object

    namedtuple の場合、メンバーの更新は can't set attritube により不可のようなので
    この Struct の方が柔軟。
"""
class Struct:
    def __init__(self, **entries): 
        self.__dict__.update(entries)
    def __str__(self):
        return str(self.__dict__)

def make_indecies_text( indecies, offset=0 ):
    assert( len( indecies ) > 0 )
    previous = None
    is_continuous = True
    for i in indecies:
        if previous is not None:
            if i != previous + 1:
                is_continuous = False
        previous = i

    if is_continuous:
        if len( indecies ) == 1:
            text = str( offset+indecies[0] )
        else:
            text = '%d-%d' % ( offset+indecies[0], offset+indecies[-1] )
    else:
        text = ','.join( [ str(offset+i) for i in indecies ] )

    return text

"""
thanks to: http://stackoverflow.com/questions/9647202/ordinal-numbers-replacement
"""
def ordinal_str(numb):
    if numb < 20: #determining suffix for < 20
        if numb == 1: 
            suffix = 'st'
        elif numb == 2:
            suffix = 'nd'
        elif numb == 3:
            suffix = 'rd'
        else:
            suffix = 'th'  
    else:   #determining suffix for > 20
        tens = str(numb)
        tens = tens[-2]
        unit = str(numb)
        unit = unit[-1]
        if tens == "1":
           suffix = "th"
        else:
            if unit == "1": 
                suffix = 'st'
            elif unit == "2":
                suffix = 'nd'
            elif unit == "3":
                suffix = 'rd'
            else:
                suffix = 'th'
    return str(numb)+ suffix


def get_caller_module( level=1 ):
    # get the caller module
    frm = inspect.stack()[level]
    mod = inspect.getmodule(frm[0])
    return mod

def get_home_folder():
    folder = __file__
    for i in range( 3 ):
        folder = os.path.dirname( folder )
    return folder.replace( '\\', '/' )

def auto_numbered_file( f ):
    while os.path.exists( f ):
        f = re.sub( r'-(\d+)(\.\w+)$', lambda m: '-' + '%03d' % ( int(m.group(1)) + 1 ) + m.group(2), f )
    return f

"""
    from http://stackoverflow.com/questions/651794/whats-the-best-way-to-initialize-a-dict-of-dicts-in-python
"""

class AutoVivifiedDict(dict):
    """Implementation of perl's autovivification feature."""
    def __getitem__(self, item):
        try:
            return dict.__getitem__(self, item)
        except KeyError:
            value = self[item] = type(self)()
            return value

def open_w_safely( path, mode, rename_cb=None ):
    assert mode == "w", "Supprted only for write mode"

    if os.path.exists( path ):
        if rename_cb is None:
            path_bak = re.sub( r"(\.\w+)$", ".bak", path )
            if os.path.exists( path_bak ):
                os.remove( path_bak )
            os.rename( path, path_bak )
        else:
            rename_cb( path )

    fh  = open( path, mode )
    return fh

def rename_existing_file(path, ext=''):
    search_pattern = "(-\\d*)?\\" + ext + "$"
    replace_pattern = "-%02d" + ext

    n = 0
    path_bak = path
    while os.path.exists( path_bak ):
        path_bak = re.sub( search_pattern, replace_pattern % n, path )
        n += 1

    if path_bak != path:
        os.rename( path, path_bak ) 

def print_exception():
    from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
    etb = ExceptionTracebacker()
    print(etb)
