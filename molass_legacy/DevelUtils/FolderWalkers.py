"""

    DevelUtils.FolderWalkers.py

    Copyright (c) 2017-2023, SAXS Team, KEK-PF
"""
import sys
import os
import glob
import re

def do_a_folder(in_folder, cb):
    dat_files = glob.glob( in_folder + '/*.dat' )
    num_files = len( dat_files )
    if num_files < 90:
        return True

    txt_files = glob.glob( in_folder + '/*.txt' ) + glob.glob( in_folder + '/*.mtd' )
    num_txt_files = len( txt_files )
    if num_txt_files == 0:
        uv_folder = '/'.join( in_folder.split( '/' )[0:-1] ) + '/spectra'
        if os.path.exists( uv_folder ):
            # print( uv_folder )
            pass
        else:
            allow = False
            for name in ['20160628', '20191031', '_no_UV']:
                if in_folder.find(name) >= 0:
                    uv_folder = in_folder
                    allow = True
                    break
            if allow:
                pass
            else:
                print('--------------------- skipping', in_folder)
                return True
    else:
        uv_folder = in_folder

    return cb( in_folder, uv_folder=uv_folder, plot=False )

def serial_folder_walk(folder, cb, level=0, depth=3, reverse=False):
    for node in sorted(os.listdir(folder), reverse=reverse):
        path = '/'.join( [ folder, node ] )
        if not os.path.isdir( path ): continue
        # print(node)

        if node.find( 'autorg' ) >= 0: continue
        if node.find( 'analysis' ) >= 0: continue
        if node.find( 'averaged' ) >= 0: continue
        if node.find( '-stain' ) >= 0: continue
        if node.find( '_original' ) >= 0: continue
        if node.find( 'CirAve' ) >= 0: continue
        if node.find( '00_model' ) >= 0: continue
        if node.find( '_full' ) >= 0: continue
        if node.find( '-skip' ) >= 0: continue

        # print('path=', path)
        if path.find( '20170209_2' ) > 0: continue
        if path.find( '/JT' ) >= 0: continue
        if path.find( '20211105' ) > 0: continue

        ret = do_a_folder(path, cb)
        if not ret:
            return False
        if level < depth:
            ret = serial_folder_walk(path, cb, level=level+1, depth=depth, reverse=reverse)
            if not ret:
                return False
    return True
