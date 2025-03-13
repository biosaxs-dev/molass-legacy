"""
    DataUtils.py

    Copyright (c) 2017-2024, SAXS Team, KEK-PF
"""
import sys
import os
import glob
import re

def get_pytools_folder():
    delimter = '\\'
    file_nodes = os.path.abspath( __file__ ).split( delimter )
    try:
        i = file_nodes.index( 'PyTools' ) + 1
    except:
        file_nodes = os.path.dirname( os.path.abspath( __file__ ) ).split( delimter )
        file_nodes[-4] = 'PyTools'
        i = len(file_nodes) - 3

    root_dir = delimter.join( file_nodes[0:i] )
    return root_dir.replace( delimter, '/' )

def cut_upper_folders(folder):
    return re.sub(r'^\w:.*/Data/', '', folder)

def get_in_folder(in_folder=None):
    if in_folder is None:
        from molass_legacy._MOLASS.SerialSettings import get_setting
        in_folder = get_setting('in_folder')
        if in_folder is None:
            return "unkown folder"      # possibly, it is not set yet as in a separate process
    nodes = in_folder.replace("\\", "/").split("/")
    if len(nodes) > 4:
        in_folder = "/".join(["…"] + nodes[-3:])
    return in_folder

def compact_path_name(path, separator='/'):
    nodes = path.split(separator)
    if len(nodes) > 4:
        return separator.join(nodes[0:2] + ['...'] + nodes[-2:])
    else:
        return path

def do_folder( in_folder, cb ):
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

    # print( in_folder, num_files, num_txt_files, uv_folder )
    applied, sd = cb( in_folder, uv_folder=uv_folder, plot=False )
    return applied

def serial_folder_walk(folder, cb, level=0, depth=3, reverse=False):
    for node in sorted( os.listdir( folder ), reverse=reverse ):
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

        ret = do_folder( path, cb )
        if not ret:
            return False
        if level < depth:
            ret = serial_folder_walk( path, cb, level=level+1, depth=depth, reverse=reverse )
            if not ret:
                return False

        if node.find( 'sample_data_no_UV' ) >= 0:
            print("avoid continuing beyond this folder")
            return False
    return True

def mct_folder_walk(folder, cb, level=0, depth=3):
    if not os.path.exists(folder):
        print("%s doed not exist." % folder)
        return True

    for node in sorted( os.listdir( folder ) ):
        if level == 0 and node.find( '20190522' ) < 0: continue
        if level > 0 and node.find( 'Individual' ) < 0: continue

        path = '/'.join( [ folder, node ] )
        if not os.path.isdir( path ): continue
        ret = do_folder( path, cb )
        if not ret:
            return False

        if level < depth:
            ret = serial_folder_walk( path, cb, level=level+1, depth=depth )
            if not ret:
                return False

    return True

def is_microfluidic(folder):
    if folder.find("microfluidic") >= 0:
        return True

    mtd_files = glob.glob(folder + r"\*.mtd")
    return len(mtd_files) > 0

def get_next_subfolder( folder, search_str ):

    found = [ False ]
    next_folder = []
    def func( in_folder, uv_folder, plot ):
        # print( in_folder )
        if found[0]:
            next_folder.append( in_folder )
            return False, None
        else:
            if in_folder.find( search_str ) >= 0:
                found[0] = True
            return True, None

    serial_folder_walk( folder, func )

    if len(next_folder) > 0:
        return next_folder[0]
    else:
        return None

def get_root_folder():
    from molass_legacy.KekLib.ImportUtils import import_module_from_path
    this_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_dir + '/../../..', 'TestEnv.py')
    module = import_module_from_path('TestEnv', path)
    env_dict = module.env_dict
    return env_dict['root']

def test_serial_folder_walk():
    roor_folder = get_root_folder()
    def func( in_folder, uv_folder, plot ):
        print( in_folder )
        return True, None

    serial_folder_walk( roor_folder, func )

def test_get_next_subfolder():
    roor_folder = get_root_folder()
    print( get_next_subfolder( roor_folder, 'Sugiyama' ) )

def get_local_path(node_name):
    import platform
    node_dict = {
        'PFSAXS10' : 'D:\\',
        'DESKTOP-DTT9QDU' : 'C:\\Users\\takahashi\\',
        'MasatsuyoPC' : 'E:\\',
    }
    hostname = platform.uname()[1]
    return node_dict[hostname] + node_name

if __name__ == '__main__':
    sys.path.append( os.path.join(os.path.dirname(__file__), '..') )
    # test_serial_folder_walk()
    # test_get_next_subfolder()
    print(get_local_path('TODO'))