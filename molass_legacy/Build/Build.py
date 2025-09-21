# -*- coding: utf-8 -*-
"""

    ファイル名：   Build.py

    処理内容：

        配布形式のビルド

    Copyright (c) 2016-2018, Masatsuyo Takahashi, KEK-PF

"""

from __future__ import division, print_function, unicode_literals
import sys
import os
import re
import glob
import platform
import shutil
from subprocess         import call
from RequiredPackages   import check_tested
from BasicUtils         import clear_dirs_with_retry, rename_with_retry, mkdirs_with_retry

class_name_dict = {
    'synthesizer'       :   'Synthesizer',
    'autorg_kek'        :   'AutorgKek',
    'serial_analyzer'   :   'SerialAnalyzer',
    'pyinstaller-test'  :   'pyinstaller-test',
    'window_size_tester'    :   'SerialAnalyzer',
    'our_python'        :   'Python',
    'our_nosetests'     :   'Python',
    }

def build( thisdir, updir, script ):
    check_tested()

    main_script = script
    main_name   = main_script.replace( '.py', '' )

    bitness = platform.architecture()[0]
    if bitness == '64bit':
        bitness_name    = 'x64'
        # assert( updir.find( 'x64' ) >= 0 )
    elif bitness == '32bit':
        bitness_name    = 'x86'
        # assert( updir.find( 'x86' ) >= 0 )
    else:
        assert( False )

    print( 'Targeted architecture bitness seems to be consistent.' )

    ret = call( 'upx -V > nul', shell=True )
    if ret == 0:
        print( 'UPX is available.' )
        upx_is_avalable = True
    else:
        print( 'UPX is not available.' )
        upx_is_avalable = False

    built_dir = './dist/%s' % ( main_name )
    temp_dirs = [ 'build', 'dist', 'temp' ]
    for dir_ in temp_dirs:
        if os.path.exists( dir_ ):
            shutil.rmtree( dir_ )

    if main_name == 'executables':
        ret = call( 'pyinstaller %s --onedir' % ( main_script ), shell=True )
    else:
        class_name = class_name_dict[main_name]
        if main_name in [ 'autorg_kek', 'window_size_tester', 'python', 'pyinstaller-test' ]:
            windowed = ''
        else:
            windowed = '--windowed'
        command = 'pyinstaller %s --onedir %s --icon=../lib/%s/%s.ico' % ( main_script, windowed, class_name, main_name )
        print( command )
        ret = call( command, shell=True )

    assert( ret == 0 )
    assert( os.path.exists( built_dir ) )
    """
        mannually execute pyinstaller
        and see if "failed to "failed to create process" is shown.
    """

    python_install_dir = os.path.dirname( sys.executable )

    print( "Python is installed in '%s'" % ( python_install_dir ) )

    import numpy.core
    numpy_core_dir = os.path.dirname( numpy.core.__file__ )
    print( "numpy.core is in '%s'" % ( numpy_core_dir ) )

    if upx_is_avalable:
        working_dll_dir = 'temp'
        clear_dirs_with_retry( [ working_dll_dir ] )
    else:
        working_dll_dir = numpy_core_dir

    existing_dll_names = []
    dll_names = [ 'libiomp5md.dll', 'mkl_avx2.dll', 'mkl_core.dll', 'mkl_def.dll', 'mkl_intel_thread.dll' ]
    for name in dll_names:
        dll_path = os.path.join( numpy_core_dir, name )
        if os.path.exists( dll_path ):
            existing_dll_names.append( name )
            if upx_is_avalable:
                print( 'Compressing %s with UPX...' % ( name ) )
                working_dll_path = os.path.join( working_dll_dir, name )
                shutil.copy2( dll_path, working_dll_path )
                ret = call( 'upx %s' % ( working_dll_path ) )
                assert( ret == 0 )

    for name in existing_dll_names:
        shutil.copy2( os.path.join( working_dll_dir, name ), built_dir )

    if upx_is_avalable:
        vs140_dll_dir   = 'C:/Program Files (x86)/Microsoft Visual Studio 14.0/Common7/IDE/Remote Debugger/' + bitness_name
        pyrhonwin_dir   = python_install_dir + '/Lib/site-packages/pythonwin'
        badly_compressed_dlls = [
            os.path.join( numpy_core_dir,       'concrt140.dll'     ),
            os.path.join( numpy_core_dir,       'msvcp140.dll'      ),
            os.path.join( numpy_core_dir,       'vcamp140.dll'      ),
            os.path.join( numpy_core_dir,       'vccorlib140.dll'   ),
            os.path.join( numpy_core_dir,       'vcomp140.dll'      ),
            os.path.join( python_install_dir,   'vcruntime140.dll'  ),
            os.path.join( vs140_dll_dir,        'ucrtbase.dll'      ),
            os.path.join( pyrhonwin_dir,        'mfc140u.dll'       ),
            ]

        for path in badly_compressed_dlls:
            name = os.path.split( path )[-1]
            target_path = os.path.join( built_dir, name )
            if not os.path.exists( target_path ):
                continue
            upxname = name.replace( '.dll', '-upx.dll' )
            rename_path = os.path.join( built_dir, upxname )
            os.rename( target_path, rename_path )
            os.remove( rename_path  )
            shutil.copy2( path, target_path )
            print( 'copied from original to' + target_path )

    dist_dir = os.path.join( updir, main_name )
    if os.path.exists( dist_dir ):
        shutil.rmtree( dist_dir )

    rename_with_retry( built_dir, dist_dir )

    for dir_ in temp_dirs:
        if os.path.exists( dir_ ):
            shutil.rmtree( dir_ )

    os.remove( main_script.replace( '.py', '.spec' ) )

    if upx_is_avalable:
        with_phrase = ' with UPX compress'
    else:
        with_phrase = ''
    print( 'The distribution directory has been successfully built%s.' % ( with_phrase ) )

arc_dict = { '32bit':'x86', '64bit':'x64' }

def split_application( app=None, out_folder=None ):
    if app is not None and app != 'autorg_kek':
        return

    import AutorgKek
    from AppVersion import get_com_version_string
    vstr = get_com_version_string()
    def sub_func(m):
        global arch
        arch = arc_dict[m.group(2)]
        return m.group(1) + '-' + arch

    ver = re.sub( r'(autorg_kek\s+\d+\.\d+\.\d+).+\s+(\d+bit).+', sub_func, vstr ).replace('.', '_').replace(' ', '-')
    print( __file__ )
    assert( __file__.find( arch ) > 0 )
    print( ver, arch )
    path = __file__.replace( '\\', '/' )
    org_folder  = '/'.join( path.split( '/' )[0:-3] )
    up_folder   = '/'.join( path.split( '/' )[0:-4] )

    if out_folder is None:
        out_folder = up_folder + '/' + ver

    print( 'org_folder=', org_folder )
    print( 'out_folder=', out_folder )

    clear_dirs_with_retry( [ out_folder ] )

    for file in [ 'autorg_kek.bat', 'autorg_kek_client.bat' ]:
        print( 'Copying ' + file )
        src_path = org_folder + '/' + file
        tgt_path = out_folder + '/' + file
        shutil.copy2( src_path, tgt_path )

    # for subfolder in [ 'build', 'glue_devel', 'lib/KekLib', 'lib/AutorgKek', 'test_autorg_kek', 'executables' ]:
    for subfolder in [ 'build', 'glue_devel', 'lib/KekLib', 'lib/AutorgKek', 'test_autorg_kek' ]:
        print( 'Copying ' + subfolder )
        src_path = org_folder + '/' + subfolder
        tgt_path = out_folder + '/' + subfolder
        shutil.copytree( src_path, tgt_path )

    # remove/build/*
    for node in [   'build/__pycache__',
                    'build/split.py',
                    'build/serial_analyzer.py',
                    'build/executables.py',
                    'build/synthesizer.py',
                    'executables/synthesizer*',
                    'executables/serial_analyzer*',
                ]:
        tgt_path = out_folder + '/' + node

        if tgt_path.find('*') > 0:
            # print( 'glob ' + tgt_path )
            files = glob.glob(tgt_path)
            for f in files:
                print( 'Removing ' + f )
                os.remove( f )
            continue

        if not os.path.exists( tgt_path ):
            continue

        print( 'Removing ' + tgt_path )
        if os.path.isdir(tgt_path):
            shutil.rmtree( tgt_path )
        else:
            os.remove( tgt_path )

    # 
    tgt_dir = out_folder + '/lib/CommandMain'
    mkdirs_with_retry( tgt_dir )
    ini_path = tgt_dir + '/__init__.py'
    fh = open( ini_path, 'wb' )
    code = '\n'.join( [
        '"""',
        '    trick to make it backward compatible',
        '"""',
        'import KekLib',
        'import AutorgKek',
        'from AutorgKek.CommandMain  import CommandMain, autorg_func',
        '',
        ] )
    fh.write( code.encode() )
    fh.close()
