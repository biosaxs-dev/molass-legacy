# coding: utf-8
"""

    ファイル名：   RequiredPackages.py

    処理内容：

        必須パッケージのインストール

    Copyright (c) 2016-2023, Masatsuyo Takahashi, KEK-PF
"""

from __future__ import division, print_function, unicode_literals
import sys
import os
import platform
import shutil
import re
import packaging    # to check, for it is required to build distributions
from subprocess import call
from BasicUtils import clear_dirs_with_retry, rename_with_retry, home_dir

required_packages = [
    'numpy',
    'matplotlib',
    'scipy',
    'packaging',
    'nose',
    'coverage',
    'statsmodels',
    # 'lmfit',
    'sklearn',
    'pymc',
    'seaborn',
    'screeninfo',
    'pyperclip',
    'openpyxl',
    'win32com',
    'pythoncom',
    'packaging',
    ]

site_url = 'http://www.lfd.uci.edu/~gohlke/pythonlibs/'

python_version_short = '%d%d' % sys.version_info[0:2]

bitness_str_map = {
    '32bit' : 'win32',
    '64bit' : 'win_amd64',
    }

bitness_str = bitness_str_map.get( platform.architecture()[0] )
if not bitness_str: assert( False )

postfix = '-cp%s-cp%sm-%s.whl' % ( python_version_short, python_version_short, bitness_str )

print( 'postfix=', postfix )

wheel_file_map = {
    # Caveate! : web page may include non-ascii characters.
    # e.g. '‑'(U+2011)s in 'numpy‑1.9.3+mkl‑cp35‑none‑win_amd64.whl'
    # Do not simply copy and paste!
    'numpy'         : 'numpy-1.11.2+mkl{0}'  .format( postfix ),
    'matplotlib'    : 'matplotlib-1.5.3{0}' .format( postfix ),
    'pandas'        : 'pandas-0.17.1{0}'    .format( postfix ),
    'scipy'         : 'scipy-0.16.1{0}'     .format( postfix ),
    }

to_be_installed_packages = []

def current_platform():
    return ( platform.python_version(), platform.system(), platform.architecture()[0] )

def check_tested():
    current_platform_ = current_platform()

    print( 'Current platform is python %s on %s %s.' % ( current_platform_ ) )

    tested_platforms = [
        ( '3.7.4rc1', 'Windows', '64bit' ),
        ( '3.7.3', 'Windows', '64bit' ),
        ( '3.7.2', 'Windows', '64bit' ),
        ( '3.7.1', 'Windows', '64bit' ),
        ( '3.7.0', 'Windows', '64bit' ),
        ( '3.6.8', 'Windows', '64bit' ),
        ( '3.6.7', 'Windows', '64bit' ),
        ( '3.6.6', 'Windows', '64bit' ),
        ( '3.6.5', 'Windows', '64bit' ),
        ( '3.6.3', 'Windows', '64bit' ),
        ( '3.6.2', 'Windows', '64bit' ),
        ( '3.6.1', 'Windows', '64bit' ),
        ( '3.5.2', 'Windows', '64bit' ),
        ( '3.5.1', 'Windows', '64bit' ),
        ( '3.5.0', 'Windows', '64bit' ),
        ( '3.5.2', 'Windows', '32bit' ),
        ( '3.5.0', 'Windows', '32bit' ),
        ]

    assert( current_platform_ in tested_platforms )

def requirements_ok():

    for package in required_packages:
        print( 'importing', package )
        try:
            exec( 'import ' + package )
        except:
            print( '%s is not yet installed.' % ( package ) )
            # print( sys.exc_info[0] )
            to_be_installed_packages.append( package )

    if len( to_be_installed_packages ) == 0:
        print( 'required modules seem to be installed.' )
        return True

    # install_packages()

    return False

def modification_ok():
    python_install_dir = os.path.dirname( sys.executable )
    # print( python_install_dir )
    pyinstaller_scrips = [
        'pyi-archive_viewer-script.py',
        'pyi-bindepend-script.py',
        'pyi-grab_version-script.py',
        'pyi-makespec-script.py',
        'pyi-set_version-script.py',
        'pyinstaller-script.py'
        ]
    first_line_modified_re = re.compile( '^".+"$' )
    for script in pyinstaller_scrips:
        path = os.path.join( python_install_dir, 'Scripts', script )
        # print( path )
        f = open( path )
        first_line = f.readline()
        f.close()
        # print( first_line )
        if first_line_modified_re.match( first_line ):
            print( '"%s" is not modified.' % path )
            return False

    print( 'pyinstaller scripts seem to be correctly modified.' )

    path = os.path.join( python_install_dir, 'Lib', 'site-packages', 'statsmodels', '__init__.py' )
    import_line_modified_re = re.compile( '^from\s+(\w*)\.tools\.sm_exceptions' )

    # statsmodels modification is still required as of pyinstaller 3.3.
    statsmodels_is_modified = False
    f = open( path )
    for i, line in enumerate( f.readlines() ):
        if i > 12:
            break
        # print( line )
        m = import_line_modified_re.match( line )
        if m:
            name = m.group(1)
            if name == 'statsmodels':
                statsmodels_is_modified = True
            else:
                print( 'WARNING: statsmodels is not modified.' )
    f.close()
    if statsmodels_is_modified:
        print( 'statsmodels seems to be correctly modified.' )
    # return statsmodels_is_modified
    return True

def install():
    check_tested()

    assert( platform.system() == 'Windows' )
    try:
        import win32api
    except:
        print( "Can't check admin privileges." )
        print( "If you are sure, comment out the following 'exit()' and retry." )
        # exit()

    ret = call( "pip -V", shell=True )
    if ret == 0:
        print( 'pip is available.' )
    else:
        print( 'Install pip manually.' )
        exit()

    """
    # TODO: download automation
    'wget' is not enough because we can't get target file url.

    print( 'Installing wget' )
    ret = call( "pip install wget" )
    if ret == 0:
        pass
    else:
        print( "Can't install wget. Please make sure to get admin privileges." )
        exit()
    """

    download_dir = os.path.join( home_dir(), 'Downloads' )
    # print( 'download_dir=', download_dir )

    num_skipped_packages = 0
    num_installed_packages = 0
    for package in required_packages:
        print( 'Installing %s' % ( package ) )
        wheel_file = wheel_file_map.get( package )
        # print( 'wheel_file=%s' % ( wheel_file) )

        if wheel_file == None:
            print( 'We will try to simply pip install {0}'.format( package ))
            ret = call( 'pip install {0}'.format( package ), shell=True )
            assert( ret == 0 )
        else:
            wheel_path = os.path.join( download_dir, wheel_file )
            # print( 'wheel_path=', wheel_path )

            if os.path.exists( wheel_path ):
                print( '%s has been downloaded, ok.' % ( wheel_file ) )
            else:
                print( '%s has not yet been downloaded.' % ( wheel_file ) )
                print( 'Skip installing {0}.'.format( package ) )
                num_skipped_packages += 1
                continue

            ret = call( 'pip install {0}'.format( wheel_path ), shell=True )
            assert( ret == 0 )
        num_installed_packages += 1

    assert( num_skipped_packages == 0 )
    assert( num_installed_packages == len( required_packages ) )

    print( 'All required packages have been successfully installed.' )
