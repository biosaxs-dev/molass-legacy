# coding: utf-8
"""

    OurScreenShot.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""
import sys
import os
from time               import strftime
import pyautogui
from molass_legacy.KekLib.BasicUtils         import mkdirs_with_retry, auto_numbered_file, get_home_folder, get_caller_module
from molass_legacy.KekLib.TkUtils            import split_geometry
from DevSettings        import get_dev_setting

def screenshot( file=None, widget=None, log=False ):
    if file is None:
        folder = get_dev_setting( 'screenshot_folder' )
        if folder is None:
            folder = get_home_folder() + '/img'
        file = folder + '/screen-000.png'

    folder, _ = os.path.split( file  )
    if not os.path.exists( folder ):
        mkdirs_with_retry( folder )

    region = None
    if widget is not None:
        w, h, x, y = split_geometry( widget.geometry() )
        region = ( x, y, w, h )

    file = auto_numbered_file( file )
    # pyautogui.screenshot( file, region=region )

    pyautogui.screenshot( file )

    if log:
        _, f = os.path.split( file  )

        log = open( folder + '/screenshot.log', 'a' )
        _, modname = os.path.split( get_caller_module( level=2 ).__name__ )
        log.write( strftime( '%H:%M:%S ' ) + modname + ' region=' + str(region) + ' file=' + f + '\n' )
        log.close()
