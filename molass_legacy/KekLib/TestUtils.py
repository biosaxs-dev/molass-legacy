# coding: utf-8
"""

    TestUtils.py

    Copyright (c) 2016-2019, Masatsuyo Takahashi, KEK-PF
"""
import os
import shutil
import time
import numpy        as np
from subprocess     import call
import ctypes
from ctypes         import c_int, byref
from pyautogui      import screenshot, moveTo, dragTo, dragRel, click, mouseUp, mouseDown
import cv2
from PIL            import ImageGrab
from PIL            import Image    as ImagePIL

from molass_legacy.KekLib.BasicUtils     import mkdirs_with_retry, clear_dirs_with_retry
from TestTkUtils    import split_geometry, geometry_fix
from MachineTypes   import get_chassistype_name, get_display_resolution, get_monitors
# from PilatusImage   import PilatusImage

class _RECT( ctypes.Structure ):
    _fields_ =  [
                    ( 'left',    ctypes.c_int ),
                    ( 'top',     ctypes.c_int ),
                    ( 'right',   ctypes.c_int ),
                    ( 'bottom',  ctypes.c_int ),
                ]

def get_window_rect( hwid ):
    rect = _RECT( 0, 0, 0, 0 )
    ret = ctypes.windll.user32.GetWindowRect ( c_int( hwid ), byref(rect) )
    # print( ret )
    # print( rect.left, rect.top, rect.right, rect.bottom )
    return ( rect.left, rect.top, rect.right, rect.bottom )

def set_window_pos( hwid, after, x, y, cx, cy, flags ):
    ret = ctypes.windll.user32.SetWindowPos(
        c_int(hwid), c_int( after ), c_int( x ),  c_int( y ),  c_int( cx ),  c_int( cy ), c_int( flags ) )
    # print( 'ret=', ret )
    if ret == 0:
        print( ctypes.FormatError() )
    return ret

def single_screenshot( hwid, file ):
    l, t, r, b = get_window_rect( hwid )
    im = ImageGrab.grab( bbox=( l, t, r, b ) )
    im.save( file )

def open_explorer( folder, at=None ):
    call( 'explorer %s' % ( folder ), shell=True )

    if at == None: return None

    assert( len( at ) == 2 )

    name = os.path.split( folder )[-1]

    retry   = 0
    hwid    = 0
    while( retry < 3 and hwid == 0 ):
        time.sleep( 1 )
        hwid = ctypes.windll.user32.FindWindowW( None, name )
        retry += 1

    if hwid == 0:
        print( "Can't find window titled '%s'" % ( name ) )
        return None

    print( "Found window titled '%s'" % ( name ) )

    x, y, r, b = get_window_rect( hwid )
    print(  x, y, r, b )

    if get_chassistype_name() == 'Desktop':
        set_window_pos( hwid, 0, at[0], at[1], 800, 400, 0 )
    else:
        # notebook では set_window_pos すると、後続の drag のタイミングが
        # よくないようなので、drag で位置調整する。
        x_      = ( x + r ) / 2
        y_      = y + 20

        rel_x   = at[0] - x
        rel_y   = at[1] - y

        moveTo( x_, y_ )
        dragRel( rel_x, rel_y, 0.5 )

    return hwid

def close_explorer( hwid ):
    _, t, r, _ = get_window_rect( hwid )
    x, y = r - 30, t + 10
    moveTo( x, y )
    click()

def dnd_from_to( from_folder, dnd_obj_image, root, to_widget, screenshot_only=False ):
    display_width, _    = get_display_resolution()
    monitors            = get_monitors()
    print( 'monitors=', monitors )
    if len( monitors ) == 1:
        display_left_position = 0
    else:
        display_left_position = display_width + 500
    print( 'display_left_position=', display_left_position )

    geometry_fix( root, display_left_position, 0 )

    print( 'from_folder=', from_folder )
    hwid = open_explorer( from_folder, at=( display_left_position + 200, 230 ) )
    time.sleep( 1 )
    temp_dir = 'dnd_from_to_temp'
    clear_dirs_with_retry( [ temp_dir ] )
    screenshot_im_file = temp_dir + '/folder_screenshot.png'

    # screenshot( screenshot_im_file )
    single_screenshot( hwid, screenshot_im_file )

    fx, fy, fr, fb = get_window_rect( hwid )
    print( 'fx, fy, fr, fb=', fx, fy, fr, fb )

    if screenshot_only:
        return

    from_screen = cv2.imread( screenshot_im_file )
    obj_image   = cv2.imread( dnd_obj_image )
    ( objHeight, objWidth ) = obj_image.shape[:2]

    result = cv2.matchTemplate( from_screen, obj_image, cv2.TM_CCOEFF )
    (_, _, minLoc, maxLoc) = cv2.minMaxLoc(result)
    print( 'result=', minLoc, maxLoc )
    topLeft = maxLoc
    objX = fx + topLeft[0] + objWidth/2
    objY = fy + topLeft[1] + objHeight/2

    W, H, X, Y = split_geometry( root.geometry() )
    print( "root geometry", W, H, X, Y )
    w, h, x, y = split_geometry( to_widget.winfo_geometry() )
    print( "to_widget geometry", w, h, x, y )

    to_widget_ok = False

    def dummy():
        pass

    while( not to_widget_ok ):
        moveTo( objX, objY )
        time.sleep( 0.5 )

        to_widget.focus_force()

        x_ = X + x + w/2
        y_ = Y + y + h/2 + 30
        dragTo( x_, y_, 0.5 )
        mouseDown( x_, y_ )
        mouseUp( x_, y_ )
        time.sleep( 0.5 )
        for k in range(3):
            root.update()
        value = to_widget.get()
        print( value )
        if value.find( '<' ) < 0:
            to_widget_ok = True

    close_explorer( hwid )
    shutil.rmtree( temp_dir )

def OldPilatusImage( path, original_image=None ):
    assert( original_image is not None )
    from PilatusImage import PilatusImage
    im = ImagePIL.open( path )
    im_array = np.array( im )
    return PilatusImage( im_array, original_image=original_image )

def max_diff(a, b):
    """
    Elegant way to perform tuple arithmetic
    https://stackoverflow.com/questions/17418108/elegant-way-to-perform-tuple-arithmetic
    """
    try:
        return np.max(np.subtract(a, b))
    except Exception as exc:
        raise ValueError( str(exc) + '\nvalues=' + '\na=' + str(a) + '\nb=' + str(b) )

def max_diff_assert_lt(a, b, c):
    try:
        assert max_diff(a, b) < c
    except Exception as exc:
        raise AssertionError( str(exc) + '\nvalues=' + '\na=' + str(a) + '\nb=' + str(b) )

def get_filesize(path):
    """
    How to check file size in python?
    https://stackoverflow.com/questions/2104080/how-to-check-file-size-in-python
    """
    statinfo = os.stat(path)
    return statinfo.st_size
