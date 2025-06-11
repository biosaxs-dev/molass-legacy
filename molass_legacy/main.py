"""
    main.py

    Copyright (c) 2016-2025, SAXS Team, KEK-PF
"""

numba_is_available = False
DEBUG = False

def molass_gui():
    """
    molass_gui()
    This function initializes the Molass GUI application.
    It sets up the environment, imports necessary modules, and starts the main GUI loop.
    It also handles debugging and logging if DEBUG is set to True.
    It is designed to be run as the main entry point of the application.

    Note that the entry point is defined in pyproject.toml as follows::

        [project.scripts]
        molass = "molass_legacy.main:molass_gui"

    This definition allows the application to be run from the command line using the `molass` command.
    
    """
    global print
    if DEBUG:
        import logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger()
        logger.addHandler(logging.FileHandler('molass-debug.log', 'a'))
        print = logger.info

    global numba_is_available

    print( 'import strftime' )
    from time   import strftime
    print( strftime('%H:%M:%S import sys') )
    import sys

    print( strftime('%H:%M:%S import os') )
    import os

    print( strftime('%H:%M:%S tkinter') )
    import tkinter as Tk
    import tkinter.ttk as ttk

    try:
        print( strftime('%H:%M:%S screeninfo') )
        import screeninfo
    except:
        print( strftime('%H:%M:%S failed to import screeninfo') )

    # ---- sys.path begin --------------------------------------------------
    this_dir = os.path.abspath(os.getcwd())
    lib_dir = os.path.dirname(this_dir)

    assert os.path.exists(lib_dir)

    sys.path.insert(0, lib_dir)
    # ---- sys.path end ----------------------------------------------------

    import win32console, win32con
    import psutil

    import molass_legacy.KekLib
    from molass_legacy.KekLib.Console    import Console
    console = Console( windows_app_only=True, console_parent_only=True )

    from molass_legacy.KekLib.TkUtils import get_tk_root
    from molass_legacy._MOLASS.SplashMessage  import AppSplashScreen
    #-------------------------------------------------------------------------------
    #   GUI
    #-------------------------------------------------------------------------------
    root = get_tk_root(withdraw=False)  # using get_tk_root to set DebugPlot environment
    splash = AppSplashScreen( root, 26, 3 )

    # ----- pyinstaller help imports begin -----------------------------------------
    print( strftime('%H:%M:%S tkinter other modules') )
    import tkinter.simpledialog
    import tkinter.scrolledtext
    import tkinter.filedialog
    import tkinter.font
    # import tkinter.tix
    splash.update()

    # print( strftime('%H:%M:%S pyautogui') )
    # import pyautogui
    # splash.update()
    print( strftime('%H:%M:%S packaging.version') )
    import packaging.version
    splash.update()
    print( strftime('%H:%M:%S packaging.specifiers') )
    import packaging.specifiers
    splash.update()
    print( strftime('%H:%M:%S numpy') )
    import numpy
    splash.update()
    #    mkl_intel_thread.dll
    #    mkl_core.dll
    #    libiomp5md.dll
    #    mkl_avx2.dll
    #    mkl_def.dll
    print( strftime('%H:%M:%S pylab') )
    import pylab
    splash.update()
    # ----- pyinstaller help imports end   -----------------------------------------
    print( strftime('%H:%M:%S glob') )
    import glob
    splash.update()
    print( strftime('%H:%M:%S scipy.stats') )
    import scipy.stats
    splash.update()
    print( strftime('%H:%M:%S lmfit') )
    import lmfit
    splash.update()
    print( strftime('%H:%M:%S statsmodels.api') )
    import statsmodels.api
    splash.update()
    print( strftime('%H:%M:%S seaborn') )
    import seaborn
    splash.update()
    print( strftime('%H:%M:%S sklearn') )
    from sklearn import gaussian_process
    splash.update()
    print( strftime('%H:%M:%S openpyxl') )
    import openpyxl
    splash.update()
    print( strftime('%H:%M:%S win32com.client') )
    import win32com.client
    splash.update()
    print( strftime('%H:%M:%S win32process') )
    import win32process
    splash.update()
    print( strftime('%H:%M:%S pythoncom') )
    import pythoncom
    splash.update()
    print( strftime('%H:%M:%S matplotlib') )
    import matplotlib.backends.backend_tkagg
    import matplotlib.animation
    splash.update()
    print( strftime('%H:%M:%S mpl_toolkits.axes_grid1') )
    import mpl_toolkits.axes_grid1
    splash.update()
    print( strftime('%H:%M:%S mpl_toolkits.mplot3d') )
    import mpl_toolkits.mplot3d
    splash.update()
    print( strftime('%H:%M:%S idlelib') )
    if sys.version_info > (3,6):
        import idlelib.tooltip
    else:
        import idlelib.ToolTip
    splash.update()
    print( strftime('%H:%M:%S pyperclip') )
    import pyperclip
    splash.update()
    print( strftime('%H:%M:%S wmi') )
    import wmi
    splash.update()
    try:
        print( strftime('%H:%M:%S numba') )
        import numba
        numba_is_available = True
        splash.update()
    except:
        print( strftime('%H:%M:%S failed to import numba') )

    splash.update()
    root.withdraw()

    print( strftime('%H:%M:%S GuiMain') )
    import molass_legacy.SerialAnalyzer
    import molass_legacy.AutorgKek
    from molass_legacy.KekLib.ArgParser import ArgParser
    from molass_legacy._MOLASS.GuiMain import GuiMain

    ap = ArgParser(sys.argv)
    kwargs = ap.get_dict()
    guimain = GuiMain(root, **kwargs)

    root.mainloop()

if __name__ == '__main__':
    # main must not be called in multiprocessing
    import sys
    import os
    this_dir = os.path.dirname(os.path.abspath(__file__))
    legacy_dir = os.path.join(this_dir, "..")
    molass_dir = os.path.join(legacy_dir, "..", "molass-library")
    sys.path.insert(0, legacy_dir)
    sys.path.insert(0, molass_dir)
    from molass_legacy import get_version
    print(f'MOLASS Legacy {get_version()}')
    from molass import get_version
    print(f'MOLASS Library {get_version()}')
    molass_gui()
