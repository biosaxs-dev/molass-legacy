"""
    SplashMessage.py

    Copyright (c) 2017-2025, SAXS Team, KEK-PF
"""

import tkinter as Tk
import tkinter.ttk as ttk
from molass_legacy.KekLib.TkUtils import adjusted_geometry
from molass_legacy.KekLib.MultiMonitor import get_selected_monitor
from molass_legacy.KekLib.SplashScreen import SplashScreen

splash_message1 = ( "MOLASS is loading...\n"
                    "Please be patient."
                    )

splash_geometry = '600x380+200+200'

class AppSplashScreen:
    def __init__( self, root, maxval, value ):
        self.root = root
        sp = SplashScreen(root)
        # sp.config(bg="#3366ff")

        pad = Tk.Frame( sp, height=50 )
        pad.pack()

        m = Tk.Label( sp, text=splash_message1, justify=Tk.CENTER, font=("calibri", 30) )
        m.pack()

        self.bar = bar = ttk.Progressbar(sp, orient='horizontal', length=400, mode='determinate')
        bar.pack( pady=40 )
        bar["value"]    = value
        bar["maximum"]  = maxval

        monitor = get_selected_monitor()
        if monitor.width >= 2560:
            # tempoary fix for high-DPI
            # splash_geometry_ = '900x570+300+300'
            splash_geometry_ = splash_geometry
        else:
            splash_geometry_ = splash_geometry
        geometry_ = adjusted_geometry(splash_geometry_)
        root.geometry( geometry_ )
        root.update()

    def update( self ):
        self.bar["value"] += 1
        self.root.update()
