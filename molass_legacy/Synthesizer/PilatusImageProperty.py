# -*- coding: utf-8 -*-
"""

    ファイル名：   PilatusImageProperty.py

    処理内容：

        TIFF ファイルヘッダ情報の取得と表示

"""
from __future__ import division, print_function, unicode_literals

import sys
import os
import re
from PIL            import Image, TiffImagePlugin
from molass_legacy.KekLib.OurTkinter     import Tk, tk_set_icon_portable
import OurMessageBox    as MessageBox

IMAGEDESCRIPTION = TiffImagePlugin.IMAGEDESCRIPTION     # 270

def get_tiffinfo( image ):
    info = TiffImagePlugin.ImageFileDirectory()

    description = image.tag.get( IMAGEDESCRIPTION )
    # returned as tuple
    # print 'type(prop_text)=', type(prop_text), '; prop_text=', prop_text
    if description:
        # TODO: len( description ) > 1
        info[IMAGEDESCRIPTION] = description[0]

    return info

def get_propertyies( path ):
    im = Image.open( path )

    tiffinfo =  get_tiffinfo( im )
    description = tiffinfo.get( IMAGEDESCRIPTION )
    # returned again as tuple! ?

    if description:
        return description[0]
    else:
        return 'Tiff file has no property text.'

class ImagePropertyWindow( Tk.Toplevel ):

    def __init__( self, parent, title, path ):
        Tk.Toplevel.__init__( self )

        tk_set_icon_portable( self, 'synthesizer' )

        self.title( title )

        text = get_propertyies( path )
        # print 'DEBUG: text="', text, '"'

        t = Tk.Text( self, setgrid=1, height=20 )
        t.pack( side=Tk.TOP, expand=Tk.Y, fill=Tk.BOTH)
        t.insert('0.0', text )

        w = Tk.Button( self, text="OK", width=5, command=self.ok, default=Tk.ACTIVE )
        w.pack(side=Tk.TOP, padx=5, pady=5)

        self.geometry("+%d+%d" % (parent.winfo_rootx()+50,
                                  parent.winfo_rooty()+50))

        self.update()

    def ok( self ):
        self.destroy()
