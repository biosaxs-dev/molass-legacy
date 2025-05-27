# coding: utf-8
"""

    ElutionMapperAnimator.py

        recognition of peaks

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF

"""
import copy
import numpy                as np
import matplotlib.pyplot    as plt
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
import matplotlib.animation as anm
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.TkUtils                import is_low_resolution
from molass_legacy.KekLib.BasicUtils             import get_caller_module
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from LightObjects           import AnyObject
from .MapperConstructor import create_mapper
from .ElutionMapperPlotter import ElutionMapperPlotter

class ElutionMapperAnimation( Dialog ):
    def __init__( self, parent, sd, mapper_hook=None ):
        self.parent         = parent
        self.serial_data    = sd

        mapper = create_mapper( parent, sd, mapper_hook=mapper_hook, anim_data=True )
        if mapper is None:
            pass

        self.mapper         = mapper
        self.three_d_guide  = False

        # self.caller_module  = get_caller_module( level=2 )

    def show( self ):
        Dialog.__init__( self, self.parent, "titile" )

    def body( self, body_frame ):   # overrides parent class method
        # tk_set_icon_portable( self, module=self.caller_module )

        cframe = Tk.Frame( body_frame )
        cframe.pack()

        plotter     = ElutionMapperPlotter( self, cframe, self.serial_data, anim_mode=True )
        animator    = ElutionMapperAnimator( plotter, self.mapper )
        animator.show()

    def update_plotter( self, i ):
        print( 'update_plotter', i )
        self.plotter.draw( clear= i != 0, animation_counter=i, mapper=self.mapper )

class ElutionMapperAnimator:
    def __init__( self, plotter, mapper ):
        self.plotter    = plotter
        self.mapper     = mapper
        self.num_frames = len( self.mapper.anim_data_list )
        print( 'num_frames=', self.num_frames )

    def show( self ):
        ani = anm.FuncAnimation( self.plotter.fig, self.update, fargs = (),
                                    interval = 100, frames = self.num_frames + 50 )

        self.plotter.mpl_canvas.draw()

    def update( self, i ):
        if i < self.num_frames:
            j = i
            done  = False
        else:
            j = -1
            done  = True
        mapper = self.mapper.anim_data_list[j]
        self.plotter.draw( clear= i != 0, animation_counter=i, mapper=mapper, animation_done=done )
