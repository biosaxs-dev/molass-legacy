# coding: utf-8
"""

    ファイル名：   OurMatplotlib.py

    処理内容：

       matplotlib 関連の追加クラス

    Copyright (c) 2016-2024, Masatsuyo Takahashi, KEK-PF

"""
import os
import sys
import re
import numpy as np
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk as NavigationToolbar
from ControlKeyState    import get_shift_key_state

default_colors = None
hex_colors = None

def mpl_1_5_backward_compatible_init( seaborn=True ):
    # for matplotlib 1.5 backward com;patibility
    # to be removed soon
    mpl.rcParams['figure.figsize'] = [8.0, 6.0]
    mpl.rcParams['figure.dpi'] = 80
    mpl.rcParams['savefig.dpi'] = 100

    mpl.rcParams['font.size'] = 12
    mpl.rcParams['legend.fontsize'] = 'large'
    mpl.rcParams['figure.titlesize'] = 'medium'

    if seaborn:
        import seaborn
        seaborn.set_theme()

    # TODO: unify these initializations with get_default_colors, get_hex_color
    global default_colors, hex_colors
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    hex_colors = [ colors.to_hex(c) for c in default_colors ]

def mpl_font_init():
    """
    learned at
    https://stackoverflow.com/questions/35593345/how-can-i-check-the-default-font-of-matplotlib-in-the-python-shell
    """
    mpl.rcParams['font.family'] = ['serif'] # default is sans-serif
    mpl.rcParams['font.serif'] = [
               'Lucida Sans Unicode',
               'cambria',
               'MS Reference Sans Serif',
               'serif',
               ]

def convert_to_fixed_font_latex( text ):
    """
        This function is a workaround to use a fixed font
        only for lengeds.
    """
    # \m causes re.error: bad escape \m at position 1 in python 3.7
    text = re.sub( r'(\S+)', r'$/mathtt{\g<1>}$', text ).replace('/', '\\')
    return text

def get_default_colors():
    # be sure that this call comes after the call to seaborn.set()
    global default_colors
    if default_colors is None:
        # https://stackoverflow.com/questions/42086276/get-default-line-colour-cycle
        # note that this list of colors differs after the call to seaborn.set()
        default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return default_colors

def get_color(k):
    if default_colors is None:
        get_default_colors()
    return default_colors[k%len(default_colors)]

def get_hex_color(k):
    global hex_colors
    if hex_colors is None:
        hex_colors = [ colors.to_hex(c) for c in default_colors ]
    return hex_colors[k%len(hex_colors)]

def cc(arg):
    return colors.to_rgba(arg, alpha=0.6)

def get_facecolors(num_colors, start=0):
    return [cc('C%d' % ((c+start)%8)) for c in range(num_colors)]

class CoordinateFormatter:
    def __init__( self, numrows, numcols, im_array_list, value_shift=0 ):
        self.numrows = numrows
        self.numcols = numcols
        self.im_array_list = im_array_list
        self.value_shift = value_shift
        self.ic = None
        self.ir = None

    def __call__( self, x, y ):
        self.ic = int(x + 0.5)
        self.ir = int(y + 0.5)
        if self.ic >= 0 and self.ic < self.numcols and self.ir >= 0 and self.ir < self.numrows:
            intensities = []
            for im_array in self.im_array_list:
                intensities.append( '%d' % ( im_array[self.ir, self.ic] -self.value_shift ) )
            return 'x=%d, y=%d, intensity=[ %s ]' % (self.ic, self.ir, ','.join( intensities ) )
        else:
            return 'x=%d, y=%d' % ( self.ic, self.ir )

class DataCursor(object):
    text_template = 'x: %d\ny: %d\ni: %d'
    x, y = None, None
    xoffset, yoffset = -20, 20

    def __init__(self, ax_list, action, value_shift=0 ):

        self.ax_list    = ax_list
        self.action     = action
        if self.action == 1:
            color_ = 'cyan'
        else:
            color_ = 'yellow'

        self.value_shift = value_shift
        self.annotations = []
        self.x  = None
        self.y  = None
        for ax in ax_list:
            annotation = ax.annotate(self.text_template, 
                        xy=(self.x, self.y), xytext=(self.xoffset, self.yoffset), 
                        textcoords='offset points', ha='right', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', fc=color_, alpha=0.5),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                        )
            annotation.set_visible(False)
            self.annotations.append( annotation )

    def __call__( self, event_type, event, im_array_list=[], im_shift_list=[] ):

        if event_type == 1:
            if event.key == 'escape':
                for annotation in self.annotations:
                    annotation.set_visible(False)
                event.canvas.draw()
                return

            if self.x == None:
                return

            if event.key == 'up':
                self.y -= 1
            elif event.key == 'down':
                self.y += 1
            elif event.key == 'right':
                self.x += 1
            elif event.key == 'left':
                self.x -= 1
            else:
                # すなわち、矢印キー以外
                return

        else:
            # event_type == 3

            if not get_shift_key_state():
                return

            # すなわち、shift + button_press のとき、
            # 新しい位置に設定する。
            self.x, self.y = event.xdata, event.ydata

        # 以下は、新しい位置に設定するか、矢印キーで移動した場合の（再）描画

        y_limit, x_limit = im_array_list[0].shape

        # 注記の表示位置を調整するための関数
        def compute_offsets( ax, x, y ):
            xmin, xmax = ax.get_xlim()
            if (x - xmin)*x_limit/(xmax - xmin) < 200:
                xoffset = 50
            else:
                xoffset = -20
            return ( xoffset, 20 )

        i = 0
        for annotation in self.annotations:
            # print( 'get_xy_limits=', ( self.ax_list[i].get_xlim(), self.ax_list[i].get_ylim() ) )
            if self.action == 1:
                shift_ = im_shift_list[i]
                if len( shift_ ) == 2:
                    x_, y_ = int(self.x + shift_[1]), int(self.y + shift_[0])
                else:
                    x_, y_ = int(self.x), int(self.y)
            else:
                x_, y_ = self.x, self.y
            if x_ < x_limit and y_ < y_limit:
                intensity = im_array_list[i][y_, x_] - self.value_shift
                annotation.xy = x_, y_
                annotation.set_text(self.text_template % ( x_, y_, intensity ) )
                annotation.set_visible(True)
                annotation.set_position( compute_offsets( self.ax_list[i], x_, y_ ) )
            else:
                annotation.set_visible(False)
            i += 1
        event.canvas.draw()

class ColorBar:
    def __init__( self, im, ax ):

        ticks = np.logspace( 1.0, 6.0, num=6 )
        string_labels = []
        for t in range(1,7):
            string_labels.append( r"$10^{%d}$" % t )

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.05 )

        self.cb = plt.colorbar( im, cax, ticks=ticks )
        self.cb.set_ticklabels( string_labels )

class MplBackGround:
    def __init__(self):
        plt.style.use('dark_background')

    def __del__(self):
        pass
        # plt.style.use('default')
        # print('MplBackGround.__del__')
        # instead, explicitly use reset_to_default_style below

def reset_to_default_style():
    # print('reset_to_default_style')
    plt.style.use('default')
    mpl_1_5_backward_compatible_init()
