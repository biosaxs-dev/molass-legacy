# -*- coding: utf-8 -*-
"""

    ZoomPan.py

    Borrowed from
    stack overflow
        "Matplotlib plot zooming with scroll wheel"
        http://stackoverflow.com/questions/11551049/matplotlib-plot-zooming-with-scroll-wheel

    Modified so as to handle multiple subplot objects

"""
from __future__ import print_function
import time
from ControlKeyState    import get_shift_key_state, get_ctrl_key_state
from OurColorMaps       import CmapAlbulaLikeDynamic, Diverging

class ZoomPan:
    def __init__( self, obj_id, canvas, ax, im_rec ):
        self.obj_id = obj_id
        self.canvas = canvas
        self.orig_xlim = ax.get_xlim()
        self.orig_ylim = ax.get_ylim()
        self.press = None
        self.cur_xlim = None
        self.cur_ylim = None
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.xpress = None
        self.ypress = None
        self.accumulated_scale = 1.0
        self.no_effect_count = 0
        self.im_rec = im_rec
        self.accumulated_cmap_adjustment = 0.0

    def zoom_factory(self, ax_array, base_scale = 2.):
        ax = ax_array[0]

        def cmap_adjust( event ):
            # print( 'obj_id=%d, time=%g' % (self.obj_id, time.time()) )

            if event.button == 'up':
                if self.accumulated_cmap_adjustment < 0.25:
                    self.accumulated_cmap_adjustment += 0.05
                else:
                    return
            else:
                if self.accumulated_cmap_adjustment > -0.25:
                    self.accumulated_cmap_adjustment -= 0.05
                else:
                    return

            im, cmap = self.im_rec
            new_cmap_ = cmap.adjusted_cmap( self.accumulated_cmap_adjustment )
            im.set_cmap( new_cmap_ )
            ax.figure.canvas.draw()

        def zoom(event):
            if event.inaxes not in ax_array: return
            if get_shift_key_state(): return
            if get_ctrl_key_state():
                cmap_adjust( event )
                return

            cur_xlim = ax.get_xlim()
            cur_ylim = ax.get_ylim()

            xdata = event.xdata # get event x location
            ydata = event.ydata # get event y location

            if event.button == 'up':
                # deal with zoom in
                scale_factor = 1 / base_scale
            elif event.button == 'down':
                # deal with zoom out
                scale_factor = base_scale
            else:
                # deal with something that should never happen
                scale_factor = 1
                print( 'unexpected: event.button=', event.button )

            if scale_factor > 1:
                if self.accumulated_scale > 1.0 - 0.001:
                    self.no_effect_count += 1
                    # print( 'self.no_effect_count=', self.no_effect_count )
                    if self.no_effect_count >= 2:
                        # 元のスケールと位置に戻す。
                        ax.set_xlim( self.orig_xlim )
                        ax.set_ylim( self.orig_ylim )
                        ax.figure.canvas.draw()
                        self.no_effect_count = 0
                    return
            else:
                if self.accumulated_scale < 1.0/256 + 0.001:
                    return

            self.no_effect_count = 0
            self.accumulated_scale *= scale_factor

            new_width = (cur_xlim[1] - cur_xlim[0]) * scale_factor
            new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor

            relx = (cur_xlim[1] - xdata)/(cur_xlim[1] - cur_xlim[0])
            rely = (cur_ylim[1] - ydata)/(cur_ylim[1] - cur_ylim[0])

            ax.set_xlim([xdata - new_width * (1-relx), xdata + new_width * (relx)])
            ax.set_ylim([ydata - new_height * (1-rely), ydata + new_height * (rely)])
            ax.figure.canvas.draw()

        # 次により、イベントに伴う呼び出しは、
        # それぞれの ZoomPan インスタンスに対して行われている様子。
        self.canvas.mpl_connect('scroll_event', zoom)

        return zoom

    def pan_factory(self, ax_array):
        ax = ax_array[0]

        def onPress(event):
            if get_shift_key_state(): return
            if get_ctrl_key_state(): return
            if event.inaxes != ax: return
            self.cur_xlim = ax.get_xlim()
            self.cur_ylim = ax.get_ylim()
            self.press = self.x0, self.y0, event.xdata, event.ydata
            self.x0, self.y0, self.xpress, self.ypress = self.press

        def onRelease(event):
            if get_shift_key_state(): return
            if get_ctrl_key_state(): return
            self.press = None
            ax.figure.canvas.draw()

        def onMotion(event):
            if get_shift_key_state(): return
            if get_ctrl_key_state(): return
            if self.press is None: return
            if event.inaxes not in ax_array: return

            dx = event.xdata - self.xpress
            dy = event.ydata - self.ypress
            self.cur_xlim -= dx
            self.cur_ylim -= dy

            for ax_ in ax_array:
                ax_.set_xlim(self.cur_xlim)
                ax_.set_ylim(self.cur_ylim)
                ax_.figure.canvas.draw()

        # 下記により、イベントに伴う呼び出しは、
        # それぞれの ZoomPan インスタンスに対して行われている様子。
        self.canvas.mpl_connect('button_press_event',onPress)
        self.canvas.mpl_connect('button_release_event',onRelease)
        self.canvas.mpl_connect('motion_notify_event',onMotion)

        #return the function
        return onMotion
