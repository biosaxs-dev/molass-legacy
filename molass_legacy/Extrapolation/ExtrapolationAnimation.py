# coding: utf-8
"""
    ExtrapolationAnimation.py

    Copyright (c) 2018-2019, SAXS Team, KEK-PF
"""
import os
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import set_icon
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurMatplotlib import get_color
from DevSettings import get_dev_setting

TOO_SMALL_TO_PLOT   = 1e-12

def format_func(value, tick_number):
    return '$10^{%d}$' % (value)

class ExtrapolationAnimationDialog(Dialog):
    def __init__( self, parent, row, q, ridge, anim_data, penalty_weights, ignore_bq=0 ):
        self.grab = 'local'     # used in grab_set
        self.q  = q
        self.ridge = ridge
        self.parent = parent
        self.row = row
        self.anim_data1 = anim_data[0]
        self.anim_data2 = anim_data[1]
        self.min_data = np.log10( max( TOO_SMALL_TO_PLOT, np.min(self.anim_data2) ) )
        self.max_data = np.log10( np.max(self.anim_data2) )
        print('min_data=', self.min_data, 'max_data=', self.max_data)
        self.row_penalty_weights = [1.0] + penalty_weights
        self.ignore_bq = ignore_bq
        self.add_conc_const = get_dev_setting('add_conc_const')
        self.aq_smoothness = parent.aq_smoothness
        self.aq_positivity = parent.aq_positivity
        self.pause = False

    def show(self):
        title = "Extrapolation Animation"
        Dialog.__init__( self, self.parent, title )

    def body(self, body_frame):
        set_icon( self )

        cframe = Tk.Frame(self)
        cframe.pack()

        fig = plt.figure( figsize=(21,7) )
        gs = GridSpec(1, 3)
        ax1 = fig.add_subplot( gs[0,0] )
        self.ax2 = ax2 = fig.add_subplot( gs[0,1:] )
        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.mpl_canvas.mpl_connect('button_press_event', self.on_click)

        ax1.set_title("Changes of objective term values", fontsize=20)
        ax1.set_xlabel("Function call count")
        ax1.set_ylabel("Objective Values")
        if self.ignore_bq:
            curve_text = "curve: A(q)"
        else:
            curve_text = "curves: A(q), B(q)"
        ax2.set_title("Optimized scattering essense " + curve_text, fontsize=20)
        ax2.set_xlabel("Q")
        ax2.set_ylabel("$log_{10}(Intensity)$")
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_func))

        # draw ax2 first to get ymin, ymax
        if not self.ignore_bq:
            ax2t = ax2.twinx()
            ax2t.set_ylabel("Intensity")
            ax2t.grid(False)
        r_color = get_color(0)
        aq_color = get_color(1)
        bq_color = 'pink'
        # note that anim_rec[0] > 0
        self.n  = np.array([ anim_rec1[0] for anim_rec1 in self.anim_data1 ])
        anim_rec1 = self.anim_data1[0]
        P  = anim_rec1[1]
        ax2.plot( self.q, np.log10(self.ridge), color=get_color(0))
        aq_y = np.log10(P[:,0])
        aq_curve, = ax2.plot( self.q, aq_y, color=aq_color, label='A(q)', alpha=1 )

        if not self.ignore_bq:
            bq_y = P[:,1]
            bq_curve, = ax2t.plot( self.q, bq_y, color=bq_color, label='B(q)' )

        # ax1.set_yscale("log", nonposy='clip')
        numerator = 0
        denominator = self.anim_data1[-1][0]
        xmin2, xmax2 = ax2.get_xlim()
        ymin2, ymax2 = ax2.get_ylim()
        tx = xmin2*0.4 + xmax2*0.6
        ty = ymin2*0.25 + ymax2*0.75
        text = ax2.text( tx, ty, r"$\frac{%d}{%d}$" % (numerator, denominator), fontsize=100, alpha=0.2 )
        tx = xmin2*0.95 + xmax2*0.05
        ty = ymin2*0.95 + ymax2*0.05
        self.guide_text = ax2.text( tx, ty, "", fontsize=50, alpha=0.1 )
        self.update_guide_text()

        ax2.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=16)
        if not self.ignore_bq:
            ax2t.legend(bbox_to_anchor=(1.0, 0.92), loc='upper right', fontsize=16)

        self.chisqr_list = []
        anim_rec2 = self.anim_data2[0]
        self.chisqr_list.append(anim_rec2)
        c_data = np.log10( np.array(self.chisqr_list) )
        c_lines = []
        label_texts = [ 'Re-construction', 'A(q)-positivity', 'Base-drift', 'A(q)-smoothness', 'B(q)-smoothness' ]
        self.c_line_indeces = []
        for k in range(c_data.shape[1]):
            if k == 1:
                if not self.aq_positivity:
                    continue
            elif k == 2:
                if not self.add_conc_const:
                    continue
            elif k > 2:
                if not self.aq_smoothness:
                    continue
            scale = self.row_penalty_weights[k]
            # note that c_data already insludes scale
            c_line, = ax1.plot( [0], np.log10(c_data[:,k]), color=get_color(k), label=('%4.2f * ' % scale) + label_texts[k] )
            c_lines.append(c_line)
            self.c_line_indeces.append(k)

        _, ymax1 = ax1.get_ylim()
        ax1.set_xlim( 1, self.n[-1] )
        ymin1 = self.min_data - 0.5 if np.isfinite(self.min_data) else ymin2 - 2
        ymax1 = self.max_data + 2 if np.isfinite(self.max_data) else max(ymax2, ymax1) + 2

        ax1.set_ylim(ymin1, ymax1)
        ax1.set_xscale("log", nonposx='clip')
        ax1.legend(fontsize=16)

        fig.tight_layout()

        def init():
            anim_rec2 = self.anim_data2[0]
            self.chisqr_list = []
            self.chisqr_list.append(anim_rec2)
            text.set_text( r"$\frac{\ }{%d}$" % ( denominator) )
            aq_curve.set_xdata(np.ma.array( self.q, mask=True))
            aq_curve.set_ydata(np.ma.array( aq_y, mask=True))
            if not self.ignore_bq:
                bq_curve.set_xdata(np.ma.array( self.q, mask=True))
                bq_curve.set_ydata(np.ma.array( bq_y, mask=True))
            chisqr_array = np.array(self.chisqr_list)
            for k, c_line in enumerate(c_lines):
                c_line.set_xdata(np.ma.array( self.n[0:chisqr_array.shape[0]], mask=True) )
                c_line.set_ydata(np.ma.array( chisqr_array[:,k], mask=True))

            if self.ignore_bq:
                return tuple( [ text, self.guide_text, aq_curve ] + c_lines )
            else:
                return tuple( [ text, self.guide_text, aq_curve, bq_curve ] + c_lines )

        def animate( anim_tuple ):
            anim_rec1, anim_rec2 = anim_tuple
            numerator = anim_rec1[0]
            P = anim_rec1[1]

            text.set_text( r"$\frac{%d}{%d}$" % (numerator, denominator) )
            aq_curve.set_xdata( self.q )
            aq_curve.set_ydata( np.log10(P[:,0]) )
            # set 0 alpha for the first few iterations to avoid them to remain visible (matplotlib bug?)
            aq_curve.set_alpha( 0 if numerator < 3 else 1 )
            if not self.ignore_bq:
                bq_curve.set_xdata( self.q )
                bq_curve.set_ydata( P[:,1] )
            chisqr_array = np.array(self.chisqr_list)
            chisqr_array[chisqr_array < TOO_SMALL_TO_PLOT] = TOO_SMALL_TO_PLOT
            for k, c_line in enumerate(c_lines):
                k_ = self.c_line_indeces[k]
                c_line.set_xdata( self.n[0:chisqr_array.shape[0]] )
                c_line.set_ydata( np.log10(chisqr_array[:,k_]) )

            if self.ignore_bq:
                return tuple( [ text, self.guide_text, aq_curve ] + c_lines )
            else:
                return tuple( [ text, self.guide_text, aq_curve, bq_curve ] + c_lines )

        def index_generator():
            i = 0
            data_size = len(self.anim_data2)
            while i < data_size + 10:
                i_ = min(data_size-1, i)
                anim_rec2 = self.anim_data2[i_]
                if self.pause:
                    pass
                else:
                    if i == 0:
                        # already appended
                        pass
                    else:
                        if i_ == i:
                            self.chisqr_list.append(anim_rec2)
                    i += 1
                yield self.anim_data1[i_], anim_rec2

        self.anim = animation.FuncAnimation(fig, animate, index_generator, init_func=init,
                    interval=100, blit=True)

        # self.mpl_canvas.draw()

    def on_click(self, event):
        if event.inaxes is None:
            return
        self.pause ^= True
        self.update_guide_text()

    def update_guide_text(self):
        self.guide_text.set_text( "Click to restart" if self.pause else "Click to pause" )
        # note that self.guide_text must be included in the animate() func return
        # for it to be shown updated
