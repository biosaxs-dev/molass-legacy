# coding: utf-8
"""
    SimpleGuinierAnimation.py

    Copyright (c) 2017-2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy as np
import matplotlib.pyplot    as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d   import Axes3D
from SimpleGuinier          import KNOT_NO_ADDITION
from SimpleGuinierScore     import NUM_SCORE_FACTORS, USE_FWD_CONSISTENCY

if USE_FWD_CONSISTENCY:
    SCORE_COLORS = [ 'blue', 'purple', 'magenta', 'pink' ]
else:
    SCORE_COLORS = [ 'blue', 'purple', 'magenta' ]
assert len(SCORE_COLORS) == NUM_SCORE_FACTORS

PLOT_BEST_HISTORY = False

class SimpleGuinierAnimation:
    def __init__( self, guinier, title, result_rg=None ):
        self.guinier    = guinier
        self.title      = title
        self.result_rg  = result_rg
        self.mpl_canvas_widget = None

    def destroy( self ):
        if self.mpl_canvas_widget is not None:
            print( '__del__' )
            self.mpl_canvas_widget.destroy()

    def draw( self, fig, interval=100, anim_iter_max=200, skip_first_stage=True ):
        if PLOT_BEST_HISTORY:
            ax1 = fig.add_subplot(231)
            ax2 = fig.add_subplot(232)
            ax3 = fig.add_subplot(233)
            ax5 = fig.add_subplot(235, projection='3d')
        else:
            ax1 = fig.add_subplot(131)
            ax2 = fig.add_subplot(132)
            ax3 = fig.add_subplot(133)

        x2      = self.guinier.x2
        log_y   = self.guinier.log_y

        if self.guinier.qrg_stop is None:
            for ax in [ax1, ax2, ax3]:
                ax.plot( x2, log_y, 'o', markersize=3 )
                xmin, xmax = ax.get_xlim()
                ymin, ymax = ax.get_ylim()
                ax.text( (0.7*xmin + 0.3*xmax), (ymin + ymax)/2, 'No Guinier interval' )
            return

        px2     = self.guinier.px2
        log_py  = self.guinier.log_py
        kx2     = self.guinier.kx2
        log_ky  = self.guinier.log_ky
        log_sy  = self.guinier.log_sy
        # log_y_max, log_y_min = np.percentile(log_y, [100,0])

        # TODO: in case self.guinier.guinier_start is None
        gn_min = log_y[self.guinier.qrg_stop]
        gn_max = log_y[self.guinier.guinier_start]
        if gn_min > gn_max:
            gn_min, gn_max = gn_max, gn_min

        ymin2 = gn_min - 1.0
        ymax2 = max( gn_max + 1.0, self.guinier.log_py + 0.2 )
        ax2.set_xlim( -0.0005, self.guinier.kx2[-3] * 1.1 )
        ax2.set_ylim( ymin2, ymax2 )
        xmax3 = self.guinier.x2[self.guinier.qrg_stop]
        ax3.set_xlim( max(-0.00025, -xmax3/6 ), xmax3 )
        ax3.set_ylim( min( self.guinier.guinier_y[1], (ymin2+ymax2)/2)-0.05, (self.guinier.guinier_y[0] + ymax2)/2 )

        knot_points = []
        for i, ax in enumerate([ax1, ax2, ax3]):
            if self.title is not None:
                ax.set_title( self.title )
            ax.set_xlabel( 'Q²' )
            ax.set_ylabel( 'Ln(I)' )
            ax.plot( x2, log_y, 'o', markersize=3 )
            ax.plot( x2, log_sy, color='green', alpha=0.5 )
            msize = 5 if i == 0 else 8

            point, = ax.plot( px2, log_py, 'o', color='red', markersize=msize )
            knot_points.append( point )
            point, = ax.plot( kx2, log_ky, 'o', color='yellow', markersize=msize )
            knot_points.append( point )
            if self.guinier.added_knot_type != KNOT_NO_ADDITION:
                point, = ax.plot( kx2[1], log_ky[1], 'o', color='pink', markersize=msize )
                knot_points.append( point )

        fig.tight_layout()

        if len(self.guinier.anim_ag_lines) > 0:
            ag_x, ag_y = self.guinier.anim_ag_lines[0]
        gstop_x = self.guinier.x2[self.guinier.anim_guinier_stop]
        anim_cand_len = len(self.guinier.anim_cand_list)
        if anim_cand_len > 0:
            start, stop, cx, cy, slope, intercept, score, score_vector, rg = self.guinier.anim_cand_list[0]
            endoints = [0, -1]
            cxe = cx[endoints]
            cye = slope * cxe + intercept
        else:
            score = 0
            score_vector = []

        lines = []
        glines = []
        clines = []
        rglines = []
        dlines = []
        bline_params = []
        ft_texts = []
        fline_bgs = []
        flines = []
        sline_bgs = []
        slines = []
        rg_texts = []
        final_rg_texts = []

        for ax in [ax2, ax3]:
            if len(self.guinier.anim_ag_lines) > 0:
                line, = ax.plot( ag_x, ag_y, color='orange', marker='o', markersize=8 )
                lines.append( line )
            gline, = ax.plot( [ gstop_x, gstop_x ], [ ymin2, ymax2 ], ':', color='gray', alpha=0.5 )
            glines.append( gline )

            if anim_cand_len > 0:
                cline, = ax.plot( cx, cy, color='yellow' )
                rgline, = ax.plot( cxe, cye, color='orange', marker='o', markersize=8 )
                clines.append( cline )
                rglines.append( rgline )

            dline, = ax.plot( self.guinier.guinier_x, self.guinier.guinier_y, color='cyan', marker='o', markersize=8 )
            dlines.append( dline )
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            bx = 0.5*xmin + 0.5*xmax
            by = 0.1*ymin + 0.9*ymax
            bsize = (xmax - xmin)*0.4
            byoffset = (ymax-ymin)*0.05
            bline_params.append( [bx, by, bsize, byoffset] )

            # start-stop
            ax.text( bx-bsize*0.34, by, 'From-To:', verticalalignment='center' )
            ft_text = ax.text( bx, by, '', verticalalignment='center' )
            ft_texts.append( ft_text )

            # Factors
            by_f = by - byoffset
            ax.text( bx-bsize*0.3, by_f, 'Factors:', verticalalignment='center' )
            by_f_ = [ by_f, by_f ]
            fline_bg, = ax.plot( [bx, bx+bsize], by_f_, color='white', linewidth=10, solid_capstyle='butt' )
            fline_bgs.append( fline_bg )
            bxstt = bx
            for s, score in enumerate(score_vector):
                bxend = bxstt + bsize*score
                bx_ = [bxstt, bxend]
                fline, = ax.plot( bx_, by_f_, color=SCORE_COLORS[s], linewidth=10, solid_capstyle='butt' )
                bxstt = bxend
                flines.append( fline )

            # Score
            by_s = by - byoffset*2
            ax.text( bx-bsize*0.25, by_s, 'Score:', verticalalignment='center' )
            by_s_ = [by_s, by_s]
            sline_bg, = ax.plot( [bx, bx+bsize], by_s_, color='white', linewidth=10, solid_capstyle='butt' )
            sline_bgs.append( sline_bg )
            sline, = ax.plot( [bx, bx], by_s_, color='green', linewidth=10, solid_capstyle='butt' )
            slines.append( sline )

            # Rg
            by_rg = by - byoffset*3
            ax.text( bx-bsize*0.13, by_rg, 'Rg:', verticalalignment='center' )
            rg_text = ax.text( bx, by_rg, '', verticalalignment='center' )
            rg_texts.append( rg_text )

            # final Rg
            by_rg = by - byoffset*4
            ax.text( bx-bsize*0.34, by_rg, 'Final Rg:', verticalalignment='center' )
            rg_text = ax.text( bx, by_rg, '', verticalalignment='center' )
            final_rg_texts.append( rg_text )

        if PLOT_BEST_HISTORY:
            best_x = []
            best_y = []
            best_z = []
            for k, rec in self.guinier.best_history_dict.items():
                v, rg = rec
                size_score = v[0]
                print( '(%3d, %3d) rg=%.3g, size_score=%.4g' % ( k[0], k[1], rg, size_score ) )
                best_x.append( k[0] )
                best_y.append( k[1] )
                best_z.append( size_score )

            ax5.plot( best_x, best_y, best_z )

        def init():
            for line in lines:
                line.set_ydata(np.ma.array( ag_x, mask=True))
            for gline in glines:
                gline.set_ydata(np.ma.array( [ gstop_x, gstop_x ], mask=True))
            for cline in clines:
                cline.set_xdata(np.ma.array( cx, mask=True))
                cline.set_ydata(np.ma.array( cy, mask=True))
            for rgline in rglines:
                rgline.set_xdata(np.ma.array( cxe, mask=True))
                rgline.set_ydata(np.ma.array( cye, mask=True))
            for dline in dlines:
                dline.set_xdata(np.ma.array( self.guinier.guinier_x, mask=True))
                dline.set_ydata(np.ma.array( self.guinier.guinier_y, mask=True))
            for sline in slines:
                sline.set_xdata(np.ma.array( [bx, bx], mask=True))
                sline.set_ydata(np.ma.array( by_s_, mask=True))
            return tuple( lines + glines + clines + rglines + slines + dlines )

        num_iter_cand = min( anim_iter_max, len( self.guinier.anim_cand_list ) )
        result_rg = self.guinier.Rg if self.result_rg is None else self.result_rg

        def draw_factors( a_score_vector, a_score ):
            for k in range(2):
                bx, by, bsize, byoffset = bline_params[k]
                by_f = by - byoffset
                by_f_ = [ by_f, by_f ]

                # score factors
                bxstt = bx
                for s, score in enumerate(a_score_vector):
                    bxend = bxstt + bsize*score
                    bx_ = [bxstt, bxend]
                    fline = flines[NUM_SCORE_FACTORS*k + s]
                    fline.set_xdata( bx_ )
                    fline.set_ydata( by_f_ )
                    bxstt = bxend

                # score
                bx_s_ = [ bx, bx + bsize*a_score ]
                by_s = by - byoffset*2
                by_s_ = [ by_s, by_s ]

                sline = slines[k]
                sline.set_xdata( bx_s_ )
                sline.set_ydata( by_s_ )

        def animate( i ):
            if i < len(self.guinier.anim_ag_lines):
                ag_x, ag_y = self.guinier.anim_ag_lines[i]
                for line in lines:
                    line.set_xdata( ag_x )
                    line.set_ydata( ag_y )
                return tuple( lines + knot_points )
            else:
                j = i - len(self.guinier.anim_ag_lines)
                if j == 0:
                    for gline in glines:
                        gline.set_ydata( [ ymin2, ymax2 ] )
                    return tuple( glines )
                elif j < num_iter_cand:

                    # draw also the final guinier line in early stages
                    for dline in dlines:
                        dline.set_xdata( self.guinier.guinier_x )
                        dline.set_ydata( self.guinier.guinier_y )

                    start, stop, cx, cy, slope, intercept, score, score_vector, rg = self.guinier.anim_cand_list[j]
                    for cline in clines:
                        cline.set_xdata( cx )
                        cline.set_ydata( cy )
                    cxe = cx[endoints]
                    cye = slope * cxe + intercept
                    for rgline in rglines:
                        rgline.set_xdata( cxe )
                        rgline.set_ydata( cye )

                    for ft_text in ft_texts:
                        ft_text.set_text( '%3d-%3d' % ( start, stop ) )

                    draw_factors(score_vector, score)

                    for rg_text in rg_texts:
                        rg_text.set_text( '%.3g' % rg )

                    for rg_text in final_rg_texts:
                        rg_text.set_text( '%.3g' % result_rg )

                    return tuple( dlines + glines + clines + rglines + ft_texts + fline_bgs + flines + slines + rg_texts + final_rg_texts + knot_points )
                else:
                    for dline in dlines:
                        dline.set_xdata( self.guinier.guinier_x )
                        dline.set_ydata( self.guinier.guinier_y )

                    if anim_cand_len > 0:
                        for ft_text in ft_texts:
                            ft_text.set_text( '%3d-%3d' % ( self.guinier.guinier_start, self.guinier.guinier_stop ) )
                        draw_factors(self.guinier.score_vector, self.guinier.score)

                    for rg_text in rg_texts:
                        rg_text.set_text( '%.3g' % result_rg )

                    for rg_text in final_rg_texts:
                        rg_text.set_text( '%.3g' % result_rg )

                    return tuple( dlines + ft_texts + fline_bgs + flines + slines + rg_texts + final_rg_texts + knot_points )

        num_pause = 50
        print( 'num_iter_cand=', num_iter_cand )
        num_iter_first = len(self.guinier.anim_ag_lines)
        if skip_first_stage:
            range_ = np.arange( num_iter_first, num_iter_first+num_iter_cand+num_pause )
        else:
            range_ = np.arange(1, num_iter_first+num_iter_cand+num_pause )
        self.anim = animation.FuncAnimation(fig, animate, range_, init_func=init,
                    interval=interval, blit=True)
