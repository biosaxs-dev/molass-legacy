"""
    AveragingDiscussion.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import copy
import numpy as np

def extrapolate(M, c, rank=2):
    U, s, VT = np.linalg.svd(M)
    M_  = U[:,0:rank] @ np.diag(s[0:rank]) @ VT[0:rank,:]
    c_  = c/np.max(c)
    C   = np.array([c_, c_**2])
    """
        P・C = M
        P = M・Cpinv
    """
    Cpinv = np.linalg.pinv(C)
    P   = M_ @ Cpinv
    return P

class AveragingDiscussion:
    def __init__(self, sd):
        import molass_legacy.KekLib.DebugPlot as plt
        self.mouse_axis = None

        self.q = sd.intensity_array[0,:,0]
        self.data = sd.intensity_array[:,:,1].T
        self.e_curve = sd.xray_curve

        peak_i = self.e_curve.primary_peak_i
        xslice = slice(0,100)
        yslice = slice(peak_i-50, peak_i+50)
        wslice = slice(peak_i-20, peak_i)

        x = self.q[xslice]
        y = np.arange(self.data.shape[1])[yslice]
        xx, yy = np.meshgrid( x, y )

        i = np.arange(xslice.start, xslice.stop)
        j = np.arange(yslice.start, yslice.stop)
        ii, jj = np.meshgrid( i, j )

        zz = self.data[ii, jj]

        self.fig = fig = plt.figure(figsize=(30, 8))
        fig.canvas.mpl_connect( 'draw_event', self.on_draw )
        fig.canvas.mpl_connect( 'button_press_event', self.on_button_press )
        ax1 = fig.add_subplot(131, projection='3d')
        ax2 = fig.add_subplot(132, projection='3d')
        ax3 = fig.add_subplot(133, projection='3d')
        self.axes = axes = [ax1, ax2, ax3]

        ax1.set_title("Five-line Average along Elutional Axis", fontsize=20)
        ax2.set_title("Extrapolation from 20 Five-line Averages", fontsize=20)
        ax3.set_title("Extrapolation from 20 raw lines", fontsize=20)

        for ax in axes:
            ax.plot_surface(xx, yy, zz, color='green', alpha=0.1 )
            x_ = np.ones(len(y))*0.02
            z_ = self.e_curve.y[yslice]
            ax.plot(x_, y, z_, color='orange', linewidth=3)

        for k in range(70, 75):
            y_ = np.ones(len(x))*k
            z_ = self.data[xslice,k]
            ax1.plot(x, y_, z_, color='pink')

        k = 72
        y_ = np.ones(len(x))*k
        z_ = np.average(self.data[xslice,70:75], axis=1)
        ax1.plot(x, y_, z_, color='yellow', linewidth=3, label='5L average')
        ax1.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=20)

        def comparison_plot(ax, data, conc_y, color, alpha):
            M   = data[xslice, wslice]
            c   = conc_y[wslice]
            P   = extrapolate(M, c)
            w   = P[:,0]

            for k in range(wslice.start, wslice.stop):
                y_ = np.ones(len(x))*k
                z_ = data[xslice,k]
                ax.plot(x, y_, z_, color=color, alpha=alpha)

            k = peak_i-10
            y_ = np.ones(len(x))*k
            z_ = np.average(data[xslice,peak_i-20:peak_i], axis=1)
            ax.plot(x, y_, z_, color='cyan', linewidth=3, label='20L average')

            y_ = np.ones(len(x))*peak_i
            ax.plot(x, y_, w, color='red', linewidth=3, label='extrapolated')
            zs = z_ * w[-1]/z_[-1]
            ax.plot(x, y_, zs, color='lime', linewidth=3, label='20L average scaled')

            ax.legend(bbox_to_anchor=(1, 0), loc='lower right', fontsize=20)

        averaged_data = copy.deepcopy(self.data)
        averaged_c = copy.deepcopy(self.e_curve.y)
        n = 5
        m = n//2
        for j in range(self.data.shape[1]-n):
            averaged_data[:,j+m] = np.average(self.data[:,j:j+n], axis=1)
            averaged_c[j+m] = np.average(self.e_curve.y[j:j+n])

        comparison_plot(ax2, averaged_data, averaged_c,'yellow', 1)
        comparison_plot(ax3, self.data, self.e_curve.y, 'green', 0.5)

        fig.tight_layout()
        plt.show()

    def on_button_press(self, event):
        if event.inaxes in self.axes:
            self.mouse_axis = event.inaxes

    def on_draw(self, event):
        if self.mouse_axis is None:
            return

        import molass_legacy.KekLib.DebugPlot as plt
        ax1, ax2, ax3 = self.axes
        if self.mouse_axis == ax1:
            this_ax, oppo_axes = ax1, [ax2, ax3]
        elif self.mouse_axis == ax2:
            this_ax, oppo_axes = ax2, [ax1, ax3]
        elif self.mouse_axis == ax3:
            this_ax, oppo_axes = ax3, [ax1, ax2]
        else:
            return

        # TODO: investigate the cause of delayed synchronization
        self.mouse_axis = None
        for ax in oppo_axes:
            ax.view_init( this_ax.elev, this_ax.azim )
        plt.dp.mpl_canvas.draw()
