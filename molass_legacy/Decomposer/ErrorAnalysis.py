# coding: utf-8
"""
    ErrorAnalysis.py

    Copyright (c) 2018, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.interpolate import LSQUnivariateSpline

NUM_KNOTS = 40

def smoothed_error( E, debug=False ):
    x = np.arange(E.shape[0])
    knots = np.linspace(x[0], x[-1], NUM_KNOTS)[1:-1]
    SE_list = []
    delta = 0.001
    for j in range(E.shape[1]):
        y = np.log( E[:,j] + delta )
        spline = LSQUnivariateSpline( x, y, knots )
        y_ = spline(x)
        if debug:
            if j == 85:
                debug_plot_smoothing( x, y, y_ )
        SE_list.append(np.exp(y_) - delta)
    return np.array(SE_list).T

def debug_plot_smoothing( x, y, y_ ):
    import molass_legacy.KekLib.DebugPlot as plt
    fig = plt.figure()
    ax = fig.gca()
    ax.plot(x, y, 'o', markersize=3)
    ax.plot(x, y_)
    fig.tight_layout()
    plt.show()

def compute_denoised_data( rank, X ):
    dslice = slice(0,rank)
    U, s, Vt = np.linalg.svd( X )
    X_ = np.dot(  np.dot( U[:,dslice], np.diag( s[dslice] ) ), Vt[dslice, :] )
    return X_

class ErrorAnalyzer:
    def __init__( self, M, E ):
        self.M = M
        self.E = smoothed_error(E)

    def compute_denoised_error( self, rank, num_iter ):
        M = self.M
        S   = np.zeros( M.shape )
        S2  = np.zeros( M.shape )
        S_  = np.zeros( M.shape )
        S_2 = np.zeros( M.shape )
        U_  = np.zeros( M.shape )
        U_2 = np.zeros( M.shape )

        M_ = compute_denoised_data( rank, M )
        Mpinv = np.linalg.pinv( M )
        W = np.dot( M_, Mpinv )

        for i in range(num_iter):
            if i % 100 == 0:
                print( 'iteration', i )
            E_ = self.generate_noise()
            X   = M + E_

            S   += X
            S2  += X**2

            X_ = compute_denoised_data( rank, X )
            S_  += X_
            S_2 += X_**2

            Y_ = np.dot(W, X)
            U_  += Y_
            U_2 += Y_**2

        S /= num_iter
        DE = np.sqrt( ( S2 - num_iter * S**2 )/( num_iter - 1 ) )

        S_ /= num_iter
        DE_ = np.sqrt( ( S_2 - num_iter * S_**2 )/( num_iter - 1 ) )

        U_ /= num_iter
        WE_ = np.sqrt( ( U_2 - num_iter * U_**2 )/( num_iter - 1 ) )

        return DE, DE_, WE_

    def generate_noise( self ):
        X = np.random.randn( *list(self.M.shape) )
        return X * self.E

def debug_plot_error( M, E_ ):
    import molass_legacy.KekLib.DebugPlot as plt
    x = np.arange(E_.shape[0])
    knots = np.linspace(x[0], x[-1], NUM_KNOTS)[1:-1]
    # for j in range(M.shape[1]):
    for j in [85, 210]:
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.set_title('original curve and a sample simulated noisy curve in $Log_{10}$ scale')
        ax2.set_title('abs(noise) in $Log_{10}$ scale')
        y_ = M[:,j]
        e_ = E_[:,j]
        ax1.plot(np.log10(y_), label='data')
        ax1.plot(np.log10(y_ + e_), label='data + noise')
        ae = np.log10(np.abs(e_))
        spline = LSQUnivariateSpline( x, ae, knots )
        ax2.plot(ae, color='yellow', label='noise')
        ax2.plot(spline(x), ':', color='red', label='noise level')
        ax1.legend()
        ax2.legend()
        fig.tight_layout()
        try:
            plt.show()
        except:
            break

def theory_simulation_2D():
    noise_ratio = 0.05
    x = np.linspace(0.001, 1.0, 100)
    size = len(x)
    y = np.exp( -x*4 )
    e = np.random.randn(size)* y * noise_ratio
    def compute_simulated_curve(N):
        s_ = np.zeros(size)
        v_ = np.zeros(size)
        for i in range(N):
            e_ = np.random.randn(size)* y * noise_ratio
            y_ = y + e_
            s_ += y_
            v_ += y_**2
        Y = s_/N
        std = np.sqrt(v_/N - Y**2)
        return Y, std

    num_iter1 = 10
    Y1, std1 = compute_simulated_curve(num_iter1)
    num_iter2 = 100
    Y2, std2 = compute_simulated_curve(num_iter2)

    import molass_legacy.KekLib.DebugPlot as plt
    fig = plt.figure(figsize=(21,7))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)
    ax1.set_title('%d%% gaussian noise (one sample)' % (noise_ratio*100), fontsize=16)
    ax2.set_title('%d%% gaussian noise (result of %d random iterations)' % (noise_ratio*100, num_iter1),  fontsize=16)
    ax3.set_title('%d%% gaussian noise (result of %d random iterations)' % (noise_ratio*100, num_iter2),  fontsize=16)
    ax2t = ax2.twinx()
    ax3t = ax3.twinx()
    ax1.set_ylim(-0.1, 1.1)
    ax1.plot(x, y, label='exp(-x*4)')
    ax1.plot(x, e, label='5% gaussian noise')
    ax1.plot(x, y+e, label='exp(-x*4) + 5% gaussian noise')
    ax1.legend()
    ax2.plot(x, y, label='exp(-x*4)')
    ax2.plot(x, std1, label='STDEV(exp(-x*4))')
    ax2.plot(x, Y1, label='AVERAGE(exp(-x*4))')
    ax2t.set_ylim(0, 0.11)
    ax2t.plot(x, std1/Y1, ':', label='noise rate (STDEV/AVERAGE)')
    ax2.legend()
    ax2t.legend(bbox_to_anchor=(1, 0.85), loc='upper right')
    ax3.plot(x, y, label='exp(-x*4)')
    ax3.plot(x, std2, label='STDEV(exp(-x*4))')
    ax3.plot(x, Y2, label='AVERAGE(exp(-x*4))')
    ax3t.set_ylim(0, 0.11)
    ax3t.plot(x, std2/Y2, ':', label='noise rate (STDEV/AVERAGE)')
    ax3.legend()
    ax3t.legend(bbox_to_anchor=(1, 0.85), loc='upper right')
    fig.tight_layout()
    plt.show()

import molass_legacy.KekLib.DebugPlot as plt
import matplotlib.animation as animation

class McAnimation:
    def __init__(self):
        # import matplotlib.pyplot as plt
        self.pause = False

        noise_ratio = 0.05
        x = np.linspace(0.001, 1.0, 100)
        size = len(x)
        y = np.exp( -x*4 )

        num_iter = 1000
        e_list = []
        for i in range(num_iter):
            e = np.random.randn(size)* y * noise_ratio
            e_list.append(e)

        self.x = x
        self.y = y
        self.e_list = e_list
        self.num_iter = num_iter

        fig = plt.figure(figsize=(14,7))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        plt.dp.mpl_canvas.mpl_connect('button_press_event', self.on_click)

        ax1.set_title("Histogram of %d errors (noise ratio %.1g%%) at x=0" % (num_iter, noise_ratio*100), fontsize=16)
        ax2.set_title("Montecarlo simulation of noisy $e^{-4x}$  by %d iterations" % num_iter, fontsize=16)

        e_array = np.array(e_list)
        ax1.hist(e_array[:,0])
        ax2.plot(x, y, label='$e^{-4x}$')

        e = e_list[0]
        self.e_curve, = ax2.plot(x, e, label='%.1g%% noise' % (noise_ratio*100))
        self.y_curve, = ax2.plot(x, y+e, label='$e^{-4x}$ + noise')
        self.text = ax2.text(0.5,0.5, "", fontsize=80, alpha=0.3)

        ax2.legend(fontsize=16)
        fig.tight_layout()
        self.anim = animation.FuncAnimation(fig, self.animate, self.index_generator, init_func=self.anim_init,
                        interval=100, blit=True)
        # anim.save("Noise-MonteCarlo.mp4")
        self.fig = fig
        self.ax1 = ax1
        self.ax2 = ax2
        plt.show()

    def index_generator(self):
        i = 0
        while i < self.num_iter:
            i_ = i
            if self.pause:
                pass
            else:
                i += 1
            yield i_

    def on_click(self, event):
        if event.inaxes is None:
            return
        self.pause ^= True

    def anim_init(self):
        x = self.x
        y = self.y
        e = self.e_list[0]
        self.e_curve.set_xdata(np.ma.array(x, mask=True))
        self.e_curve.set_ydata(np.ma.array(e, mask=True))
        self.y_curve.set_xdata(np.ma.array(x, mask=True))
        self.y_curve.set_ydata(np.ma.array(y+e, mask=True))
        self.text.set_text("")
        return (self.e_curve, self.y_curve, self.text)

    def animate(self, i):
        x = self.x
        y = self.y
        e = self.e_list[i]
        self.e_curve.set_xdata(x)
        self.e_curve.set_ydata(e)
        self.y_curve.set_xdata(x)
        self.y_curve.set_ydata(y+e)
        self.text.set_text(str(i))
        return (self.e_curve, self.y_curve, self.text)

def simulated_svd_error_propagation_plot(x, M, E, M_, E_, SE, PE, WE_, eno, formula=True, mpmc=False):
    import molass_legacy.KekLib.DebugPlot    as plt

    if formula:
        fig = plt.figure(figsize=(21,7))
        ax0  = fig.add_subplot(131)
        ax1  = fig.add_subplot(132)
        ax2  = fig.add_subplot(133)
    else:
        fig = plt.figure(figsize=(14,7))
        ax1  = fig.add_subplot(121)
        ax2  = fig.add_subplot(122)

    if formula:
        ax0.set_title("Error propagation in denoising with SVD using MR-inverve")
        ax0.axes.get_xaxis().set_ticks([])
        ax0.axes.get_yaxis().set_ticks([])
        ax0.text(0.1,0.85, r"$M = U \cdot \Sigma \cdot V^T$", fontsize=30)
        ax0.text(0.1,0.65, r"$\widetilde{M} = \widetilde{U} \cdot \widetilde{\Sigma} \cdot  \widetilde{V}^T $", fontsize=30)
        ax0.text(0.1,0.45, r"$\widetilde{M} \approx  W \cdot M$", fontsize=30)
        ax0.text(0.1,0.25, r"$W \approx  W \cdot M \cdot M^+ \approx  \widetilde{M} \cdot M^+$", fontsize=30)
        ax0.text(0.1,0.05, r"$\widetilde{E} = \sqrt{ W^{(2)} \cdot E^{(2)} }$", fontsize=30)

    """
        [ a b ]  [x]
        [ c d ]  [y]
        ax + by
        cx + dy
    """

    ax1.set_title( "Propagated errors in linear scale at eno=%d" % eno )
    ax1.set_xlabel('Q')
    ax1.set_ylabel('Intensity')
    ax1.plot(x, M[eno, :], label='data')
    ax1.plot(x, E[eno, :], label='error')
    ax1.plot(x, M_[eno, :], label='denoised data')
    ax1.plot(x, E_[eno, :], label='error (MC)')
    ax1.plot(x, SE[eno, :], label='error of denoised data (MC)')
    ax1.plot(x, PE[eno, :], label='error of denoised data (matrix product)')
    if mpmc:
        ax1.plot(x, WE_[eno, :], label='error of denoised data (matrix product MC)')

    ax1.legend()
    ax2.set_title( "Propagated errors in $Log_{10}$ scale at eno=%d" % eno )
    ax2.set_xlabel('Q')
    ax2.set_ylabel('$Log_{10}(Intensity)$')
    ax2.plot(x, np.log10(M[eno, :]), label='data')
    ax2.plot(x, np.log10(E[eno, :]), label='error')
    ax2.plot(x, np.log10(M_[eno, :]), label='denoised data')
    ax2.plot(x, np.log10(E_[eno, :]), label='error (MC)')
    ax2.plot(x, np.log10(SE[eno, :]), label='error of denoised data (MC)')
    ax2.plot(x, np.log10(PE[eno, :]), label='error of denoised data (matrix produc)')
    if mpmc:
        ax2.plot(x, np.log10(WE_[eno, :]), label='error of denoised data (matrix produc MC)')
    ax2.legend()
    fig.tight_layout()
    plt.show()
