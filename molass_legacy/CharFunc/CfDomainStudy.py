"""
    CharFunc/CfDomainStudy.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt

def study1():
    from SecTheory.BasicModels import single_pore_pdf
    from SecTheory.SecCF import simple_phi, shifted_phi
    from SecTheory.SecPDF import FftInvImpl

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20,8))
    ax1, ax2, ax3, ax4 = axes[0,:]
    ax5, ax6, ax7, ax8 = axes[1,:]
    ax1.set_title("PDF Domain")
    ax2.set_title("CF Real Domain")
    ax3.set_title("CF Imaginary Domain")
    x = np.arange(21, 321)
    x0 = 20
    npi = 100
    tpi = 1
    y1 = single_pore_pdf(x, npi, tpi)
    y2 = single_pore_pdf(x - x0, npi, tpi)
    # y3 = single_pore_pdf(x - x0 - 10, npi, tpi)
    ty = y1 + y2
    N = 1024
    tz = np.fft.fft(ty, N)
    w = np.fft.fftfreq(N)
    u = np.fft.ifft(tz, len(x))
    z1 = np.fft.fft(y1, N)
    z2 = np.fft.fft(y2, N)
    # z3 = np.fft.fft(y3)

    """
    relation between w and w_
    """

    finv = FftInvImpl()
    w_ = finv.get_w()
    z1_ = simple_phi(w_, npi, tpi)
    u1_ = finv.compute(x, z1_)

    z2_ = shifted_phi(w_, npi, tpi, x0)
    u2_ = finv.compute(x, z2_)

    def objective(p):
        z_ = shifted_phi(w_, npi, tpi, p[0])
        sz = z1_ + z_
        return (np.real(sz) - np.real(tz))**2 + (np.imag(sz) - np.imag(tz))**2

    # res = minimize(objective, [10])
    # print("res.x", res.x)

    ax1.plot(x, ty, label="data")
    ax1.plot(x, y1, ":", label="component-1")
    ax1.plot(x, y2, ":", label="component-2")
    # ax1.plot(x, y3, ":", label="component-3")

    ax1.plot(x, u1_, ":", color="yellow", label="component-1 inverse FFT")
    ax1.plot(x, u2_, ":", color="yellow", label="component-2 inverse FFT")

    tu_ = finv.compute(x, z1_ + z2_)
    ax1.plot(x, tu_, ":", color="red", label="sum inverse FFT")

    print("real(tz)=", np.real(tz))

    ax2.plot(w, np.real(tz), "o", markersize=1, alpha=0.5)
    p2, = ax2.plot(w[0], np.real(tz)[0], "o", color="red", markersize=5, alpha=0.5)
    p2_, = ax2.plot(w[-1], np.real(tz)[-1], "o", color="blue", markersize=5, alpha=0.5)
    ax3.plot(w, np.imag(tz), "o", markersize=1, alpha=0.5)
    p3, = ax3.plot(w[0], np.imag(tz)[0], "o", color="red", markersize=5, alpha=0.5)
    p3_, = ax3.plot(w[-1], np.imag(tz)[-1], "o", color="blue", markersize=5, alpha=0.5)
    ax2.plot(w, np.real(z1), "o", markersize=1, alpha=0.5)
    ax2.plot(w, np.real(z2), "o", markersize=1, alpha=0.5)
    ax3.plot(w, np.imag(z1), "o", markersize=1, alpha=0.5)
    ax3.plot(w, np.imag(z2), "o", markersize=1, alpha=0.5)

    if False:
        ax1.plot(x, u, ":", color="cyan", label="inverse FFT")
        su = np.fft.ifft(z1 + z2, len(x))
        ax1.plot(x, su, ":", color="red", label="inverse FFT of sum")
    ax1.legend()

    ax4.plot(w, "o", markersize=1, label="w")
    ax4.plot(w_/(2*np.pi), "o", markersize=1, label="w_")
    v_ = np.concatenate([w_[N//2:], w_[:N//2]])
    ax4.plot(v_/(2*np.pi), "o", markersize=1, label="v_")
    ax4.legend()

    tz_ = z1_ + z2_
    print("diff w : v_/(2pi) =", np.max(np.abs(w - v_/(2*np.pi))))
    print("diff tz : tz_ =", np.max(np.abs(np.real(tz) - np.real(tz_))))
    # sz_ = np.concatenate([np.flip(tz_[N//2:]), np.flip(tz_[:N//2])])
    # print("diff tz : sz_ =", np.max(np.abs(np.real(tz) - np.real(sz_))))

    for ax in [ax5, ax8]:
        ax.set_axis_off()

    sz_ = np.concatenate([tz_[N//2:], tz_[:N//2]])
    ax6.plot(v_, np.real(sz_), "o", markersize=1, alpha=0.5)
    p6, = ax6.plot(v_[0], np.real(sz_)[0], "o", color="red", markersize=5, alpha=0.5)
    p6_, = ax6.plot(v_[-1], np.real(sz_)[-1], "o", color="blue", markersize=5, alpha=0.5)
    ax7.plot(v_, np.imag(sz_), "o", markersize=1, alpha=0.5)
    p7, = ax7.plot(v_[0], np.imag(sz_)[0], "o", color="red", markersize=5, alpha=0.5)
    p7_, = ax7.plot(v_[-1], np.imag(sz_)[-1], "o", color="blue", markersize=5, alpha=0.5)    
    ax6.plot(w_, np.real(z1_), "o", markersize=1, alpha=0.5)
    ax6.plot(w_, np.real(z2_), "o", markersize=1, alpha=0.5)
    ax7.plot(w_, np.imag(z1_), "o", markersize=1, alpha=0.5)
    ax7.plot(w_, np.imag(z2_), "o", markersize=1, alpha=0.5)

    def on_click(event):
        nonlocal p2, p6, p3, p7
        if event.xdata and event.ydata:
            if event.inaxes == ax2:
                p, q = p2, p6
            elif event.inaxes == ax3:
                p, q = p3, p7
            else:
                return
            
            index = np.argmin((w - event.xdata)**2 + (np.real(tz) - event.ydata)**2)
            print("index=", index)
            px, py = w[index], np.real(tz)[index]
            p.set_data([px], [py])
            px_, py_ = v_[index], np.real(sz_)[index]
            q.set_data([px_], [py_])
            fig.canvas.draw()

    fig.tight_layout()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

def study2():
    from SecTheory.BasicModels import single_pore_pdf
    from SecTheory.SecCF import simple_phi, shifted_phi
    from SecTheory.SecPDF import BidirectFft

    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20,8))
    fig.suptitle("Proof of Numerical Fourier Transform Inversion of the Sum of Two Monopore CFs", fontsize=20)
    ax1, ax2, ax3, ax4 = axes[0,:]
    ax5, ax6, ax7, ax8 = axes[1,:]
    ax1.set_title("PDF Domain")
    ax2.set_title("CF Real Domain")
    ax3.set_title("CF Imaginary Domain")
    ax4.set_title("Frequency Vector Values")
    x = np.arange(21, 321)
    x0 = 20
    npi = 100
    tpi = 1
    y1 = single_pore_pdf(x, npi, tpi)
    y2 = single_pore_pdf(x - x0, npi, tpi)
    ty = y1 + y2

    bifft = BidirectFft(x)

    tz = bifft.compute_z(ty)
    w = bifft.get_w()
    u = bifft.compute_y(tz)

    z1 = bifft.compute_z(y1)
    z1_ = simple_phi(w, npi, tpi)
    u1_ = bifft.compute_y(z1_)

    z2 = bifft.compute_z(y2)
    z2_ = shifted_phi(w, npi, tpi, x0)
    u2_ = bifft.compute_y(z2_)

    # res = minimize(objective, [10])
    # print("res.x", res.x)

    ax1.plot(x, ty, label="data")
    ax1.plot(x, y1, ":", label="component-1")
    ax1.plot(x, y2, ":", label="component-2")
    # ax1.plot(x, y3, ":", label="component-3")

    ax1.plot(x, u1_, ":", color="yellow", label="component-1 inverse FFT")
    ax1.plot(x, u2_, ":", color="yellow", label="component-2 inverse FFT")

    tu_ = bifft.compute_y(z1_ + z2_)
    ax1.plot(x, tu_, ":", color="red", label="sum inverse FFT")

    print("real(tz)=", np.real(tz))

    ax2.plot(w, np.real(tz), "o", markersize=1, alpha=0.5)
    p2, = ax2.plot(w[0], np.real(tz)[0], "o", color="red", markersize=5, alpha=0.5)
    p2_, = ax2.plot(w[-1], np.real(tz)[-1], "o", color="blue", markersize=5, alpha=0.5)
    ax3.plot(w, np.imag(tz), "o", markersize=1, alpha=0.5)
    p3, = ax3.plot(w[0], np.imag(tz)[0], "o", color="red", markersize=5, alpha=0.5)
    p3_, = ax3.plot(w[-1], np.imag(tz)[-1], "o", color="blue", markersize=5, alpha=0.5)
    ax2.plot(w, np.real(z1), "o", markersize=1, alpha=0.5)
    ax2.plot(w, np.real(z2), "o", markersize=1, alpha=0.5)
    ax3.plot(w, np.imag(z1), "o", markersize=1, alpha=0.5)
    ax3.plot(w, np.imag(z2), "o", markersize=1, alpha=0.5)
    ax1.legend()

    ax4.plot(w, "o", markersize=1, label="w")
    ax4.legend()

    tz_ = z1_ + z2_
    diff_tz = np.abs(np.real(tz) - np.real(tz_))
    print("real diff tz : tz_ =", np.max(diff_tz))
    ax5.set_title("Difference in CF Real Domain")
    ax5.plot(w, diff_tz)
    diff_tz = np.abs(np.imag(tz) - np.imag(tz_))
    print("imag diff tz : tz_ =", np.max(diff_tz))
    ax8.set_title("Difference in CF Imaginary Domain")
    ax8.plot(w, diff_tz)

    ax6.plot(w, np.real(tz_), "o", markersize=1, alpha=0.5)
    p6, = ax6.plot(w[0], np.real(tz_)[0], "o", color="red", markersize=5, alpha=0.5)
    p6_, = ax6.plot(w[-1], np.real(tz_)[-1], "o", color="blue", markersize=5, alpha=0.5)
    ax7.plot(w, np.imag(tz_), "o", markersize=1, alpha=0.5)
    p7, = ax7.plot(w[0], np.imag(tz_)[0], "o", color="red", markersize=5, alpha=0.5)
    p7_, = ax7.plot(w[-1], np.imag(tz_)[-1], "o", color="blue", markersize=5, alpha=0.5)    
    ax6.plot(w, np.real(z1_), "o", markersize=1, alpha=0.5)
    ax6.plot(w, np.real(z2_), "o", markersize=1, alpha=0.5)
    ax7.plot(w, np.imag(z1_), "o", markersize=1, alpha=0.5)
    ax7.plot(w, np.imag(z2_), "o", markersize=1, alpha=0.5)

    def on_click(event):
        nonlocal p2, p6, p3, p7
        if event.xdata and event.ydata:
            if event.inaxes == ax2:
                p, q = p2, p6
            elif event.inaxes == ax3:
                p, q = p3, p7
            else:
                return
            
            index = np.argmin((w - event.xdata)**2 + (np.real(tz) - event.ydata)**2)
            print("index=", index)
            px, py = w[index], np.real(tz)[index]
            p.set_data([px], [py])
            px_, py_ = w[index], np.real(tz_)[index]
            q.set_data([px_], [py_])
            fig.canvas.draw()

    fig.tight_layout()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

    # study1()
    study2()