# coding: utf-8
"""
    DenssFitData.py

    Copyright (c) 2020-2021, SAXS Team, KEK-PF
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import Slider, Button, RadioButtons, TextBox, CheckButtons #, RangeSlider
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ScrolledText
from molass_legacy.KekLib.TkSupplements import set_icon
from StdoutRedirector import StdoutRedirector

class DenssFitDataDialog(Dialog):
    def __init__(self, parent, sasrec, work_info, infile_name, out_folder):
        self.sasrec = sasrec
        self.work_info = work_info
        self.infile_name = infile_name
        self.out_folder = out_folder
        self.applied = False
        Dialog.__init__(self, parent, "Figure in denss.fit_data.py", visible=False )

    def show(self):
        self._show()

    def body(self, body_frame):
        set_icon( self )

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.log_text = ScrolledText(body_frame, height=8)
        self.log_text.pack(fill=Tk.X)
        self.redirexctor = StdoutRedirector(self.log_text)

        fig = self.create_figure(cframe)
        name, ext = os.path.split(self.infile_name)
        output = os.path.join(self.out_folder, name + '_fit')
        self.buttons = fit_data_plot(self.sasrec, self.work_info, fig, output)
        self.mpl_canvas.draw()

    def create_figure(self, cframe):
        self.fig = fig = plt.figure(figsize=(14,7))
        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        return fig

    def apply(self):
        self.applied = True

    def get_sasrec(self):
        global sasrec
        return sasrec

def fit_data_plot(init_sasrec, work_info, fig, output):
    from .saxstats import saxstats as saxs
    global sasrec

    args = work_info.args
    alpha = work_info.alpha
    n1 = work_info.n1
    n2 = work_info.n2
    Iq_orig = work_info.Iq_orig

    sasrec = init_sasrec
    qc = sasrec.qc
    D = sasrec.D            # has been initialized by saxs.estimate_dmax(Iq) in DenssUtils.fit_data_impl
    Iq = work_info.Iq
    Icerr = sasrec.Icerr

    #set a maximum alpha range, so when users click in the slider
    #it at least does something reasonable, rather than either nothing
    #significant, or so huge it becomes difficult to find the right value
    if args.max_alpha is None:
        if alpha == 0.0:
            max_alpha = 2.0
        else:
            max_alpha = 2*alpha

    def store_parameters_as_string(event=None):
        param_str = ("Parameter Values:\n"
        "Dmax  = {dmax:.5e}\n"
        "alpha = {alpha:.5e}\n"
        "I(0)  = {I0:.5e} +- {I0err:.5e}\n"
        "Rg    = {rg:.5e} +- {rgerr:.5e}\n"
        "r_avg = {r:.5e} +- {rerr:.5e}\n"
        "Vp    = {Vp:.5e} +- {Vperr:.5e}\n"
        "MW_Vp = {mwVp:.5e} +- {mwVperr:.5e}\n"
        "Vc    = {Vc:.5e} +- {Vcerr:.5e}\n"
        "MW_Vc = {mwVc:.5e} +- {mwVcerr:.5e}\n"
        "Lc    = {lc:.5e} +- {lcerr:.5e}\n"
        ).format(dmax=sasrec.D,alpha=sasrec.alpha,
            I0=sasrec.I0,I0err=sasrec.I0err,
            rg=sasrec.rg,rgerr=sasrec.rgerr,
            r=sasrec.avgr,rerr=sasrec.avgrerr,
            Vp=sasrec.Vp,Vperr=sasrec.Vperr,
            mwVp=sasrec.mwVp,mwVperr=sasrec.mwVperr,
            Vc=sasrec.Vc,Vcerr=sasrec.Vcerr,
            mwVc=sasrec.mwVc,mwVcerr=sasrec.mwVcerr,
            lc=sasrec.lc,lcerr=sasrec.lcerr)
        return param_str

    def print_values(event=None):
        print("---------------------------------")
        param_str = store_parameters_as_string()
        print(param_str)

    def save_file(event=None):
        #sascif = saxs.Sascif(sasrec)
        #sascif.write(output+".sascif")
        #print "%s file saved" % (output+".sascif")
        param_str = store_parameters_as_string()
        #add column headers to param_str for output
        param_str += 'q, I, error, fit'
        #quick, interpolate the raw data, sasrec.I, to the new qc values, but be sure to 
        #put zeros in for the q values not measured
        Iinterp = np.interp(sasrec.qc, sasrec.q_data, sasrec.I_data, left=0.0, right=0.0)
        np.savetxt(output+'.fit', np.vstack((sasrec.qc, Iinterp, sasrec.Icerr, sasrec.Ic)).T,delimiter=' ',fmt='%.5e',header=param_str)
        np.savetxt(output+'_pr.dat', np.vstack((sasrec.r, sasrec.P, sasrec.Perr)).T,delimiter=' ',fmt='%.5e')
        print("%s and %s files saved" % (output+".fit",output+"_pr.dat"))

    if args.plot:

        #fig, (axI, axP) = plt.subplots(1, 2, figsize=(12,6))
        #fig = plt.figure(0, figsize=(12,6))
        # fig.canvas.set_window_title(output)   # commented out due to AttributeError: 'FigureCanvasTkAgg' object has no attribute 'set_window_title'
        fig.suptitle(output)
        axI = plt.subplot2grid((3,2), (0,0),rowspan=2)
        axR = plt.subplot2grid((3,2), (2,0),sharex=axI)
        axP = plt.subplot2grid((3,2), (0,1),rowspan=3)
        plt.subplots_adjust(left=0.068, bottom=0.25, right=0.98, top=0.95)

        #add a plot of untouched light gray data for reference for the user
        I_l0, = axI.plot(Iq_orig[:,0], Iq_orig[:,1], '.', c='0.8', ms=3)
        I_l1, = axI.plot(sasrec.q_data, sasrec.I_data, 'k.', ms=3, label='test')
        I_l2, = axI.plot(sasrec.qc, sasrec.Ic, 'r-', lw=2)
        I_l3, = axI.plot(sasrec.qn, sasrec.In, 'bo', mec='b', mfc='none', mew=2)
        if args.log: axI.semilogy()
        axI.set_ylabel('I(q)')
        axI.set_xlabel('q')

        #residuals
        #first, just ensure that we're comparing similar q ranges, so
        #interpolate from qc to q_data to enable subtraction, since there's
        #q values in qc that are not in q_data, and possibly vice versa
        Icinterp = np.interp(sasrec.q_data, sasrec.qc, np.abs(sasrec.Ic))
        res = sasrec.I_data/sasrec.Ierr_data - Icinterp/sasrec.Ierr_data
        #in case qc were fewer points than the data, for whatever reason,
        #only grab the points up to qc.max
        ridx = np.where((sasrec.q_data<sasrec.qc.max()))
        absolute_maximum_q = np.max([sasrec.qc.max(),sasrec.q.max(),Iq_orig[:,0].max()])
        Ires_l0, = axR.plot([0,absolute_maximum_q], [0,0], 'k--')
        Ires_l1, = axR.plot(sasrec.q_data[ridx], res[ridx], 'r.', ms=3)
        axR.set_ylabel('Residuals')
        axR.set_xlabel('q')

        P_l1, = axP.plot(sasrec.r*100, sasrec.r*0, 'k--')
        P_l2, = axP.plot(sasrec.r, sasrec.P, 'b-', lw=2)
        axP.set_ylabel('P(r)')
        axP.set_xlabel('r')

        #axI.set_xlim([0,1.1*np.max(sasrec.q)])
        #axR.set_xlim([0,1.1*np.max(sasrec.q)])
        axI.set_xlim([0,Iq_orig[-1,0]])
        axR.set_xlim([0,Iq_orig[-1,0]])
        axP.set_xlim([0,1.1*np.max(sasrec.r)])
        axI.set_ylim([0.25*np.min(sasrec.Ic[sasrec.qc<Iq_orig[-1,0]]),2*np.max(sasrec.Ic[sasrec.qc<Iq_orig[-1,0]])])
        #axR.set_ylim([0,Iq_orig[-1,0]])
        #axP.set_ylim([0,1.1*np.max(sasrec.r)])
        #the "q" axis label is a little low, so let's raise it up a bit
        axR.xaxis.labelpad = -10

        axcolor = 'lightgoldenrodyellow'
        #axn1n2 = plt.axes([0.05, 0.175, 0.4, 0.03], facecolor=axcolor)
        axdmax = plt.axes([0.05, 0.125, 0.4, 0.03], facecolor=axcolor)
        axalpha = plt.axes([0.05, 0.075, 0.4, 0.03], facecolor=axcolor)
        #axnes = plt.axes([0.05, 0.025, 0.4, 0.03], facecolor=axcolor)

        axI0 = plt.figtext(.57, .125,   "$I(0)$  = %.2e $\pm$ %.2e"%(sasrec.I0,sasrec.I0err),family='monospace')
        axrg = plt.figtext(.57, .075,   "$R_g$   = %.2e $\pm$ %.2e"%(sasrec.rg,sasrec.rgerr),family='monospace')
        axrav = plt.figtext(.57, .025,  "$\overline{r}$    = %.2e $\pm$ %.2e"%(sasrec.avgr,sasrec.avgrerr),family='monospace')
        axVp = plt.figtext(.77, .125,   "$V_p$ = %.2e $\pm$ %.2e"%(sasrec.Vp,sasrec.Vperr),family='monospace')
        axVc = plt.figtext(.77, .075,   "$V_c$ = %.2e $\pm$ %.2e"%(sasrec.Vc,sasrec.Vcerr),family='monospace')
        axlc = plt.figtext(.77, .025,   "$\ell_c$ = %.2e $\pm$ %.2e"%(sasrec.lc,sasrec.lcerr),family='monospace')
        #axVpmw = plt.figtext(.55, .075, "Vp MW = %.2e $\pm$ %.2e"%(sasrec.mwVp,sasrec.mwVperr),family='monospace')
        #axVcmw = plt.figtext(.55, .025, "Vc MW = %.2e $\pm$ %.2e"%(sasrec.mwVc,sasrec.mwVcerr),family='monospace')

        #RangeSlider is for very new versions of matplotlib, so for now ignore it
        #sn1n2 = RangeSlider(axn1n2, 'n1n2', 0, Iq_orig.shape[0], valinit=(n1, n2))
        #sn1n2.valtext.set_visible(False)

        sdmax = Slider(axdmax, 'Dmax', 0.0, args.max_dmax, valinit=D)
        sdmax.valtext.set_visible(False)
        # set up ticks marks on the slider to denote the change in interaction
        axdmax.set_xticks([0.9 * sdmax.valmax, 0.1 * sdmax.valmax]) 
        #axdmax.xaxis.tick_top()
        axdmax.tick_params(labelbottom=False)

        salpha = Slider(axalpha, r'$\alpha$', 0.0, max_alpha, valinit=alpha)
        salpha.valtext.set_visible(False)

        dmax = D
        n1 = str(n1)
        n2 = str(n2)

        def analyze(dmax,alpha,n1,n2,extrapolate):
            global sasrec
            sasrec = saxs.Sasrec(Iq_orig[n1:n2], dmax, qc=qc, r=r, nr=args.nr, alpha=alpha, ne=nes, extrapolate=extrapolate)
            Icinterp = np.interp(sasrec.q_data, sasrec.qc, np.abs(sasrec.Ic))
            res = sasrec.I_data/sasrec.Ierr_data - Icinterp/sasrec.Ierr_data
            ridx = np.where((sasrec.q_data<sasrec.qc.max()))
            I_l1.set_data(sasrec.q_data, sasrec.I_data)
            I_l2.set_data(sasrec.qc, sasrec.Ic)
            I_l3.set_data(sasrec.qn, sasrec.In)
            Ires_l1.set_data(sasrec.q_data[ridx], res[ridx])
            P_l2.set_data(sasrec.r, sasrec.P)
            axI0.set_text("$I(0)$  = %.2e $\pm$ %.2e"%(sasrec.I0,sasrec.I0err))
            axrg.set_text("$R_g$   = %.2e $\pm$ %.2e"%(sasrec.rg,sasrec.rgerr))
            axrav.set_text("$\overline{r}$    = %.2e $\pm$ %.2e"%(sasrec.avgr,sasrec.avgrerr))
            axVp.set_text("$V_p$ = %.2e $\pm$ %.2e"%(sasrec.Vp,sasrec.Vperr))
            axVc.set_text("$V_c$ = %.2e $\pm$ %.2e"%(sasrec.Vc,sasrec.Vcerr))
            axlc.set_text("$\ell_c$ = %.2e $\pm$ %.2e"%(sasrec.lc,sasrec.lcerr))
            #axVpmw.set_text("Vp MW = %.2e $\pm$ %.2e"%(sasrec.mwVp,sasrec.mwVperr))
            #axVcmw.set_text("Vc MW = %.2e $\pm$ %.2e"%(sasrec.mwVc,sasrec.mwVcerr))

        def n1_submit(text):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(text)
            n2 = int(n2_box.text)
            extrapolate = extrapolate_check.get_status()[0]
            analyze(dmax,alpha,n1,n2,extrapolate)
            fig.canvas.draw_idle()

        def n2_submit(text):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(text)
            extrapolate = extrapolate_check.get_status()[0]
            analyze(dmax,alpha,n1,n2,extrapolate)
            fig.canvas.draw_idle()

        def extrapolate_submit(text):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            extrapolate = extrapolate_check.get_status()[0]
            analyze(dmax,alpha,n1,n2,extrapolate)
            fig.canvas.draw_idle()

        def D_submit(text):
            dmax = float(text)
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            extrapolate = extrapolate_check.get_status()[0]
            analyze(dmax,alpha,n1,n2,extrapolate)
            # this updates the slider value based on text box value
            sdmax.set_val(dmax)
            if (dmax > 0.9 * sdmax.valmax) or (dmax < 0.1 * sdmax.valmax):
                sdmax.valmax = 2 * dmax
                sdmax.ax.set_xlim(sdmax.valmin, sdmax.valmax)
                axdmax.set_xticks([0.9 * sdmax.valmax, 0.1 * sdmax.valmax])
            fig.canvas.draw_idle()

        def A_submit(text):
            dmax = sdmax.val
            alpha = float(text)
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            extrapolate = extrapolate_check.get_status()[0]
            analyze(dmax,alpha,n1,n2,extrapolate)
            # this updates the slider value based on text box value
            salpha.set_val(alpha)
            # partions alpha slider
            if (alpha > 0.9 * salpha.valmax) or (alpha < 0.1 * salpha.valmax):
                salpha.valmax = 2 * alpha
                # alpha starting at zero makes initial adjustment additive not multiplicative
                if alpha != 0:
                    salpha.ax.set_xlim(salpha.valmin, salpha.valmax)
                elif alpha == 0:
                    salpha.valmax = alpha + 10
                    salpha.valmin = 0.0
                    salpha.ax.set_xlim(salpha.valmin, salpha.valmax)
            fig.canvas.draw_idle()

        def update(val):
            dmax = sdmax.val
            alpha = salpha.val
            n1 = int(n1_box.text)
            n2 = int(n2_box.text)
            extrapolate = extrapolate_check.get_status()[0]
            #print(extrapolate)
            analyze(dmax,alpha,n1,n2,extrapolate)
            # partitions the slider, so clicking in the upper and lower range scale valmax
            if (dmax > 0.9 * sdmax.valmax) or (dmax < 0.1 * sdmax.valmax):
                sdmax.valmax = 2 * dmax
                sdmax.ax.set_xlim(sdmax.valmin, sdmax.valmax)
                axdmax.set_xticks([0.9 * sdmax.valmax, 0.1 * sdmax.valmax])
            # partions slider as well
            if (alpha > 0.9 * salpha.valmax) or (alpha < 0.1 * salpha.valmax):
                salpha.valmax = 2 * alpha
                # alpha starting at zero makes initial adjustment additive not multiplicative
                if alpha != 0:
                    salpha.ax.set_xlim(salpha.valmin, salpha.valmax)
                elif alpha == 0:
                    salpha.valmax = alpha + 10
                    salpha.valmin = 0.0
                    salpha.ax.set_xlim(salpha.valmin, salpha.valmax)

            Dmax_box.set_val("%.4e"%dmax)
            Alpha_box.set_val("%.4e"%alpha)

            fig.canvas.draw_idle()

        # making a text entry for dmax that allows for user input
        Dvalue = "{dmax:.4e}".format(dmax=dmax)
        axIntDmax = plt.axes([0.45, 0.125, 0.08, 0.03])
        Dmax_box = TextBox(axIntDmax, '', initial=Dvalue)
        Dmax_box.on_submit(D_submit)

        # making a text entry for alpha that allows for user input
        Avalue = "{alpha:.4e}".format(alpha=alpha)
        axIntAlpha = plt.axes([0.45, 0.075, 0.08, 0.03])
        Alpha_box = TextBox(axIntAlpha, '', initial=Avalue)
        Alpha_box.on_submit(A_submit)

        # making a text entry for n1 that allows for user input
        n1value = "{}".format(n1)
        plt.figtext(0.0085, 0.178, "First point")
        axIntn1 = plt.axes([0.075, 0.170, 0.08, 0.03])
        n1_box = TextBox(axIntn1, '', initial=n1)
        n1_box.on_submit(n1_submit)

        # making a text entry for n2 that allows for user input
        n2value = "{}".format(n2)
        plt.figtext(0.17, 0.178, "Last point")
        axIntn2 = plt.axes([0.235, 0.170, 0.08, 0.03])
        n2_box = TextBox(axIntn2, '', initial=n2)
        n2_box.on_submit(n2_submit)

        # create a checkbox for extrapolation
        axExtrap = plt.axes([0.35, 0.170, 0.015, 0.03], frameon=True)
        axExtrap.margins(0.0)
        extrapolate_check = CheckButtons(axExtrap, ["Extrapolate"], [args.extrapolate])
        #the axes object for the checkbutton is crazy large, and actually
        #blocks the sliders underneath even when frameon=False
        #so we have to manually set the size and location of each of the
        #elements of the checkbox after setting the axes margins to zero above
        #including the rectangle checkbox, the lines for the X, and the label
        check = extrapolate_check
        size =  1.0 #size relative to axes axExtrap
        for rect in extrapolate_check.rectangles:
            rect.set_x(0.)
            rect.set_y(0.)
            rect.set_width(size)
            rect.set_height(size)
        first = True
        for l in check.lines:
            for ll in l:
                llx = ll.get_xdata()
                lly = ll.get_ydata()
                #print(llx)
                #print(lly)
                ll.set_xdata([0.0,size])
                if first:
                    #there's two lines making
                    #up the checkbox, so need
                    #to set the y values separately
                    #one going from bottom left to 
                    #upper right, the other opposite
                    ll.set_ydata([size,0.0])
                    first = False
                else:
                    ll.set_ydata([0.0, size])
        check.labels[0].set_position((1.5,.5))

        #here is the slider updating
        sdmax.on_changed(update)
        salpha.on_changed(update)
        extrapolate_check.on_clicked(update)
        #snes.on_changed(update)

        axreset = plt.axes([0.05, 0.02, 0.1, 0.04])
        reset_button = Button(axreset, 'Reset Sliders', color=axcolor, hovercolor='0.975')

        def reset_values(event):
            sdmax.reset()
            salpha.reset()
        reset_button.on_clicked(reset_values)

        axprint = plt.axes([0.2, 0.02, 0.1, 0.04])
        print_button = Button(axprint, 'Print Values', color=axcolor, hovercolor='0.975')

        print_button.on_clicked(print_values)

        axsave = plt.axes([0.35, 0.02, 0.1, 0.04])
        save_button = Button(axsave, 'Save File', color=axcolor, hovercolor='0.975')

        save_button.on_clicked(save_file)

        # plt.show()

    return reset_button, print_button, save_button
