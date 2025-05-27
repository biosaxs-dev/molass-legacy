# coding: utf-8
"""
    QmmOptions.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import re
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

NUM_SEEDS = 4

class QmmOptionsDialog(Dialog):
    def __init__(self, parent):
        Dialog.__init__(self, parent, title="QMM Options", visible=False, location='lower right')

    def show(self):
        self._show()

    def body(self, bframe):
        iframe = Tk.Frame(bframe)
        iframe.pack(padx=20, pady=10)

        grid_row = 0
        label = Tk.Label(iframe, text="Number of components")
        label.grid(row=grid_row, column=0, sticky=Tk.W, pady=5)

        self.auto_n_components = Tk.IntVar()
        self.auto_n_components.set(get_setting('auto_n_components'))
        auto_cb = Tk.Checkbutton(iframe, text="automatic", variable=self.auto_n_components)
        auto_cb.grid(row=grid_row, column=1, sticky=Tk.W, padx=5 )

        grid_row += 1
        self.n_components = Tk.IntVar()
        self.n_components.set(get_setting('n_components'))

        self.n_components_spinbox = Tk.Spinbox( iframe, textvariable=self.n_components,
                                from_=1, to=16, increment=1,
                                justify=Tk.CENTER, width=6)

        self.n_components_spinbox.grid(row=grid_row, column=1, sticky=Tk.W, padx=5)

        self.auto_n_components_tracer()
        self.auto_n_components.trace("w", self.auto_n_components_tracer)

        grid_row += 1
        space = Tk.Frame(iframe, height=5)
        space.grid(row=grid_row, column=0)

        grid_row += 1
        label = Tk.Label(iframe, text="Number of iterations")
        label.grid(row=grid_row, column=0, sticky=Tk.W, pady=5)

        self.max_iterations = Tk.IntVar()
        self.max_iterations.set(get_setting('max_iterations'))
        spinbox = Tk.Spinbox( iframe, textvariable=self.max_iterations,
                                from_=10, to=1000, increment=100,
                                justify=Tk.CENTER, width=6)

        spinbox.grid(row=grid_row, column=1, sticky=Tk.W, padx=5, pady=5)

        self.conc_dependence = Tk.IntVar()
        cd = get_setting('conc_dependence')
        self.conc_dependence.set(cd)

        grid_row += 1
        label = Tk.Label(iframe, text="Conc. dependence")
        label.grid(row=grid_row, column=0, sticky=Tk.W, pady=5)

        spinbox = Tk.Spinbox( iframe, textvariable=self.conc_dependence,
                                from_=1, to=2, increment=1,
                                justify=Tk.CENTER, width=6)

        spinbox.grid(row=grid_row, column=1, sticky=Tk.W, padx=5)

        grid_row += 1
        label = Tk.Label(iframe, text="QMM fitting mode")
        label.grid(row=grid_row, column=0, sticky=Tk.W, pady=5)

        self.qmm_separately = Tk.IntVar()
        self.qmm_separately.set(get_setting('qmm_separately'))
        mode_cb = Tk.Checkbutton(iframe, text="separately", variable=self.qmm_separately)
        mode_cb.grid(row=grid_row, column=1, sticky=Tk.W, padx=5)

        grid_row += 1
        label = Tk.Label(iframe, text="Random seeds")
        label.grid(row=grid_row, column=0, sticky=Tk.W, pady=5)

        self.random_seed_fix = Tk.IntVar()
        fixed_random_seeds = get_setting('fixed_random_seeds')
        self.random_seed_fix.set(0 if fixed_random_seeds is None else 1)
        seeds_cb = Tk.Checkbutton(iframe, text="fixed", variable=self.random_seed_fix)
        seeds_cb.grid(row=grid_row, column=1, sticky=Tk.W, padx=5 )

        grid_row += 1
        random_seed_frame = Tk.Frame(iframe)
        random_seed_frame.grid(row=grid_row, column=1, sticky=Tk.W, padx=5)
        self.random_seeds = Tk.StringVar()
        if fixed_random_seeds is None:
            fixed_random_seeds = get_setting('last_random_seeds')
            if fixed_random_seeds is None:
                import numpy as np
                fixed_random_seeds = np.random.randint(1000, 9999, NUM_SEEDS)

        self.random_seeds.set( ' '.join(['%4d' % seed for seed in fixed_random_seeds]) )
        self.seed_entry = Tk.Entry(random_seed_frame, textvariable=self.random_seeds, width=19, justify=Tk.CENTER)
        self.seed_entry.pack()

        self.random_seed_fix.trace("w", self.random_seed_fix_tracer)
        self.random_seed_fix_tracer()

        grid_row += 1
        label = Tk.Label(iframe, text="Denoise rank")
        label.grid(row=grid_row, column=0, sticky=Tk.W, pady=5)

        forced_denoise_rank = get_setting('forced_denoise_rank')
        self.auto_denoise_rank = Tk.IntVar()
        self.auto_denoise_rank.set(1 if forced_denoise_rank is None else 0)
        rank_cb = Tk.Checkbutton(iframe, text="automatic", variable=self.auto_denoise_rank)
        rank_cb.grid(row=grid_row, column=1, sticky=Tk.W, padx=5 )

        grid_row += 1
        if forced_denoise_rank is None:
            denoise_rank = get_setting('last_denoise_rank')
        else:
            denoise_rank = forced_denoise_rank
        self.denoise_rank = Tk.IntVar()
        self.denoise_rank.set(denoise_rank)

        spinbox = Tk.Spinbox( iframe, textvariable=self.denoise_rank,
                                from_=1, to=32, increment=1,
                                justify=Tk.CENTER, width=6)

        spinbox.grid(row=grid_row, column=1, sticky=Tk.W, padx=5)
        self.denoise_rank_sbox = spinbox

        self.auto_denoise_rank.trace("w", self.auto_denoise_rank_tracer)
        self.auto_denoise_rank_tracer()

        grid_row += 1
        label = Tk.Label(iframe, text="Rg estimation")
        label.grid(row=grid_row, column=0, sticky=Tk.W, pady=5)

        denss_fitted_rg = get_setting('denss_fitted_rg')
        self.denss_fitted_rg = Tk.IntVar()
        self.denss_fitted_rg.set(denss_fitted_rg)
        fitted_rg_cb = Tk.Checkbutton(iframe, text="apply fit_data (DENSS)", variable=self.denss_fitted_rg)
        fitted_rg_cb.grid(row=grid_row, column=1, sticky=Tk.W, padx=5 )

    def auto_n_components_tracer(self, *args):
        auto = self.auto_n_components.get()
        self.n_components_spinbox.config(state=Tk.DISABLED if auto else Tk.NORMAL)

    def random_seed_fix_tracer(self, *args):
        fix = self.random_seed_fix.get()
        state = Tk.NORMAL if fix else Tk.DISABLED
        self.seed_entry.config(state=state)

    def auto_denoise_rank_tracer(self, *args):
        auto = self.auto_denoise_rank.get()
        self.denoise_rank_sbox.config(state=Tk.DISABLED if auto else Tk.NORMAL)

    def validate(self):
        try:
            self.input_random_seeds = [int(seed) for seed in re.split(r'\s+', self.random_seeds.get())]
            self.seed_entry.config(fg='black')
            ret = True
        except:
            self.seed_entry.config(fg='red')
            ret = False
        return ret

    def apply(self):
        set_setting('auto_n_components', self.auto_n_components.get())
        set_setting('n_components', self.n_components.get())
        set_setting('max_iterations', self.max_iterations.get())
        set_setting('conc_dependence', self.conc_dependence.get())
        set_setting('qmm_separately', self.qmm_separately.get())
        if self.random_seed_fix.get():
            fixed_random_seeds = self.input_random_seeds
        else:
            fixed_random_seeds = None
        set_setting('fixed_random_seeds', fixed_random_seeds)

        auto_denoise_rank = self.auto_denoise_rank.get()
        if auto_denoise_rank:
            denoise_rank = None
        else:
            denoise_rank = self.denoise_rank.get()
        set_setting('forced_denoise_rank', denoise_rank)
        set_setting('denss_fitted_rg', self.denss_fitted_rg.get())
