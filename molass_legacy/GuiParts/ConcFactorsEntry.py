"""

    ConcFactorsEntry.py

    Copyright (c) 2020-2023, SAXS Team, KEK-PF

"""
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting, get_beamline_name
from molass_legacy.KekLib.OurTkinter import Tk

class ConcFactorsEntry:
    def __init__(self, frame, grid_row):
        self.path_length = Tk.StringVar()
        self.extinction = Tk.StringVar()
        self.beamline_name = Tk.StringVar()
        self.reset_entries()

        self.widgets = []

        # UV Device No
        device_label = Tk.Label(frame, text='Beamline: ')
        device_label.grid(row=grid_row, column=0, sticky=Tk.E)
        self.widgets.append(device_label)

        device_no_label = Tk.Entry(frame, textvariable=self.beamline_name, width=20)
        device_no_label.grid(row=grid_row, column=1, sticky=Tk.W )
        self.widgets.append(device_no_label)

        # Conc. Factors
        row = grid_row+1
        conc_label = Tk.Label( frame, text='Conc. Factors: ' )
        conc_label.grid( row=row, column=0, sticky=Tk.E )
        self.widgets.append(conc_label)

        conc_factor_frame = Tk.Frame( frame )
        conc_factor_frame.grid( row=row, column=1, columnspan=2, sticky=Tk.W )

        conc_factor_label1 = Tk.Label( conc_factor_frame, text='Absorbance(λ=%g) × Path Length Factor(' % ( get_setting( 'absorbance_picking' ) ) )
        conc_factor_label1.grid( row=0, column=0 )
        self.widgets.append(conc_factor_label1)

        path_length_entry = Tk.Entry( conc_factor_frame, textvariable=self.path_length, width=6, justify=Tk.CENTER )
        path_length_entry.grid( row=0, column=1 )
        self.widgets.append(path_length_entry)

        conc_factor_label2 = Tk.Label( conc_factor_frame, text=') ÷ Extinction Coefficient(' )
        conc_factor_label2.grid( row=0, column=2 )
        self.widgets.append(conc_factor_label2)

        extinction_entry = Tk.Entry( conc_factor_frame, textvariable=self.extinction, width=7, justify=Tk.CENTER )
        extinction_entry.grid( row=0, column=3 )
        self.widgets.append(extinction_entry)

        conc_factor_label3 = Tk.Label( conc_factor_frame, text=')' )
        conc_factor_label3.grid( row=0, column=4 )
        self.widgets.append(conc_factor_label3)

    def config(self, **kwargs):
        for w in self.widgets:
            w.config(**kwargs)

    def compute_conc_factor(self):
        try:
            path_length = float(self.path_length.get())
            extinction = float(self.extinction.get())
            conc_factor = path_length/extinction
        except:
            raise ValueError

        set_setting('path_length', path_length)
        set_setting('extinction', extinction)
        set_setting('conc_factor', conc_factor)
        return conc_factor

    def update_conc_factors(self, sd, mapper):
        conc_factor = self.compute_conc_factor()
        sd.set_mc_vector(mapper, conc_factor)

    def update_path_length(self):
        path_length = get_setting( 'path_length' )
        self.path_length.set( '%.3f' % path_length )
        self.beamline_name.set(get_beamline_name())

    def reset_entries(self):
        path_length = get_setting( 'path_length' )
        self.path_length.set( '%.3f' % path_length )
        extinction = get_setting( 'extinction' )
        self.extinction.set( '%.3f' % extinction )
        self.beamline_name.set(get_beamline_name())

    def apply_entries(self):
        set_setting('beamline_name', self.beamline_name.get())      # used in SummaryBook
