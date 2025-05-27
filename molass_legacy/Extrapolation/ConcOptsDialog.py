# coding: utf-8
"""
    ConcOptsDialog.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.TkUtils import rational_geometry
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable, BlinkingFrame
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

class ConcOptsDialog( Dialog ):
    def __init__(self, parent, dialog, pranges):
        # print('pranges=', pranges)
        self.dialog = dialog
        self.pranges = pranges
        self.sd = dialog.sd
        self.xr_j0 = dialog.sd.xr_j0
        self.range_id_list = []
        self.range_id_dict = {}
        for k, r in enumerate(pranges):
            ft_list = r.get_fromto_list()
            print([k], ft_list)
            for m, ft in enumerate(ft_list):
                ft_ = [ self.xr_j0 + j for j in ft ]
                rande_id = '%d-%d (%s)' % (k+1, m+1, str(ft_))
                self.range_id_dict[rande_id] = ft_
                self.range_id_list.append(rande_id)
        Dialog.__init__( self, parent, "Concentration Options", visible=False, geometry_cb=self.adjust_geometry )

    def adjust_geometry(self):
        rational_geometry(self, self.dialog, 0.5, 0.5)

    def show(self):
        self._show()

    def body( self, body_frame ):
        tk_set_icon_portable( self )

        iframe = Tk.Frame(body_frame)
        iframe.pack(padx=20, pady=10)

        grid_row = 0

        label = Tk.Label(iframe, text="Scattering Curve Inspection View: ")
        label.grid(row=grid_row, column=0, sticky=Tk.W)

        label = Tk.Label(iframe, text="Range ID Selection")
        label.grid(row=grid_row, column=1)

        self.range_id = Tk.StringVar()
        self.range_id.set(self.range_id_list[0])
        range_id_box = ttk.Combobox( iframe, textvariable=self.range_id, width=16, justify=Tk.CENTER )
        range_id_box[ 'values' ] = self.range_id_list
        range_id_box.grid(row=grid_row, column=2)

        grid_row += 1
        plot_button = Tk.Button(iframe, text="Show", command=self.plot_normalized_curves)
        plot_button.grid(row=grid_row, column=1, columnspan=2, pady=10)

        grid_row += 1
        space = Tk.Label(iframe)
        space.grid(row=grid_row, column=0)

        grid_row += 1
        label = Tk.Label(iframe, text="Concentration Type: ")
        label.grid(row=grid_row, column=0, sticky=Tk.W)

        self.conc_curve_type = Tk.IntVar()
        self.conc_curve_type.set(get_setting('conc_curve_type'))

        for k, cname in enumerate([ 'Synchronized UV Elution Curve', 'Scaled Xray Elution Curve', ]):
            rb = Tk.Radiobutton( iframe, text=cname,
                        variable=self.conc_curve_type, value=k,
                        )
            rb.grid( row=grid_row+k, column=1, sticky=Tk.W )

        grid_row += 2
        space = Tk.Label(iframe)
        space.grid(row=grid_row, column=0)

        grid_row += 1
        label = Tk.Label(iframe, text="Weight Matrix Type: ")
        label.grid(row=grid_row, column=0, sticky=Tk.W)

        self.weight_matrix_type = Tk.IntVar()
        self.weight_matrix_type.set(get_setting('weight_matrix_type'))

        for k, cname in enumerate([ 'Uniform', 'Concentration Reciprocal', ]):
            rb = Tk.Radiobutton( iframe, text=cname,
                        variable=self.weight_matrix_type, value=k,
                        )
            rb.grid( row=grid_row+k, column=1, sticky=Tk.W )

    def apply(self):
        set_setting('conc_curve_type', self.conc_curve_type.get())
        set_setting('weight_matrix_type', self.weight_matrix_type.get())

    def plot_normalized_curves(self):
        from ScatteringCurveViewer import ScatteringCurveViewer

        editor = self.dialog.editor_frame
        range_id = self.range_id.get()
        ft = self.range_id_dict[range_id]
        scaled_y = self.dialog.mapper.make_scaled_xray_curve_y()
        viewer = ScatteringCurveViewer(editor, editor, self.sd, ft, scaled_y)
        viewer.show()
