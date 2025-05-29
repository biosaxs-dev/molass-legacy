"""
    Selective.ModelSelectFrame.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""

from molass_legacy.KekLib.OurTkinter import Tk, ttk
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy._MOLASS.Version import is_developing_version
from molass_legacy.GuiParts.ColumnTypeSelector import ColumnTypeSelector

enable_edm_model = get_setting('enable_edm_model')
if enable_edm_model:
    # MODEL_NAMES = ['EMG', 'EGH', 'EDM', 'STC', 'SDM'] # SDM is not used in the current version
    MODEL_NAMES = ['EMG', 'EGH', 'EDM', 'STC']
else:
    MODEL_NAMES = ['EMG', 'EGH']

class ModelSelectFrame(Tk.LabelFrame):
    def __init__(self, parent, editor, **kwargs):
        Tk.LabelFrame.__init__(self, parent, **kwargs)

        self.editor = editor
        select_frame = Tk.Frame(self)
        select_frame.pack(padx=10, pady=20)

        sf_row = 0
        label = Tk.Label(select_frame, text="Elution Curve Model: ")
        label.grid(row=sf_row, column=0, sticky=Tk.E)
        self.model_name  = Tk.StringVar()
        self.model_name.set(MODEL_NAMES[editor.current])
        w = ttk.Combobox(select_frame, textvariable=self.model_name, values=MODEL_NAMES,
                         width=10, justify=Tk.CENTER, state="readonly")
        w.grid(row=sf_row, column=1, sticky=Tk.W, padx=5, pady=10)
        w.bind("<<ComboboxSelected>>", self.change_model)
        self.model_select_box = w

        sf_row += 1
        self.use_column_constraints = Tk.IntVar()
        self.use_column_constraints.set(0)
        self.ucc_check_button = Tk.Checkbutton(select_frame, text="Use Column Constraints", variable=self.use_column_constraints)
        self.ucc_check_button.grid(row=sf_row, column=0, columnspan=2, sticky=Tk.W, padx=5, pady=10)
        self.use_column_constraints.trace_add("write", self.use_column_constraints_tracer)

        sf_row += 1
        label = Tk.Label(select_frame, text="SEC Column Type: ")
        label.grid(row=sf_row, column=0, sticky=Tk.E)
        self.selector = ColumnTypeSelector(select_frame)
        self.selector.grid(row=sf_row, column=1, sticky=Tk.W, padx=5, pady=0)

        sf_row += 1
        button = Tk.Button(select_frame, text="PDS Simulation", command=self.show_psd_simulation)
        button.grid(row=sf_row, column=0, columnspan=2, pady=30)

        if is_developing_version():
            button = Tk.Button(select_frame, text="Devel Test", command=self.do_devel_test)
            button.grid(row=sf_row, column=2, columnspan=2, padx=20)

    def change_model(self, *args):
        self.editor.change_model(self.model_name.get())

    def get_model_name(self):
        return self.model_name.get()

    def use_column_constraints_tracer(self, *args):
        state = Tk.DISABLED if self.use_column_constraints.get() == 0 else Tk.NORMAL
        self.selector.config(state=state)

    def update_selection_state(self):
        if self.editor.get_current_model().is_traditional():
            state = Tk.DISABLED
            self.use_column_constraints.set(0)
        else:
            state = Tk.NORMAL        
        self.ucc_check_button.config(state=state)

    def show_psd_simulation(self, devel=True):
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        yn = MessageBox.askyesno("Simulation Start Confirmation",
            'It may take several minutes to prepare for the simulation.\n'
            'Are you sure to proceed?"\n'
            '(Although, it will not be blocked since it will start as a separate process)"\n'
            ,
            parent=self.editor.parent
            )

        if not yn:
            return

        if devel:
            from importlib import reload
            import Simulative.PsdSimulation
            reload(Simulative.PsdSimulation)
        from Simulative.PsdSimulation import pds_simulation_impl

        pds_simulation_impl(self.editor)

    def do_devel_test(self):
        from importlib import reload
        import Selective.DevelTest
        reload(Selective.DevelTest)
        from Selective.DevelTest import devel_test_impl
        devel_test_impl(self.editor)
