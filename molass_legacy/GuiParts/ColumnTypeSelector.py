"""
    GuiParts.ColumnTypeSelector.py

    Copyright (c) 2024-2025, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk, ttk
from molass_legacy.Experiment.ColumnTypes import get_productnames, get_columntype_from_name
from molass_legacy.Experiment.DataUtils import get_columntype

class ColumnTypeSelector(Tk.Frame):
    def __init__(self, parent, column_name_var=None, excl_limit_var=None):
        Tk.Frame.__init__(self, parent)
        columntype = get_columntype()
        column_name = columntype.name
        if column_name_var is None:
            column_name_var = Tk.StringVar()
        self.column_name = column_name_var
        self.column_name.set(column_name)
        if excl_limit_var is None:
            excl_limit_var = Tk.DoubleVar()
        self.excl_limit = excl_limit_var
        self.excl_limit.set(columntype.excl_limit)
        self.column_products = ttk.Combobox(self, textvariable=self.column_name,
                                            width=35, justify=Tk.CENTER, state="readonly")
        self.column_products.pack(side=Tk.LEFT)
        self.column_products['values'] = get_productnames()
        self.column_name.trace("w", self.column_name_tracer)

    def config(self, **kwargs):
        self.column_products.config(**kwargs)

    def column_name_tracer(self, *args):
        columntype = get_columntype_from_name(self.column_name.get())
        self.excl_limit.set(columntype.excl_limit)