"""
    ModelParams.ParamsSheetBase.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk

"""
    task: let this class have more in common
"""
class ParamsSheetBase(Tk.Frame):
    def __init__(self, parent, params, dsets, optimizer):
        self.parent = parent
        self.active_indeces = None
        self.num_params = len(params)
        Tk.Frame.__init__(self, parent)

    def enable_selection(self):
        self.sheet.enable_bindings(bindings="toggle_select")

    def enable_copy(self):
        self.sheet.enable_bindings()

    def set_selection(self, sel_list):
        for k in sel_list:
            r, c = self.get_cell_address(k)
            self.sheet.toggle_select_cell(r, c)

    def set_params_addr(self, k, addr):
        self.params_addr[k] = addr
        self.params_addr_inv[addr] = k

    def get_cell_address(self, k):
        return self.params_addr[k]

    def get_selection(self):
        ret_list = []
        for c in self.sheet.get_selected_cells():
            k = self.params_addr_inv.get(c)
            if k is None:
                import molass_legacy.KekLib.CustomMessageBox as MessageBox
                MessageBox.showerror("Invalid Cell", "Invalid Cell", parent=self.parent)
                raise ValueError
            else:
                ret_list.append(k)

        if len(ret_list) != 2:
            import molass_legacy.KekLib.CustomMessageBox as MessageBox
            MessageBox.showerror("Number of Cells", "You must select exactly two cells.", parent=self.parent)
            raise ValueError

        return sorted(ret_list)

    def save_as(self, file):
        with open(file, "w") as fh:
            for k, row in enumerate(self.data_list, start=1):
                fh.write(",".join(row) + "\n")
                if k >= self.num_valid_rows:
                    break

    def get_active_indeces(self, state_info):
        # print("get_active_indeces: params_addr=", self.params_addr)
        if self.active_indeces is None:
            indeces = state_info.get_active_indeces()
            if len(indeces) == 0:
                indeces = list(range(self.num_params))
            self.active_indeces = indeces
        return self.active_indeces