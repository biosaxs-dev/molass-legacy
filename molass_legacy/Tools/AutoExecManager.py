"""
    AutoExecManager.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
import sys
import os
import getopt
import re

# execute this file with python this-path demo
if len(sys.argv) > 1 and sys.argv[1].find("demo") >= 0:
    this_dir = os.path.dirname(os.path.abspath( __file__ ))
    sys.path.append(this_dir + '/..')

import psutil
from tksheet import Sheet
from tkinter import messagebox
from molass_legacy.KekLib.OurTkinter import Tk, Dialog

def shortened_list(L):
    ret_list = []
    for item in L:
        if item.find("\\") >= 0 or item.find("/") >= 0:
            nodes = re.split(r"[\\/]", item)
            if len(nodes) > 3:
                item = "/".join([nodes[0], "...",nodes[-1]])
        ret_list.append(item)
    return ret_list

CMDLINE_COL = 4

class AutoExecManager(Dialog):
    def __init__(self, parent):
        Dialog.__init__(self, parent, "AutoExecManager", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        column_width = 90
        cmdline_column_width = 1200
        colnames = ["pid", "ppid", "proc name", "func code", "cmdline"]
        num_columns = len(colnames)
        data_list = [colnames]
        pid_dict = {}
        ppid_dict = {}
        row = 1
        for proc in psutil.process_iter():
            if proc.name()[0:7] == "pythonw":
                pid = proc.pid
                ppid = proc.ppid()
                pid_dict[pid] = (row, pid, ppid, proc)
                ppid_dict[ppid] = (row, pid, ppid, proc)
                cmdline = proc.cmdline()
                func_code = self.get_func_code(cmdline)
                data_list.append((pid, ppid, proc.name(), func_code, shortened_list(cmdline)))
                row += 1

        num_rows = row

        height = int(22*num_rows) + 60
        width = column_width*num_columns + 60 + cmdline_column_width - column_width

        sheet = Sheet(body_frame, data=data_list, column_width=column_width, width=width, height=height)
        sheet.column_width(column=CMDLINE_COL, width=cmdline_column_width)
        sheet.enable_bindings("single_select", "row_select", "right_click_popup_menu")
        sheet.popup_menu_add_command("Kill", self.kill_paired_processes)
        sheet.pack()
        self.data_list = data_list
        self.sheet = sheet
        self.pid_dict = pid_dict
        self.ppid_dict = ppid_dict

    def get_func_code(self, cmdline):
        if str(cmdline).find("--") >= 0:
            return ""
        optlist, args = getopt.getopt(cmdline[2:], 'c:w:f:n:i:d:m:s:r:t:p:T:M:')
        optdict = dict(optlist)
        func_code = optdict.get("-c", "")
        return func_code

    def kill_paired_processes(self):
        selected = self.sheet.get_currently_selected()
        print("kill_paired_processes", selected)
        row = selected.row
        row_data = self.data_list[row]
        print(row_data)
        cmdline = row_data[CMDLINE_COL]
        if cmdline[1].find("molass.py") >= 0:
            row_type = "monitor"
            monitor_pid = row_data[0]
            try:
                row_, optimizer_pid = self.ppid_dict[monitor_pid][0:2]
                row_data_ = self.data_list[row_]
                cmdline_ = row_data_[CMDLINE_COL]
                func_code = self.get_func_code(cmdline_)
            except:
                optimizer_pid = None
                func_code = ""
            print(row_type, monitor_pid, optimizer_pid, func_code)
        else:
            row_type = "optimizer"
            monitor_pid = row_data[1]
            optimizer_pid = row_data[0]
            func_code = self.get_func_code(cmdline)
            print(row_type, monitor_pid, optimizer_pid, func_code)

        monitor_proc = self.pid_dict[monitor_pid][3]
        if optimizer_pid is None:
            optimizer_proc = None
            cwd = ""
            message =  ("%s row is selected.\n"
                        "are you sure to kill the following monitor processe?\n"
                        "    monitor: %d\n"
                        %  (row_type, monitor_pid) )
        else:
            optimizer_proc = self.pid_dict[optimizer_pid][3]
            cwd = optimizer_proc.cwd()
            message =  ("%s row is selected.\n"
                        "are you sure to kill the following paired processes?\n"
                        "    monitor: %d\n"
                        "    optimizer: %d\n"
                        "    func_code: %s\n"
                        "    cwd: %s"
                        %  (row_type, monitor_pid, optimizer_pid, func_code, cwd) )
        ok = messagebox.askokcancel(title="Kill confirmation", message=message)
        if ok:
            if optimizer_proc is None:
                monitor_proc.kill()
                messagebox.showinfo(title="Killed notification", message="the monitor process has been killed.")
            else:
                for proc in [monitor_proc, optimizer_proc]:
                    proc.kill()
                messagebox.showinfo(title="Killed notification", message="the paired processes have been killed.")

def run_manager():
    from molass_legacy.KekLib.TkUtils import get_tk_root
    root = get_tk_root()

    def show_manager():
        ts = AutoExecManager(root)
        ts.show()
        root.quit()

    root.after(0, show_manager)
    root.mainloop()
    root.destroy()

if __name__ == '__main__':
    run_manager()
