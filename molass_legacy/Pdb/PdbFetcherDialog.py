# coding: utf-8
"""
    PdbFetcherDialog.py

    Copyright (c) 2020-2021, SAXS Team, KEK-PF
"""
import os
import sys
import re
import logging
import queue
from molass_legacy.KekLib.KillableThread import Thread
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ScrolledText
from molass_legacy.KekLib.TkCustomWidgets import FolderEntry
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.BasicUtils import is_empty_dir, mkdirs_with_retry
from molass_legacy.DENSS.DenssUtils import run_pdb2mrc

count_re = re.compile(r'(\d+)$')

def get_new_folder(folder):
    while os.path.exists(folder) and is_empty_dir(folder):
        folder = re.sub(count_re,  lambda m: str(int(m.group(1))+1), folder)
    return folder

"""
    Unifying these logging procesures with those in DenssGui
    seems to be a challenging task,
    because simply moving them to a module file beaks the logging
    from the separate thread.
    See also DenssFitData.py
    for an example of logging in the same thread.
"""
class StdoutRedirector:
    def __init__(self, queue):
        self.saved_stdout = sys.stdout
        self.queue = queue
        sys.stdout = self

    def write(self, string):
        self.queue.put([1, string])

    def flush(self):
        pass

    def __del__(self):
        sys.stdout = self.saved_stdout
        print('__del__ ok')

class ProgressLoghandler(logging.Handler):
    def __init__(self, log_text, log_file):
        logging.Handler.__init__(self)
        self.log_text   = log_text
        self.fh = open(log_file, 'w')
        self.formatter  = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')

    def emit( self, record ):
        if record.levelno >= logging.INFO:
            try:
                log_message = self.formatter.format(record) + '\n'
                self.fh.write(log_message)
                self.fh.flush()
                self.log_text.insert(Tk.INSERT, log_message)
                self.log_text.see(Tk.END)
            except:
                # this error can occur when canceled
                pass

    def __del__(self):
        self.fh.close()

class PdbFetcherDialog(Dialog):
    def __init__(self, parent):
        self.applied = False
        self.mrc_file = None
        Dialog.__init__( self, parent, "PDB Fetcher", visible=False )

    def show(self):
        self._show()

    def body(self, body_frame):
        input_frame = Tk.Frame(body_frame)
        input_frame.pack(padx=20, pady=10)
        log_frame = Tk.Frame(body_frame)
        log_frame.pack(padx=20, pady=10)

        entry_width = 50
        grid_row = 0
        label = Tk.Label(input_frame, text="PDB ID: ")
        label.grid(row=grid_row, column=0, sticky=Tk.E)
        self.search_key = Tk.StringVar()
        entry = Tk.Entry(input_frame, textvariable=self.search_key, width=entry_width)
        entry.grid(row=grid_row, column=1, sticky=Tk.W)
        w = Tk.Button(input_frame, text="Fetch", width=10, command=self.start_fetch_thread)
        w.grid(row=grid_row, column=2, padx=20, pady=5)
        self.fetch_btn = w

        grid_row += 1
        label = Tk.Label(input_frame, text="Out Folder: ")
        label.grid(row=grid_row, column=0, sticky=Tk.E)
        self.out_folder = Tk.StringVar()
        out_folder = get_setting("analysis_folder") + r"\PDB\000"
        out_folder = get_new_folder(out_folder)
        self.out_folder.set(out_folder)
        entry = FolderEntry( input_frame, textvariable=self.out_folder, width=entry_width,
                            slimbutton=True)
        entry.grid(row=grid_row, column=1, sticky=Tk.W)

        self.log_text = ScrolledText(log_frame, width=78, height=10 )
        self.log_text.pack()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w.pack_forget()
        self.ok_btn = w

        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.cancel_btn = w

        self.bind("<Escape>", self.cancel)

    def apply(self):
        print('apply')
        self.applied = True

    def check_module(self):
        try:
            import requests
            import pypdb
            import mrcfile
            ok = True
        except:
            ok = False

        if not ok:
            from UpdateDailog import UpdateDailog
            modules = ['requests', 'pypdb', 'mrcfile']
            dialog = UpdateDailog(self, modules)
            dialog.show()
            if dialog.applied:
                ok = True

        return ok

    def start_fetch_thread(self):
        ok = self.check_module()
        if not ok:
            return

        from molass_legacy.KekLib.BasicUtils import get_home_folder
        self.fetch_btn.config(state=Tk.DISABLED)

        out_folder = self.out_folder.get()
        out_folder = get_new_folder(out_folder)
        self.out_folder.set(out_folder)

        log_folder = get_home_folder() + '/log'
        log_file = log_folder + '/fetcher.log'
        self.root_logger  = logging.getLogger()
        self.root_logger.addHandler( ProgressLoghandler(self.log_text, log_file) )
        self.queue = queue.Queue()
        self.thread = Thread(
                        target=self.fetch,
                        name='DenssThread',
                        args=[]
                        )
        self.thread.start()
        self.counter = 0
        self.maximum = None
        self.last_index = self.log_text.index(Tk.INSERT)
        self.after(200, self.update_progress_text)      # 200: need some time to have self.log_text written enough

    def update_progress_text(self):
        try:
            while True:
                ret = self.queue.get(block=False)
                if ret[0] == 0:
                    self.thread.join()
                    self.cancel_btn.pack_forget()
                    self.ok_btn.pack(side=Tk.LEFT, padx=5, pady=5)
                    self.fetch_btn.config(state=Tk.NORMAL)
                    return
                else:
                    message = ret[1]
                    if self.counter > 0:
                        self.log_text.delete( self.last_index, Tk.INSERT )

                    self.last_index = self.log_text.index(Tk.INSERT)
                    if len(message) > 0 and message[0] == '\r':
                        self.counter += 1
                    else:
                        # self.counter = 0
                        pass
                    self.log_text.insert(Tk.INSERT, message)

        except queue.Empty:
            pass
        self.after(100, self.update_progress_text)

    def fetch(self):
        from time import sleep
        from .PypdbLite import Query, get_pdb_file
        self.redirector = StdoutRedirector(self.queue)

        pdbid = self.search_key.get()
        print("Query('%s')" % pdbid)
        results = Query(pdbid).search()
        self.queue.put((1, ""))

        if len(results) > 0:
            out_folder = self.out_folder.get()
            mkdirs_with_retry(out_folder)
            ret = get_pdb_file(pdbid, filetype='pdb')
            # print(ret)
            pdb_file = os.path.join(out_folder, pdbid + '.pdb')
            try:
                with open(pdb_file, "w") as fh:
                    fh.write(ret)
            except Exception as exc:
                print("Error", exc)
                return
            print("Saved to %s" % pdb_file)
            self.queue.put((1, ""))
            self.mrc_file = run_pdb2mrc(pdb_file, queue=self.queue)
            if self.mrc_file is not None:
                print("It is ready. Press OK.")
        else:
            print("Not found.")

        self.queue.put((0, ""))

    def get_mrc_file_path(self):
        return self.mrc_file
