"""
    DenssGui.py

    Copyright (c) 2019-2024, SAXS Team, KEK-PF
"""
import sys
import os
import re
import logging
from time import sleep
import numpy as np
import queue
import matplotlib.pyplot    as plt
from matplotlib.backends.backend_tkagg  import FigureCanvasTkAgg
from molass_legacy.KekLib.KillableThread import Thread
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk, ScrolledText, is_empty_val
from molass_legacy.KekLib.TkSupplements import set_icon
from molass_legacy.KekLib.TkCustomWidgets import FileEntry, FolderEntry
from molass_legacy.KekLib.ReadOnlyText import ReadOnlyText
import molass_legacy.KekLib.CustomMessageBox as MessageBox
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.OurMatplotlib import get_color
from .DenssUtils import get_outfolder
from molass_legacy.KekLib.NumpyUtils import np_loadtxt
import molass_legacy.KekLib.DebugPlot as dplt

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

log_step_re = re.compile(r'^\s*(\d+)')
log_maxnum_re = re.compile(r'Maximum number of steps: (\d+)')

class DenssGuiDialog(Dialog):
    def __init__(self, parent, q=None, a=None, e=None, infile_name=None, debug=False):
        self.debug = debug
        if debug:
            import logging
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = None
        self.grab = 'local'     # used in grab_set
        self.parent = parent
        self.denss_action = False   # used for test control in test_990_DenssGui.py
        self.q = q
        self.a = a
        self.e = e
        self.dmax = Tk.DoubleVar()
        self.dmax.set(100)
        if q is None:
            # as invoked from the DENSS tools menu
            pass
        else:
            # as invoked from the LRF preview
            self.prepare_from_args(q, a, e, infile_name)
        self.cwd_init = os.getcwd()
        self.thread = None
        from .denss._version import __version__ as denss_version
        Dialog.__init__( self, parent, "Gui for DENSS-%s" % denss_version, visible=False )

    def show(self):
        self._show()

    def add_dnd_bind(self):
        self.mpl_canvas_widget.register_drop_target("*")

        def dnd_handler(event):
            self.on_fig_dnd(event)

        self.mpl_canvas_widget.bind("<<Drop>>", dnd_handler)

    def body(self, body_frame):
        set_icon( self )

        upper_frame = Tk.Frame(body_frame)
        upper_frame.pack()
        lower_frame = Tk.Frame(body_frame)
        lower_frame.pack(fill=Tk.X)

        cframe = Tk.Frame(upper_frame)
        cframe.pack(side=Tk.LEFT, padx=10, pady=5)

        iframe = Tk.Frame(upper_frame)
        iframe.pack(side=Tk.LEFT, padx=10, pady=5)

        self.create_figure(cframe)

        grid_row = 0
        io_frame = Tk.Frame(iframe)
        io_frame.grid(row=grid_row, column=0)

        grid_row_io = 0

        if self.q is None:
            self.is_memory_input = False
            in_file_label = Tk.Label(io_frame, text='Input File: ')
            in_file_label.grid(row=grid_row_io, column=0, pady=5)
            self.in_file = Tk.StringVar()
            self.in_file.set(None)
            self.in_file_entry = FileEntry(io_frame, textvariable=self.in_file, width=70, on_entry_cb=self.on_file_entry)
            self.in_file_entry.grid(row=grid_row_io, column=1, pady=10)
        else:
            self.is_memory_input = True
            self.input_info_label = Tk.Label(io_frame, text='Input is from the memory equivalent to %s' % self.infile_name, bg='white')
            self.input_info_label.grid(row=grid_row_io, column=0, columnspan=2, pady=10)

        grid_row_io += 1
        dmax_label = Tk.Label(io_frame, text='Dmax: ')
        dmax_label.grid(row=grid_row_io, column=0, sticky=Tk.E)
        dmax_entry_frame = Tk.Frame(io_frame)
        dmax_entry_frame.grid(row=grid_row_io, column=1, columnspan=2, sticky=Tk.W)
        self.dmax_entry = Tk.Entry(dmax_entry_frame, textvariable=self.dmax, width=8, justify=Tk.CENTER)
        self.dmax_entry.grid(row=0, column=0, sticky=Tk.W)
        self.illust_btn = Tk.Button(dmax_entry_frame, text="Illustrate", command=self.illustrate_dmax)
        self.illust_btn.grid(row=0, column=1, padx=40)
        self.fit_data_btn = Tk.Button(dmax_entry_frame, text="Show denss.fit_data.py figure", command=self.denss_fit_data)
        self.fit_data_btn.grid(row=0, column=12)

        grid_row_io += 1
        out_folder_label = Tk.Label(io_frame, text='Output Folder: ')
        out_folder_label.grid(row=grid_row_io, column=0)

        out_folder_init = get_outfolder(parent=self.parent)

        self.out_folder = Tk.StringVar()
        self.out_folder.set(out_folder_init)
        self.out_folder_entry = FolderEntry(io_frame, textvariable=self.out_folder, width=70, on_entry_cb=self.on_folder_entry)
        self.out_folder_entry.grid(row=grid_row_io, column=1, sticky=Tk.W, pady=5)

        grid_row += 1
        self.progress_label = Tk.Label(iframe, text="DENSS Progress" )
        self.progress_label.grid(row=grid_row, column=0)

        grid_row += 1
        self.progress_text = ReadOnlyText(iframe, width=80, height=5)
        self.progress_text.grid(row=grid_row, column=0)
        self.progress_text.update()

        grid_row += 1
        length = self.progress_text.winfo_width()
        self.progress_bar = ttk.Progressbar(iframe, orient ="horizontal", length=length, mode="determinate")
        self.progress_bar.grid(row=grid_row, column=0, pady=10)

        grid_row += 1
        self.log_text_label = Tk.Label(iframe, text="Log from DENSS" )
        self.log_text_label.grid(row=grid_row, column=0)

        grid_row += 1
        self.log_text = ScrolledText(iframe, width=78, height=20 )
        self.log_text.grid(row=grid_row, column=0)

        self.guide_message = Tk.StringVar()
        self.guide_message_label = Tk.Label(lower_frame, bg='white', textvariable=self.guide_message)
        self.guide_message_label.pack(fill=Tk.X, padx=10, pady=10)
        self.update_guide_message()

        if False:
            fig = dplt.figure()
            ax = fig.gca()
            ax.plot( self.q, self.a, label='data' )
            ax.plot( self.qc, self.ac, label='fit data' )
            ax.legend()
            dplt.show()

    def create_figure(self, cframe):
        self.fig = fig = plt.figure( figsize=(7, 6) )
        self.mpl_canvas = FigureCanvasTkAgg( fig, cframe )
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )
        self.ax = fig.gca()

        if self.q is None:
            self.add_dnd_bind()
            self.ax.text(0.5, 0.5, "Drag and drop\nor enter\nan input file.", alpha=0.3,
                        fontsize=36, ha='center', va='center')
        else:
            self.draw_curve()

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()

        self.ok_btn = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        self.ok_btn.pack(side=Tk.LEFT, padx=5, pady=5)
        self.ok_btn.pack_forget()

        self.cancel_btn = Tk.Button(box, text="Cancel", width=10, command=self.ask_cancel)
        self.cancel_btn.pack(side=Tk.LEFT, padx=5, pady=5)

        self.run_btn = Tk.Button(box, text="Run here", width=20, command=self.run_denss_thread)
        self.run_btn.pack(side=Tk.LEFT, padx=5, pady=5)

        self.submit_btn = Tk.Button(box, text="Run in background", width=20, command=self.submit_to_denss_manager)
        self.submit_btn.pack(side=Tk.LEFT, padx=5, pady=5)

        self.show_btn = Tk.Button(box, text="Show Manager", width=14, command=self.show_denss_manager)
        self.show_btn.pack(side=Tk.LEFT, padx=5, pady=5)

        self.edv_btn = Tk.Button(box, text="Electron Density Viewer", width=20, command=self.show_ed_viewer)
        self.edv_btn.pack(side=Tk.LEFT, padx=5, pady=5)

        self.update_button_states()

        # self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def is_ready(self):
        if self.is_memory_input:
            return not is_empty_val(self.out_folder.get())
        else:
            if is_empty_val(self.in_file.get()) or is_empty_val(self.out_folder.get()):
                return False
            else:
                return True

    def update_button_states(self):
        state = Tk.NORMAL if self.is_ready() else Tk.DISABLED
        for b in [self.illust_btn, self.fit_data_btn, self.run_btn, self.submit_btn]:
            b.config(state=state)

    def update_guide_message(self, message=None):

        if message is None:
            entry_names = []
            if self.is_memory_input:
                pass
            else:
                if is_empty_val(self.in_file.get()):
                    entry_names.append('"Input File"')

            if is_empty_val(self.out_folder.get()):
                entry_names.append('"Output Folder"')

            if len(entry_names) > 0:
                message = "You need to specify %s." % (" and ".join(entry_names))
            else:
                message = "You are ready to Run."

        self.guide_message.set(message)

    def ok(self, event=None):

        if not self.validate():
            self.initial_focus.focus_set() # put focus back
            return

        self.withdraw()
        self.update_idletasks()

        try:
            self.apply()
        finally:
            self.cancel()

        os.chdir(self.cwd_init)

    def ask_cancel(self):
        if self.thread is None:
            self.cancel()
            os.chdir(self.cwd_init)
            return

        yn = MessageBox.askyesno(
                'Cancel Confirmation',
                'Do you really want to stop the execution?',
                parent=self,
                )
        if not yn:
            return

        self.thread.terminate()
        self.thread.join()
        self.cancel_btn.pack_forget()
        self.ok_btn.pack(side=Tk.LEFT, padx=5, pady=5)
        os.chdir(self.cwd_init)
        logging.info('Execution canceled by user.')

    def prepare_from_args(self, q, a, e, infile_name):
        self.data = np.array([q, a, e]).T
        self.apply_fit_data(q, a, e, infile_name, use_memory_data=True)
        self.dmax.set(round(self.sasrec.D,2))

    def apply_fit_data(self, q, a, e, infile_name, use_memory_data=False, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.DENSS.DenssUtils as DenssUtils
            reload(DenssUtils)
        from .DenssUtils import fit_data_impl
        sasrec, work_info = fit_data_impl(q, a, e, file=infile_name, gui=True, use_memory_data=use_memory_data)
        self.sasrec = sasrec
        self.work_info = work_info
        self.qc = sasrec.qc
        self.ac = sasrec.Ic
        self.ec = sasrec.Icerr
        _, self.infile_name = os.path.split(infile_name)

    def on_file_entry(self):
        self.on_input(self.in_file.get())

    def on_folder_entry(self):
        self.update_button_states()
        self.update_guide_message()

    def on_fig_dnd(self, event):
        files = event.data.split(' ')
        file = files[0]
        self.in_file.set(file)
        self.on_input(file)

    def on_input(self, file):
        data, _ = np_loadtxt(file)
        q, a, e = data[:,0], data[:,1], data[:,2]
        self.in_file.set(file)
        self.q = q
        self.a = a
        self.e = e
        self.data = data
        self.apply_fit_data(q, a, e, file)
        self.dmax.set(round(self.sasrec.D,2))
        self.draw_curve()
        self.update_button_states()
        self.update_guide_message()

    def draw_curve(self):
        ax = self.ax
        ax.cla()
        ax.semilogy()
        ax.set_title( "Extrapolated profile: " + self.infile_name )
        ax.set_xlabel('Q')
        ax.set_ylabel('Intensity ($Log_{10}$)')
        ax.plot(self.q, self.a, label='input curve (LRF result)', color=get_color(1))
        ax.plot(self.qc, self.ac, label='modified curve with I(q=0)', color=get_color(2))
        ax.legend()
        self.fig.tight_layout()
        self.mpl_canvas.draw()

    def run_denss_thread(self):
        self.update_guide_message(message="Running DENSS.")
        self.run_btn.pack_forget()
        self.submit_btn.pack_forget()
        self.show_btn.pack_forget()
        self.edv_btn.pack_forget()
        self.fit_data_btn.config(state=Tk.DISABLED)     # avoid pressing this button while running denss
        out_folder = self.out_folder.get()
        mkdirs_with_retry(out_folder)
        self.root_logger  = logging.getLogger()
        log_file = out_folder + '/denss.log'
        self.root_logger.addHandler( ProgressLoghandler(self.log_text, log_file) )
        self.queue = queue.Queue()
        self.thread = Thread(
                        target=self.denss_thread,
                        name='DenssThread',
                        args=[]
                        )
        self.thread.start()
        self.counter = 0
        self.maximum = None
        self.last_index = self.progress_text.index(Tk.INSERT)
        self.after(200, self.update_progress_text)      # 200: need some time to have self.log_text written enough

    def update_progress_text(self):
        if self.maximum is None:
            text = self.log_text.get('1.0', Tk.END)
            """
            with open('debug.txt', 'w') as fh:
                fh.write(text)
            """
            m = log_maxnum_re.search(text)
            if m:
                self.maximum = int(m.group(1))
            else:
                self.maximum = 9999
            # print('maximum=', self.maximum)
            self.progress_bar['maximum'] = self.maximum

        try:
            while True:
                ret = self.queue.get(block=False)
                if ret[0] == 0:
                    if self.logger is not None:
                        self.logger.info("ret=%s", str(ret))

                    self.thread.join()
                    self.progress_bar['value'] = self.maximum
                    self.cancel_btn.pack_forget()
                    self.ok_btn.pack(side=Tk.LEFT, padx=5, pady=5)
                    self.edv_btn.pack(side=Tk.LEFT, padx=5, pady=5)
                    self.fit_data_btn.config(state=Tk.NORMAL)
                    self.update_guide_message(message="DENSS Done.")
                    return
                else:
                    message = ret[1]
                    if self.counter > 0:
                        self.progress_text.delete( self.last_index, Tk.INSERT )

                    self.last_index = self.progress_text.index(Tk.INSERT)
                    if len(message) > 0 and message[0] == '\r':
                        self.counter += 1
                        m = log_step_re.search(message[1:])
                        if m:
                            self.progress_bar['value'] = int(m.group(1))
                            self.progress_bar.update()
                    else:
                        # self.counter = 0
                        pass
                    self.progress_text.insert(Tk.INSERT, message)

        except queue.Empty:
            pass
        self.after(100, self.update_progress_text)

    def denss_thread(self, debug=True):
        os.chdir(self.out_folder.get())
        redirector = StdoutRedirector(self.queue)

        from molass_legacy.Env.EnvInfo import get_global_env_info
        if debug:
            from importlib import reload
            import molass_legacy.DENSS.DenssUtils as DenssUtils
            reload(DenssUtils)
        from .DenssUtils import run_denss_impl

        env_info = get_global_env_info(gpu_info=True)

        use_gpu = (env_info is not None and env_info.nvidiagpu_is_available)
        if use_gpu:
            logging.info('usign GPUs in DENSS.')

        dmax = self.dmax.get()
        # gui = False ok because redirector sets sys.stdout
        run_denss_impl(self.qc, self.ac, self.ec, dmax, self.infile_name, use_gpu=use_gpu)

        logging.info('DENSS thread finished.')
        sleep(0.3)      # to avoid hanging in pythonw.exe (while not sure if this is really effective)
        self.queue.put([0])

        os.chdir(self.cwd_init)

    def _cancel(self):
        os.chdir(self.cwd_init)
        self.cancel()

    def denss_fit_data(self, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.DENSS.DenssFitData as DenssFitData
            reload(DenssFitData)
        from .DenssFitData import DenssFitDataDialog
        dialog = DenssFitDataDialog(self, self.sasrec, self.work_info, self.infile_name, self.out_folder.get())
        dialog.show()
        if dialog.applied:
            sasrec = dialog.get_sasrec()
            self.dmax.set(round(sasrec.D,2))

    def submit_to_denss_manager(self):
        from .DenssManager import JobInfo
        from .DenssManagerDialog import show_manager_dialog
        dmax = self.dmax.get()
        job = JobInfo('denss', q=self.qc, a=self.ac, e=self.ec, dmax=dmax, infile_name=self.infile_name)
        def show_dialog():
            self._cancel()
            show_manager_dialog(self.parent, [job])
        self.parent.after(0, show_dialog)
        self.config(cursor='wait')
        self.denss_action = True

    def show_denss_manager(self):
        from .DenssManagerDialog import show_manager_dialog
        def show_dialog():
            self._cancel()
            show_manager_dialog(self.parent)
        self.parent.after(0, show_dialog)
        self.config(cursor='wait')
        self.denss_action = True

    def show_ed_viewer(self):
        print('show_ed_viewer')
        from molass_legacy.Saxs.EdViewer import EdViewer
        from molass_legacy.KekLib.OurMatplotlib import reset_to_default_style
        viewer = EdViewer(self)
        viewer.show()
        reset_to_default_style()

    def illustrate_dmax(self, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.DENSS.DmaxEstimation as DmaxEstimation
            reload(DmaxEstimation)
        from .DmaxEstimation import illustrate_dmax
        from molass_legacy.KekLib.DebugPlot import DialogWrapper

        with DialogWrapper(parent=self, window_title="Illustration"):
            fig, ax = dplt.subplots()
            ax.set_title("Dmax Estimation illustrated", fontsize=20)
            illustrate_dmax(ax, self.data)
            fig.tight_layout()
