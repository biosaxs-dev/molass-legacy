"""

    UpdateDailog.py

    Copyright (c) 2020, Masatsuyo Takahashi, KEK-PF

"""
from time import sleep
from molass_legacy.KekLib.KillableThread import Thread
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ScrolledText
from molass_legacy.KekLib.TkSupplements import set_icon, BlinkingFrame
from molass_legacy.KekLib.StdoutRedirector import StdoutRedirector

class UpdateDailog(Dialog):
    def __init__(self, parent, modules):
        self.modules = modules
        self.applied = False
        Dialog.__init__( self, parent, "PyTools Update", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):   # overrides Dialog.body
        set_icon(self)

        message_frame = Tk.Frame(body_frame)
        message_frame.pack(padx=20, pady=10)

        message =   (
                'The following modules are resuired\n'
                'to perform the function you have just requested.\n'
                'Press "Update" to install them into the embeddables\n'
                'before you proceed further.\n'
                'Cancelling here will result in cancelling the function.'
                )

        label = Tk.Label(message_frame, text=message, bg='white', anchor=Tk.W, justify=Tk.LEFT)
        label.pack()

        table_frame = Tk.Frame(body_frame)
        table_frame.pack(padx=20, pady=10)

        log_frame = Tk.Frame(body_frame)
        log_frame.pack(padx=20, pady=10)

        i = 0
        name_width = 10
        state_width = 40
        module_label = Tk.Label(table_frame, text="Name", width=name_width, relief=Tk.GROOVE)
        module_label.grid(row=i, column=0)
        module_state = Tk.Label(table_frame, text="State", width=state_width, relief=Tk.GROOVE)
        module_state.grid(row=i, column=1)

        self.state_frames = []
        for i, m in enumerate(self.modules, start=1):
            module_label = Tk.Label(table_frame, text=m, width=name_width, relief=Tk.GROOVE, bg="white")
            module_label.grid(row=i, column=0, sticky=Tk.W)
            state_frame = BlinkingFrame(table_frame)
            state_frame.grid(row=i, column=1, sticky=Tk.W)
            module_state = Tk.Label(state_frame, text="required to install", width=state_width,
                                    anchor=Tk.W, justify=Tk.LEFT, relief=Tk.GROOVE, bg="white")
            module_state.pack()
            state_frame.objects = [module_state]
            self.state_frames.append(state_frame)

        self.log_text = ScrolledText(log_frame, height=8)
        self.log_text.pack(fill=Tk.X)
        self.redirexctor = StdoutRedirector(self.log_text)

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="Update", width=10, command=self.update_modules, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.update_btn = w

        w = Tk.Button(box, text="OK", width=10, command=self.ok)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w.pack_forget()
        self.ok_btn = w

        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        self.cancel_btn = w

        self.bind("<Escape>", self.cancel)

    def apply(self):
        self.applied = True

    def update_modules(self):
        failed = False
        for k, m in enumerate(self.modules):
            state_frame = self.state_frames[k]
            state_frame.start()
            module_state = state_frame.objects[0]
            module_state.config(text="installing")
            self.start_install(m)
            self.wait_install_complete()
            state_frame.stop()
            if self.ret.returncode == 0:
                state = "installed"
                fg_color = 'black'
            else:
                state = "failed to install"
                failed = True
                fg_color = 'red'
            module_state.config(text=state, fg=fg_color)

        if not failed:
            self.update_btn.pack_forget()
            self.cancel_btn.pack_forget()
            self.ok_btn.pack(side=Tk.LEFT, padx=5, pady=5)

    def start_install(self, m):
        self.install_thread = Thread(
                        target=self.install,
                        name='InstallThread',
                        args=[m]
                        )
        self.install_complete = False
        self.install_thread.start()

    def install(self, m):
        import sys
        import subprocess
        from SubProcess import get_startup_info
        print('installing', m)
        python  = sys.executable.replace('pythonw.exe', 'python.exe')
        ret = subprocess.run([python, '-m', 'pip', 'install', '-U', m], capture_output=True, startupinfo=get_startup_info())
        stdout = ret.stdout.decode()
        stderr = ret.stderr.decode()
        print(stdout)
        if ret.returncode != 0:
            print(stderr)
        self.install_complete = True
        self.ret = ret

    def wait_install_complete(self):
        while not self.install_complete:
            self.update()
            sleep(0.1)
