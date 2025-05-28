"""
    DenssManagerDialog.py

    Copyright (c) 2019-2025, SAXS Team, KEK-PF
"""
import numpy as np
from time import strftime, localtime
import logging
import queue
from molass_legacy.KekLib.OurTkinter import Tk, Dialog, ttk
from molass_legacy.KekLib.TkUtils import get_widget_geometry
from molass_legacy.KekLib.TkSupplements import set_icon
from molass_legacy.KekLib.ScrolledFrame import ScrolledFrame
from .DenssUtils import MAXNUM_STEPS

dialogs = []
# DIALOG_HEIGHT = 250
PROGRES_BAR_SCALE = 3
PROGRES_BAR_WIDTH = 240 # 80*3
REFRESH_INTERVAL = 500
VALUE_FORMATS = ["%d", "%4.2e", "%3.2f", "%5i"]
STATE_TEXT = ["running", "started", "waiting", "finished", "error", "canceled"]
STATE_COLOR = ["black", "black", "black", "green", "red", "orange"]

def format_time(t):
    return "" if t is None else strftime("%H:%M:%S", localtime(t))

def format_duration(t):
    if t is None:
        return ""
    else:
        minutes, seconds = divmod(int(t), 60)
        return "%d:%02d" % (minutes, seconds) if minutes > 0 else "%d" % (seconds)

def get_state_text(state):
    i = abs(state)
    return STATE_TEXT[i]

def get_state_color(state):
    i = abs(state)
    return STATE_COLOR[i]

def get_ch2_fg_color(value):
    color = 'black'

    if value is not None:
        if value > 0.1:
            color = 'red'
        elif value > 0.01:
            color = 'orange'

    return color

def show_manager_dialog(parent, jobs=None):

    if len(dialogs) > 0:
        dialog = dialogs[0]
        dialog.focus_force()
        return

    from .DenssManager import activate_manager, get_list
    from molass_legacy.KekLib.BasicUtils import get_home_folder
    from molass_legacy.Env.EnvInfo import get_global_env_info

    log_folder = get_home_folder() + '/log'
    env_info = get_global_env_info(gpu_info=True)
    use_gpu = env_info.nvidiagpu_is_available
    num_workers = activate_manager(log_folder, use_gpu)

    dialog = DenssManagerDialog(parent, num_workers, use_gpu)
    dialogs.append(dialog)

    def post_processor():
        dialog.update()

        if jobs is not None:
            dialog.submit_jobs(jobs)

        job_list = get_list()
        num_jobs = max(1, len(job_list))
        dialog.update_geometry(num_jobs)

    parent.after(100, post_processor)

    dialog.show()
    dialogs.pop()

def terminate_manager(parent):
    from .DenssManager import get_list
    job_list = get_list()
    num_remaining_jobs = 0
    for i, row_rec in enumerate(job_list):
        job_id, job_rec = row_rec
        state = job_rec[2]
        if state > -3:
            num_remaining_jobs += 1
    if num_remaining_jobs > 0:
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        MessageBox.showinfo(
                'Cancel All Confirmation',
                'There are %d DENSS jobs remaininig.\n' % num_remaining_jobs
                + 'You must wait or cancel them before exit\n'
                + 'in the "DENSS Manager Dialog", which will appear next.',
                parent=parent,
                )
        show_manager_dialog(parent)
        return False

    from .DenssManager import terminate_manager_impl
    terminate_manager_impl()
    return True

class DenssManagerDialog(Dialog):
    def __init__(self, parent, num_workers, use_gpu):
        self.grab = 'local'     # used in grab_set
        self.parent = parent
        self.num_workers = num_workers
        self.use_gpu = use_gpu
        self.repeat_refresh = True
        self.basic_geometry = None
        Dialog.__init__(self, parent, "DENSS Process Manager", visible=False)

    def show(self):
        self.parent.after(REFRESH_INTERVAL, self.refresh)
        self._show()

    def body(self, body_frame):
        set_icon( self )

        t_frame = Tk.Frame(body_frame)
        t_frame.pack(fill=Tk.X, expand=1, padx=10, pady=20)

        label = Tk.Label(t_frame, text="Number of Worker Processes: ")
        label.grid(row=0, column=0)
        label = Tk.Label(t_frame, text="%d" % self.num_workers, bg='white')
        label.grid(row=0, column=1)
        if self.use_gpu:
            label = Tk.Label(t_frame, text="Using GPU", bg='white')
            label.grid(row=0, column=2, padx=20)

        s_frame = ScrolledFrame(body_frame)
        s_frame.pack(padx=10, pady=5)
        i_frame = s_frame.interior

        items = ['Job', 'Analysis Name', 'File Name', 'PID', 'State', 'Progress (max=%d)' % MAXNUM_STEPS, 'Step', 'Chi2', 'Rg', 'Support Volume', 'Submitted', 'Started', 'Finished', 'Duration', 'Cancel']

        for j, item in enumerate(items):
            label = Tk.Label(i_frame, text=item)
            label.grid(row=0, column=j)

        widths = [30, 80, 80, 40, 60, PROGRES_BAR_WIDTH, 50, 50, 50, 50, 60, 60, 60, 40, 40]
        self.total_width = np.sum(widths)
        for j, w in enumerate(widths):
            space = Tk.Frame(i_frame, width=w)
            space.grid(row=1, column=j)

        self.widget_table = []
        self.i_frame = i_frame

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack(fill=Tk.X)

        w = Tk.Button(box, text="Cancel All", width=10, command=self.cancel_all_jobs, fg='red', bg='khaki')
        w.pack(side=Tk.RIGHT, padx=40, pady=5)
        self.calcel_all_btn = w

        f = Tk.Frame(box)
        f.pack(side=Tk.RIGHT, fill=Tk.X, expand=1)

        w = Tk.Button(f, text="Close and Keep", width=14, command=self.ok, default=Tk.ACTIVE)
        w.pack()

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def update_geometry(self, num_jobs):
        self.update()
        btn_w, btn_h, btn_x, btn_y = get_widget_geometry(self.calcel_all_btn)
        # w, h, x, y = get_widget_geometry(self)

        print('update_geometry', btn_w, btn_h, btn_x, btn_y)
        # print('update_geometry', w, h, x, y)
        w_ = int(btn_w * 14.5)
        h_ = btn_h*(num_jobs+7)
        self.geometry('%dx%d' % (w_, h_))
        self.update()

    def update_table(self):
        from .DenssManager import get_list
        job_list = get_list()
        if False:
            print(strftime("%H:%M:%S"),'update_table')
            for rec in job_list:
                print(rec)

        if self.basic_geometry is None:
            # w, h, x, y
            self.basic_geometry = get_widget_geometry(self.calcel_all_btn)

        i_frame = self.i_frame
        bg_color = 'white'

        need_refresh = False
        for i, row_rec in enumerate(job_list):
            job_id, job_rec = row_rec
            analysis, file = job_rec[0]
            pid = job_rec[1]
            state = job_rec[2]
            step = job_rec[3]
            chi2 = job_rec[4]
            rg = job_rec[5]
            vol = job_rec[6]
            submitted = job_rec[7]
            started = job_rec[8]
            finished = job_rec[9]
            duration = job_rec[10]

            if i < len(self.widget_table):
                widget_rec = self.widget_table[i]

                if state > -3:
                    need_refresh = True

                update_duration = False
                if pid is None:
                    if state == 0:
                        pass
                    elif state <= -3:
                        finished_label = widget_rec[12]
                        text = finished_label.cget('text')
                        if text == '':
                            pid_label = widget_rec[3]
                            pid_label.config(text='')
                            finished_label.config(text=format_time(finished))
                            state_ = get_state_text(state)
                            color = get_state_color(state)
                            state_label = widget_rec[4]
                            state_label.config(text=state_, fg=color)
                            cancel_button = widget_rec[14]
                            cancel_button.config(state=Tk.DISABLED)
                            update_duration = True
                else:
                    pid_label = widget_rec[3]
                    text = pid_label.cget('text')
                    if text == "":
                        pid_label.config(text=str(pid))

                    if state < 0 or step < 100:
                        state_ = get_state_text(state)
                        color = get_state_color(state)
                        state_label = widget_rec[4]
                        state_label.config(text=state_, fg=color)
                        for k, time_ in enumerate([submitted, started]):
                            text = strftime("%H:%M:%S", localtime(time_))
                            time_label = widget_rec[k+10]
                            time_label.config(text=text)
                        cancel_button = widget_rec[14]
                        cancel_button.config(state=Tk.DISABLED if state <= -3 else Tk.NORMAL)

                    update_duration = True

                if state < 0:
                    if state == -1:
                        update_duration = True
                else:
                    step_label = widget_rec[6]
                    step_label.config(text=VALUE_FORMATS[0] % step)
                    chi2_label = widget_rec[7]
                    fg_color = get_ch2_fg_color(chi2)
                    chi2_label.config(text=VALUE_FORMATS[1] % chi2, fg=fg_color)
                    rg_label = widget_rec[8]
                    rg_label.config(text=VALUE_FORMATS[2] % rg)
                    vol_label = widget_rec[9]
                    vol_label.config(text=VALUE_FORMATS[3] % vol)

                if update_duration:
                    duration_label = widget_rec[13]
                    duration_label.config(text=format_duration(duration))

                pbar = widget_rec[5]
                if step >= 0:
                    pbar["value"] = step
            else:
                need_refresh = True
                widget_rec = []
                row = i+1

                # [0] Job
                col = 0
                label = Tk.Label(i_frame, text="%03d" % job_id, bg=bg_color)
                label.grid(row=row, column=col)
                widget_rec.append(label)

                # [1] Analysis Name
                col +=1
                label = Tk.Label(i_frame, text=analysis, bg=bg_color)
                label.grid(row=row, column=col, padx=5)
                widget_rec.append(label)

                # [2] File Name
                col +=1
                label = Tk.Label(i_frame, text=file, bg=bg_color)
                label.grid(row=row, column=col)
                widget_rec.append(label)

                # [3] PID
                col +=1
                label = Tk.Label(i_frame, text="" if pid is None else str(pid), bg=bg_color)
                label.grid(row=row, column=col, padx=5)
                widget_rec.append(label)

                # [4] State
                col +=1
                color = get_state_color(state)
                label = Tk.Label(i_frame, text=get_state_text(state), bg=bg_color, fg=color)
                label.grid(row=row, column=col)
                widget_rec.append(label)

                # [5] Progress
                col +=1
                length = self.basic_geometry[0] * PROGRES_BAR_SCALE
                pbar = ttk.Progressbar(i_frame, orient ="horizontal", length=length, mode="determinate")
                pbar.grid(row=row, column=col, padx=10)
                pbar["maximum"] = MAXNUM_STEPS
                step_ = MAXNUM_STEPS if step == -3 else step
                pbar["value"] = max(0, step_)
                widget_rec.append(pbar)

                # [6] Step, [7] Chi2, [8] Rg, [9] Support Volume
                for k, value in enumerate([step, chi2, rg, vol]):
                    col +=1
                    text = "" if value is None or value < 0 else VALUE_FORMATS[k] % value
                    if k == 1:  # i.e., Ch2
                        fg_color = get_ch2_fg_color(value)
                    else:
                        fg_color = None
                    label = Tk.Label(i_frame, text=text, bg=bg_color, fg=fg_color)
                    label.grid(row=row, column=col, padx=5)
                    widget_rec.append(label)

                # [10] Submitted
                col +=1
                label = Tk.Label(i_frame, text=format_time(submitted), bg=bg_color)
                label.grid(row=row, column=col)
                widget_rec.append(label)

                # [11] Started
                col +=1
                label = Tk.Label(i_frame, text=format_time(started), bg=bg_color)
                label.grid(row=row, column=col, padx=5)
                widget_rec.append(label)

                # [12] Finished
                col +=1
                label = Tk.Label(i_frame, text=format_time(finished), bg=bg_color)
                label.grid(row=row, column=col)
                widget_rec.append(label)

                # [13] Duration
                col +=1
                label = Tk.Label(i_frame, text=format_duration(duration), bg=bg_color)
                label.grid(row=row, column=col)
                widget_rec.append(label)

                # [14] Cancel
                col +=1
                state_ = Tk.DISABLED if pid is None and state <= -3 else Tk.NORMAL
                button = Tk.Button(i_frame, text="Cancel", fg='red', bg='khaki', state=state_, command=lambda j_=job_id :self.cancel_job(j_))
                button.grid(row=row, column=col)
                widget_rec.append(button)

                self.widget_table.append(widget_rec)

        if need_refresh:
            if not self.repeat_refresh:
                self.repeat_refresh = True
                self.after(REFRESH_INTERVAL, self.refresh)
        else:
            self.repeat_refresh = False

        self.calcel_all_btn.config(state=Tk.NORMAL if self.repeat_refresh else Tk.DISABLED)

    def submit_jobs(self, jobs):
        from .DenssManager import submit
        for job_info in jobs:
            submit(job_info)

    def refresh(self):
        self.update_table()
        if self.repeat_refresh:
            self.after(REFRESH_INTERVAL, self.refresh)

    def apply(self):
        # print('apply')
        self.repeat_refresh = False
        self.update()

    def cancel_job_impl(self, job_id):
        from .DenssManager import get_table
        table = get_table()
        last_rec = table[job_id]
        last_rec[2] = -5    # canceled
        last_rec[11] = 1
        table[job_id] = last_rec

    def cancel_job(self, job_id):
        print('cancel_job', job_id)
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        yn = MessageBox.askyesno(
                'Cancel Job Confirmation',
                'Do you really want to cancel job %03d ?' % job_id,
                parent=self,
                )
        if not yn:
            return

        self.cancel_job_impl(job_id)

    def cancel_all_jobs(self):
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        yn = MessageBox.askyesno(
                'Cancel All Confirmation',
                'Do you really want to cancel all jobs?',
                parent=self,
                )
        if not yn:
            return

        from .DenssManager import get_list
        job_list = get_list()
        for i, row_rec in enumerate(job_list):
            job_id, job_rec = row_rec
            state = job_rec[2]
            if state >= -2:
                self.cancel_job_impl(job_id)
