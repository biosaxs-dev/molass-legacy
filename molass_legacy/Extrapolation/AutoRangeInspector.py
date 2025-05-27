# coding: utf-8
"""
    AutoRangeInspector.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import sys
from time import sleep
import queue
import logging
import numpy as np
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.ReadOnlyText import ReadOnlyText
from molass_legacy.KekLib.KillableThread import Thread
from TkTester import AppAgent

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

class AutoRangeInspector(Dialog):
    def __init__(self, parent, inspector):
        self.inspector = inspector
        Dialog.__init__(self, parent, "Auto Range Inspector", visible=False)

    def body(self, body_frame):
        self.log_text = ReadOnlyText(body_frame, width=80, height=10)
        self.log_text.pack()
        self.log_text.update()

    def inspect(self):
        self.after(100, self.inspect_start)
        self._show()

    def inspect_start(self):
        self.agent = AppAgent(self.inspector)
        self.queue = queue.Queue()
        self.thread = Thread(
                        target=self.inspect_thread,
                        name='InspectThread',
                        args=[]
                        )
        self.thread.start()

        self.watch()

    def watch(self):
        try:
            while True:
                ret = self.queue.get(block=False)
                if ret[0] == 0:
                    self.thread.join()
                    self.ok()
                    return
                else:
                    message = ret[1]
                    self.log_text.insert(Tk.INSERT, message)
                    self.log_text.see(Tk.END)
        except queue.Empty:
            pass

        self.after(100, self.watch)

    def inspect_thread(self):
        redirector = StdoutRedirector(self.queue)

        for k in range(10):
            print([k])
            self.agent.debug_print(str([k]))
            sleep(1)

        logging.info('Inspect thread finished.')
        self.queue.put([0])
