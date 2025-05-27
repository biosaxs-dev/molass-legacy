# coding: utf-8
"""
    SecTools.CorMap.CorMapTestUtils.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import os
from molass_legacy.KekLib.OurTkinter import Tk

class Manipulator:
    def __init__(self, restart_str=None):
        self.counter = -1
        self.restart_str = restart_str
        self.restarting = restart_str is not None
        self.prepare_out_folder()

    def prepare_out_folder(self):
        self.out_folder = "temp/figs"
        if not os.path.exists(self.out_folder):
            from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
            mkdirs_with_retry(self.out_folder)

    def do_for_a_file(self, in_folder, uv_folder, plot):
        assert in_folder == uv_folder
        self.counter += 1
        if self.restarting:
            if in_folder.find(self.restart_str) >= 0:
                self.restarting = False
            else:
                print("skipping", in_folder)
                return True, None
        print(in_folder)
        agent = self.agent
        agent.fe.focus_force()
        agent.fe.delete( 0, Tk.END)
        agent.fe.insert( 0, in_folder )
        agent.on_in_folder_entry()
        file = os.path.join(self.out_folder, "fig-%03d" % self.counter)
        agent.save_the_figure(file)
        return True, None

    def manipulate(self, client, agent):
        from DataUtils import get_pytools_folder, serial_folder_walk
        self.agent = agent
        self.client = client
        pytools = get_pytools_folder()
        root_folder = os.path.join(pytools, "Data")
        serial_folder_walk(root_folder, self.do_for_a_file)
