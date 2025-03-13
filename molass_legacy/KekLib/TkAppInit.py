# coding: utf-8
"""

    TkAppInit.py

    Copyright (c) 2020, Masatsuyo Takahashi, KEK-PF

    not yet used as of 20200630

"""

from MultiMonitor import get_selected_monitor

class TkAppInit:
    def __init__(self, root):
        m = get_selected_monitor()
        print(m)
        if m.width >= 2560:
            root.tk.call('tk', 'scaling', '-displayof', '.', 1)
