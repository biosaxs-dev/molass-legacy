"""
    Processes.GUI.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
def gui_loop(si, pn, queues):
    print("gui_loop")

    from molass_legacy.KekLib.TkUtils import get_tk_root

    root = get_tk_root()

    root.after(100, lambda: gui_main(si, pn, root, queues))
    root.mainloop()
    root.destroy()

def gui_main(si, pn, root, queues):
    from GuiProcess.GuiMain import GuiMain
    print("gui_main", si)

    gm = GuiMain(root, si, pn, queues)
    gm.show()

    for q, stop_signal in zip(queues[1:], ["__stop__", -9, -9]):
        q.put(stop_signal)

    root.quit()
    print("gui quit")
