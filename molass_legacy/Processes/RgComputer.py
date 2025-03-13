"""
    Processes.RgComputer.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time, sleep

def rg_computer(si, pn, queues):
    import molass_legacy.KekLib, SerialAnalyzer
    from molass_legacy._MOLASS.SerialSettings import initialize_settings
    from .SharedInfo import QUEUE_RG
    from molass_legacy.GuinierAnalyzer.SimpleGuinier import SimpleGuinier
    initialize_settings()

    print("rg_computer", si)
    rg_queue = queues[QUEUE_RG]

    while True:
        i = rg_queue.get()
        if i < 0:
            if i == -1:
                t0 = time()
                si.get_buffer_ready()
            elif i == -2:
                print("It took %.3g seconds" % (time()-t0))
            else:
                break
            continue

        data = si.get_xr_data(i)
        sg = SimpleGuinier(data)
        # print([i], "rg computed", sg.Rg)

    print("rg_computer stop")
