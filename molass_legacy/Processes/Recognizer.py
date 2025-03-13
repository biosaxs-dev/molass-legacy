"""
    Processes.Recognizer.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import time, sleep

def recognizer(si, pn, queues):
    import molass_legacy.KekLib, SerialAnalyzer
    from molass_legacy._MOLASS.SerialSettings import initialize_settings
    from .SharedInfo import QUEUE_RECOG
    initialize_settings()

    print("recognizer", si)
    recog_queue = queues[QUEUE_RECOG]

    while True:
        i = recog_queue.get()
        if i < 0:
            if i == -1:
                t0 = time()
            elif i == -2:
                print("It took %.3g seconds" % (time()-t0))
            else:
                break
            continue

    print("recognizer stop")
