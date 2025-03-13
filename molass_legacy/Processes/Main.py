"""
    Main.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""

def main():
    from multiprocessing import Process, Queue
    from .SharedInfo import SharedInfo, NUM_QUEUES
    from .Loader import loader
    from .RgComputer import rg_computer
    from .Recognizer import recognizer
    from .GUI import gui_loop
    from .Dummy import dummy

    queues = [Queue() for i in range(NUM_QUEUES)]

    print("main")
    si = SharedInfo()

    loader_proc = Process(target=loader, args=(si, 0, queues))
    loader_proc.start()

    rg_proc = Process(target=rg_computer, args=(si, 1, queues))
    rg_proc.start()

    recog_proc = Process(target=recognizer, args=(si, 2, queues))
    recog_proc.start()

    gui_proc = Process(target=gui_loop, args=(si, 3, queues))
    gui_proc.start()

    if False:
        dummy_proc = Process(target=dummy, args=(si, 9))
        dummy_proc.start()
