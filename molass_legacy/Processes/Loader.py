"""
    Processes.Loader.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""

def loader(si, pn, queues):
    from time import sleep
    from .SharedInfo import QUEUE_GUI, QUEUE_LOADER, QUEUE_RG, QUEUE_RECOG
    from LoaderProcess.DataLoader import DataLoader
    print("loader", si)

    gui_queue = queues[QUEUE_GUI]
    loader_queue = queues[QUEUE_LOADER]
    rg_queue = queues[QUEUE_RG]
    recog_queue = queues[QUEUE_RECOG]

    dl = DataLoader(si, rg_queue)

    while True:
        in_folder = loader_queue.get()
        if in_folder == "__stop__":
            break

        print(f'Working on {in_folder}')
        dl.load(in_folder)
        gui_queue.put(1)
        recog_queue.put(1)
        print(f'Finished {in_folder}')

    print("loader stop")
