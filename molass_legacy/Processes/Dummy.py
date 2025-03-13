"""
    Processes.Dummy.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
from time import sleep

def dummy(si, pn):
    print("dummy", si)

    si.in_folder.set("dummy_folder")

    while True:
        sleep(1)
        si.value.value += 1
        si.set_procstate(pn, si.value.value)
        print("dummy", si)
