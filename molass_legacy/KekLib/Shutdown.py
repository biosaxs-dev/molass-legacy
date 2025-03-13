# coding: utf-8
"""
    Shutdown.py

    from
        How to shutdown a computer using Python
        https://stackoverflow.com/questions/34039845/how-to-shutdown-a-computer-using-python

"""
import sys
import os

def shutdown_machine():
    if sys.platform == 'win32':
        os.system("shutdown /s /t 1")
    else:
        os.system('sudo shutdown now')

if __name__ == '__main__':
    shutdown_machine()
