"""
    StdoutLogger.py

    adapted from
    https://stackoverflow.com/questions/616645/how-to-duplicate-sys-stdout-to-a-log-file

    Copyright (c) 2020, Masatsuyo Takahashi, KEK-PF
"""
import sys

class StdoutLogger(object):
    def __init__(self, logfile="stdout.log"):
        self.terminal = sys.stdout
        self.log = open(logfile, "a", encoding='utf-8')
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __del__(self):
        self.log.close()
        sys.stdout = self.terminal
