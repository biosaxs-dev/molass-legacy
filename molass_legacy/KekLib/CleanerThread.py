# coding: utf-8
"""
    CleanerThread.py

    craetes a thread which terminates its own process
    when its parent's process have died

    borrowed from
    Python: how to kill child process(es) when parent dies?
    https://stackoverflow.com/questions/23434842/python-how-to-kill-child-processes-when-parent-dies
"""
import os
import logging
from ctypes import WinDLL, WinError
from ctypes.wintypes import DWORD, BOOL, HANDLE
from molass_legacy.KekLib.KillableThread import Thread

class CleanerThread:
    def __init__(self, parent_pid, cleanup=None):
        self.parent_pid = parent_pid
        self.cleanup = cleanup
        self._start()

    def start(self):
        pass

    def _start(self):
        self.thread = Thread( target=self.wait_proc, name='WaitThread', args=[] )
        self.thread.start()

    def wait_proc(self):
        # Magic value from http://msdn.microsoft.com/en-us/library/ms684880.aspx
        SYNCHRONIZE = 0x00100000
        kernel32 = WinDLL("kernel32.dll")
        kernel32.OpenProcess.argtypes = (DWORD, BOOL, DWORD)
        kernel32.OpenProcess.restype = HANDLE
        parent_handle = kernel32.OpenProcess(SYNCHRONIZE, False, self.parent_pid)
        # Block until parent exits
        try:
            os.waitpid(parent_handle, 0)
        except Exception as exc:
            print(exc)
            # [Errno 13] Permission denied
        logger = logging.getLogger(__name__)

        if self.cleanup is not None:
            logger.info("child process cleanup call.")
            self.cleanup()
            logger.info("child process cleanup done.")

        pid = os.getpid()
        logger.info("terminates child process %d of died parent process %d", pid, self.parent_pid)
        os._exit(0)

    def set_cleanup(self, cleanup):
        self.cleanup = cleanup

    def __del__(self):
        self.thread.terminate()
