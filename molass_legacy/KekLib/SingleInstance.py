"""
    SingleInstance.py

    borrowed from
        http://code.activestate.com/recipes/578453-python-single-instance-cross-platform/

    modified as follows:
        1) allow LOCK_PATH to be given by user
"""
import sys
import os

try:
    import fcntl
except ImportError:
    fcntl = None

OS_WIN = False
if 'win32' in sys.platform.lower():
    OS_WIN = True


class SingleInstance:
    def __init__(self, lock_path=None, raise_=True):
        if lock_path is None:
            lock_path = os.path.join(os.path.abspath(os.path.dirname(sys.argv[0])), "lock")
        self.lock_path = lock_path
        self.fh = None
        self.is_running = False
        try:
            self.do_magic()
        except:
            if raise_:
                raise
            else:
                self.is_running = None

    def do_magic(self):
        if OS_WIN:
            try:
                if os.path.exists(self.lock_path):
                    os.unlink(self.lock_path)
                self.fh = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
            except EnvironmentError as err:
                if err.errno == 13:
                    self.is_running = True
                else:
                    raise

        else:
            try:
                self.fh = open(self.lock_path, 'w')
                fcntl.lockf(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except EnvironmentError as err:
                if self.fh is not None:
                    self.is_running = True
                else:
                    raise

    def clean_up(self):
        # this is not really needed
        try:
            if self.fh is not None:
                if OS_WIN:
                    os.close(self.fh)
                    os.unlink(self.lock_path)
                else:
                    fcntl.lockf(self.fh, fcntl.LOCK_UN)
                    self.fh.close() # ???
                    os.unlink(self.lock_path)
        except Exception as err:
            # logger.exception(err)
            raise # for debugging porpuses, do not raise it on production


if __name__ == "__main__":
    import time

    si = SingleInstance()
    try:
        if si.is_running:
            sys.exit("This app is already running!")
        time.sleep(20) # remove
        # do other stuff
    finally:
        si.clean_up()
