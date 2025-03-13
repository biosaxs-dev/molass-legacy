"""
    Test.TesterLogger.py

    Copyright (c) 2017-2022, SAXS Team, KEK-PF
"""

import os
import re
import queue
import threading
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry, open_w_safely

tester_log_queue = None

log_fh = None
dev_fh = None

def open_tester_log( home_dir ):
    global log_fh

    log_dir     = os.path.abspath( home_dir + '/log' ).replace( '\\', '/' )
    if not os.path.exists( log_dir ):
        mkdirs_with_retry(log_dir  )

    log_path = os.path.join(log_dir, "tester.log")
    log_fh = open_w_safely( log_path, "w", rename_cb=rename_log )
    create_log_queue()

    return log_fh

def rename_log( path ):
    n = 0
    path_bak = path
    while os.path.exists( path_bak ):
        path_bak = re.sub( r"(-\d*)?\.log$", "-%02d.log" % n, path )
        n += 1
    os.rename( path, path_bak ) 

def write_to_log( recstr ):
    global log_fh

    lock = threading.Lock()
    with lock:
        for fh in [ log_fh, dev_fh ]:
            if fh is None:
                continue
            fh.write( recstr )
            fh.flush()

def create_log_queue():
    global tester_log_queue
    if tester_log_queue is None:
        tester_log_queue = queue.Queue()

def put_to_tester_log_queue( recstr ):
    global tester_log_queue
    if tester_log_queue is None: return

    tester_log_queue.put( recstr )

def write_to_tester_log( recstr ):
    put_to_tester_log_queue( recstr )

def write_from_log_queue():
    global tester_log_queue

    if tester_log_queue is None:
        return

    lock = threading.Lock()
    with lock:
        while not tester_log_queue.empty():
            write_to_log( tester_log_queue.get() )

def open_dev_log( path ):
    global dev_fh
    dev_fh  = open( path, "w" )

def write_to_dev_log( recstr  ):
    global dev_fh
    if dev_fh is None: return

    dev_fh.write( recstr )
    dev_fh.flush()

def close_dev_log():
    global dev_fh
    if dev_fh is None: return
    dev_fh.close()
    dev_fh = None
