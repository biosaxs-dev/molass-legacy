# coding: utf-8
"""
    DecompDriver.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import os, sys
from time import time, sleep
from multiprocessing import current_process
import logging
import numpy as np
from molass_legacy.KekLib.BasicUtils import mkdirs_with_retry
from molass_legacy.KekLib.NumpyUtils import np_savetxt

class UserCancel(Exception): pass

class DecompLoghandler(logging.Handler):
    def __init__(self, out_folder):
        logging.Handler.__init__(self)
        self.fh = open(out_folder + '/decomp.log', 'w')
        self.formatter  = logging.Formatter('%(asctime)s %(message)s', '%H:%M:%S')

    def emit( self, record ):
        if record.levelno >= logging.INFO:
            try:
                log_message = self.formatter.format(record) + '\n'
                self.fh.write(log_message)
                self.fh.flush()
            except:
                # this error can occur when canceled
                pass

    def __del__(self):
        self.fh.close()

def run_decomp(job_id, out_folder, job_info, table, use_gpu):
    try:
        from .DecompProcessUtils import run_decomp_impl

        logger  = logging.getLogger()
        started = time()

        if sys.stdout is None:
            sys.stdout = open(os.devnull, 'w')

        last_rec = table[job_id]

        process = current_process()
        pid = process.pid
        file_info = last_rec[0]
        submitted = last_rec[7]

        cancel_flag = table[job_id][11]
        if cancel_flag:
            raise UserCancel('User Cancel')

        # should not make the out_folder when canceled before actual execution.
        if not os.path.exists(out_folder):
            mkdirs_with_retry(out_folder)

        logger.addHandler(DecompLoghandler(out_folder))
        logger.setLevel(logging.INFO)

        os.chdir(out_folder)

        q = job_info.q
        a = job_info.a
        e = job_info.e
        dmax = job_info.dmax

        file = file_info[1]
        np_savetxt(file, np.array([q, a, e]).T)

        table[job_id] = [file_info, pid, -1, *last_rec[3:7], submitted, started, None, 0, cancel_flag]

        def progress_cb(step, chi2, rg, volume):
            duration = time() - started
            cancel_flag = table[job_id][11]
            if cancel_flag:
                raise UserCancel('User Cancel')
            table[job_id] = [file_info, pid, 0, step, chi2, rg, volume, submitted, started, None, duration, 0]

        ret = run_decomp_impl(q, a, e, dmax, file, job_info.infile_name,
                                progress_cb=progress_cb, steps=None,    # steps=1000, etc. for debug
                                use_gpu=use_gpu,
                                )
        state = -3
    except UserCancel as exc:
        last_rec = table[job_id]
        ret = exc
        state = -5
        logger.info('Execution canceled by user.')
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
        etb = ExceptionTracebacker()
        print(etb)
        logger.error(str(etb))
        ret = etb
        last_rec = table[job_id]
        state = -4

    finished = time()
    duration = finished - started
    last_rec = table[job_id]
    table[job_id] = [file_info, None, state, *last_rec[3:9], finished, duration, 0]
    print('run_decomp', job_id, 'finish')
    return ret
