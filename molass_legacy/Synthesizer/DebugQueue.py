# -*- coding: utf-8 -*-
"""

    ファイル名：   DebugQueue.py

    処理内容：

        テスト用の情報を伝達する。

        cf. http://stackoverflow.com/questions/7109093/checking-for-empty-queue-in-pythons-multiprocessing

"""
from __future__ import division, print_function, unicode_literals

import sys
import queue
import logging

debug_queue = queue.Queue()

def debug_queue_put( data ):
    debug_queue.put( data )

def debug_queue_get():
    try:
        return debug_queue.get( block = False )
    except queue.Empty:
        return None
    except:
        logging.exception( 'Unexpected error' )
        raise
