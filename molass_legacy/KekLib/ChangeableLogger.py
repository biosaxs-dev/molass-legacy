# coding: utf-8
"""

    ファイル名：   ChangeableLogger.py

    処理内容：

       cf.  How to change filehandle with Python logging on the fly with different classes and imports
            http://stackoverflow.com/questions/13839554/how-to-change-filehandle-with-python-logging-on-the-fly-with-different-classes-a

    Copyright (c) 2015-2020, Masatsuyo Takahashi, KEK-PF

"""
import os
import re
import logging
from io import StringIO

def arg_join( *args ):
    return ' '.join( [ str(x) for x in args ] )

class Logger:
    def __init__( self, path=None ):
        # ログの設定を行う。
        format_csv_ = '%(asctime)s,%(levelname)s,%(name)s,%(message)s'
        format_ssv_ = '%(asctime)s %(message)s'
        datefmt_    = '%Y-%m-%d %H:%M:%S'
        # logging.basicConfig( filename=path, format=format_csv_, datefmt=datefmt_ )

        if path is None:
            self.stream = StringIO()
            self.fileh = logging.StreamHandler(self.stream)
        else:
            self.stream = None
            self.fileh = logging.FileHandler( path, 'a' )
        self.formatter_csv_ = logging.Formatter( format_csv_, datefmt_ )
        self.fileh.setFormatter( self.formatter_csv_ )

        self.logger = logging.getLogger()
        self.logger.setLevel( logging.INFO )
        self.logger.addHandler( self.fileh )

        # コンソールへのログを追加する。
        self.ch = logging.StreamHandler()
        self.ch.setLevel( logging.DEBUG )
        self.formatter_ssv_ = logging.Formatter( format_ssv_, datefmt_ )
        self.ch.setFormatter( self.formatter_ssv_ )
        self.logger.addHandler( self.ch )

    def get_final_log_path( self ):
        this_dir = os.path.dirname( os.path.abspath( __file__ ) )
        log_path = os.path.abspath( this_dir + '/../../log/final_error.log' )
        n = 0
        while os.path.exists(log_path):
            n += 1
            log_path = re.sub(r'(-\d+)?\.log$', '-%02d.log'%n, log_path)
        return log_path

    def __del__( self ):
        # print( 'Logger.__del__' )
        if self.stream is not None:
            self.moveto( self.get_final_log_path() )
        self.logger.removeHandler( self.ch )
        self.logger.removeHandler( self.fileh )

    def changeto( self, path ):
        self.logger.removeHandler( self.fileh )
        self.fileh = logging.FileHandler( path, 'a' )
        self.fileh.setFormatter( self.formatter_csv_ )
        self.logger.addHandler( self.fileh )

    def moveto( self, path ):
        assert self.stream is not None
        fh = open(path, 'a')
        fh.write( self.stream.getvalue() )
        fh.close()
        self.stream = None

    def addHandler( self, handler ):
        self.logger.addHandler( handler )

    def removeHandler( self, handler ):
        self.logger.removeHandler( handler )

    def info( self, *args ):
        self.logger.info( *args )

    def warning( self, *args ):
        self.logger.warning( *args )

    def error( self, *args ):
        self.logger.error( *args )

    def debug( self, *args ):
        self.logger.debug( *args )

    def get_stream_buffer(self):
        return self.stream.getvalue()
