"""
    ExceptionTracebacker.py

    Copyright (c) 2016-2024, Masatsuyo Takahashi, KEK-PF
"""
import sys
import traceback
import logging

class ExceptionTracebacker:
    def __init__( self, call_stack=False ):
        ( exc_type_type_, exc_value, exc_traceback ) = sys.exc_info()
        self.c_seq = traceback.format_stack()[:-2] if call_stack else []
        self.e_seq = traceback.format_exception( exc_type_type_, exc_value, exc_traceback)[1:]

    def __repr__( self ):
        return ''.join( self.c_seq + self.e_seq )

    def log( self ):
        logger  = logging.getLogger( __name__ )
        logger.error( str(self) )

    def last_line( self ):
        return self.e_seq[-1][:-1]      # remove trailing '\n'

    def last_lines( self, n=2 ):
        return ''.join(self.e_seq[-n:])

def log_exception(logger, message, n=2):
    if logger is None:
        logger = logging.getLogger( __name__ )
    etb = ExceptionTracebacker()
    logger.warning(message + etb.last_lines(n=n))
    return etb

def warnlog_exception(logger, message, n=2):
    if logger is None:
        logger = logging.getLogger( __name__ )
    etb = ExceptionTracebacker()
    logger.warning(message + etb.last_lines(n=n).replace('Error', 'Warning'))
    return etb
