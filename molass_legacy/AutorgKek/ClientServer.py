# coding: utf-8
"""
    ClientServer.py

    Copyright (c) 2016-2019, Masatsuyo Takahashi, KEK-PF
"""

import os
import socket
import logging
import itertools
import time
import warnings
from threading              import Thread
from molass_legacy.KekLib.BasicUtils             import mkdirs_with_retry
# from KekToolsGP             import AutorgKek
from molass_legacy.AutorgKekAdapter       import AutorgKekAdapter as AutorgKek
from ResultPrinter          import ResultPrinter
from Settings               import get_setting

DEBUG = False

HOST        = 'localhost'
PORT        = 49999
BUFFER_SIZE = 1024
STOP        = b'--stop_server'
EXECUTE     = b'EXECUTE'
ASK_READY   = b'ASK_READY'
ACK         = b'ACK'
QVERSION    = b'--version'
SUCCESS     = b'SUCCESS'
FAILURE     = b'FAILURE'
MESSAGE     = b'MESSAGE'
SERIAL_ANALYZER = b'SERIAL_ANALYZER'
PROCESS_ARGS    = b'PROCESS_ARGS'
START_PROCESS   = b'START_PROCESS'
GET_NUM_STEPS   = b'GET_NUM_STEPS'
STOP_PROCESS    = b'STOP_PROCESS'
REQUEST_INFO    = b'REQUEST_INFO'
CANCEL_PROCESS  = b'CANCEL_PROCESS'
UNEXPECTED      = b'UNEXPECTED'

exe_options = {
    '--ATSAS'   : ( 'ATSAS', True ),
    '--atsas'   : ( 'ATSAS', True ),
    '--robust'  : ( 'robust', True ),
    }

class AutorgKekServer:
    def __init__( self, command_obj, host=HOST, port=PORT ):
        self.cmmd   = command_obj
        self.host   = host
        self.port   = port
        self.thread_array = []

        updir   = os.path.abspath( __file__ )
        for i in range(3):
            updir   = os.path.dirname( updir )
        logdir  = updir + '/log'
        mkdirs_with_retry( logdir )
        logfile = logdir + '/autorg_kek_server-log.csv'
        bakfile = logfile.replace( '.csv', '.bak' )

        if os.path.exists( logfile ):
            if os.path.exists( bakfile ):
                os.remove( bakfile )
            os.rename( logfile, bakfile )

        logging.basicConfig(
            filename=logfile, filemode='w',
            format='%(asctime)s,%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level=logging.DEBUG,
            )

        self.logger = logging.getLogger( __name__ )
        self.logger.info( 'Server started' )

    def get_port( self ):
        return self.port

    def run( self ):
        warnings.filterwarnings("ignore")

        self.socket = socket.socket( socket.AF_INET, socket.SOCK_STREAM )
        self.socket.bind( ( self.host, self.port ) )
        self.socket.listen( 1 )

        stop_server = False

        while not stop_server:
            conn, addr = self.socket.accept()
            if DEBUG: print( 'Connection address:', addr )

            args = []
            data = self.communicate( conn, args )
            if data == STOP:
                stop_server = True

            if data == SERIAL_ANALYZER:
                # conn will be closed in the thread
                self.start_serial_controller_thread( conn )

            else:
                while data != b'':
                    data = self.communicate( conn, args )
                    if data == STOP:
                        stop_server = True
                conn.close()

        self.logger.info( 'Server stopped' )

    def communicate( self, conn, args ):
        data = conn.recv( BUFFER_SIZE )
        if DEBUG: print( "server received data:", data )

        if data == b'':
            return data

        if data==ASK_READY:
            reply = ACK
        elif data == STOP:
            reply = ACK
        elif data == EXECUTE:
            reply = self.execute( args )
        elif data == QVERSION:
            from molass_legacy.AutorgKek.AppVersion import get_com_version_string
            reply = MESSAGE + bytes( ';' + get_com_version_string(), 'utf-8' )
        else:
            args.append( data.decode() )
            reply = ACK
        conn.send( reply )
        return data

    def execute( self, args ):
        # print( 'TO EXECUTE: args=', args )
        rq_args = args[0:2]
        self.log( rq_args )
        kw_args = {}
        for i, v in enumerate( args[2:] ):
            opt = exe_options.get( v )
            if opt is not None:
                kw_args[ opt[0] ] = opt[1]
        if DEBUG: print( 'TO EXECUTE: ', rq_args, kw_args )

        # ret = self.cmmd.execute( *rq_args, **kw_args )

        in_file, out_file = rq_args
        ATSAS       = False if len(args) <= 2 else args[2]
        robust      = True  if len(args) <= 3 else args[3]
        optimize    = True  if len(args) <= 4 else args[4]

        try:
            autorg = AutorgKek( in_file )
            result = autorg.run( robust=robust, optimize=True )
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            self.logger.info( str(etb) )
            result = None

        if result is None: return FAILURE

        if robust or result.Rg is not None:
            fh = open( out_file, "w" )

            printer = ResultPrinter(
                fh = fh,
                csv_output = out_file.find( '.csv' ) > 0,
                ATSAS=ATSAS,
                )
            printer.print_result( 0, in_file, result )

            fh.close()

        ret = result

        if ret is None:
            Rg = None
        else:
            Rg = ret.Rg
        self.log( rq_args + [ 'NA' if Rg is None else '%g' % Rg ] )

        reply = FAILURE if ret is None else SUCCESS

        return reply

    def log( self, *args ):
        message = ','.join( list( itertools.chain( *args ) ) )
        self.logger.info( message )

    def start_serial_controller_thread( self, conn ):
        thread = SerialControllerThread( conn )
        thread.start()
        self.logger.info( 'Serial Analyzer started' )
        self.thread_array.append( thread )

class SerialControllerThread(Thread):
    def __init__( self, conn ):
        Thread.__init__( self )
        self.conn = conn

    def receive( self, decode=False ):
        data = self.conn.recv( BUFFER_SIZE )
        if DEBUG: print( "SerialControllerThread received data:", data )
        self.conn.send( ACK )
        if decode:
            return data.decode()
        else:
            return data

    def run( self ):
        from SerialController           import SerialController
        warnings.filterwarnings("ignore")
        if DEBUG: print( "SerialControllerThread.run" )
        args = []

        while True:

            try:
                if DEBUG: print( 'SerialControllerThread: recv at loop top' )

                data = self.receive()

                if DEBUG: print( 'SerialControllerThread: data=', data )
                if data == b'' or data == REQUEST_INFO:
                    break

            except Exception as exc:
                print( exc )
                break

            if data == PROCESS_ARGS:
                args = []
                for i in range(3):
                    args.append( self.receive( decode=True ) )
                continue

            elif data == STOP_PROCESS:
                break

            elif data == START_PROCESS:
                if len(args) != 3:
                    print( 'len(args) != 3:' )
                    break

                data_folder = args[0]
                conc_folder = args[1]
                outp_folder = args[2]
                result_book = get_setting( 'result_book' )
                book_file   = os.path.join( outp_folder, result_book )
                serial_file = serial_file = os.path.join( outp_folder + '/.temp', '--serial_result.csv' )
                guinier_folder = outp_folder + '/.guinier_result'
                stamp_file     = os.path.join( self.guinier_folder, '--stamp.csv' )
                controller = SerialController( data_folder, conc_folder, outp_folder,
                                                guinier_folder=guinier_folder,
                                                stamp_file=stamp_file,
                                                parent=self, serial_file=serial_file, book_file=book_file )

                data = self.conn.recv( BUFFER_SIZE )
                print( 'data=', data )
                # should be GET_NUM_STEPS

                num_steps  = controller.start()
                self.conn.send( bytes( str(num_steps), 'utf-8' ) )

                for i in range( num_steps ):
                    try:
                        data = self.conn.recv( BUFFER_SIZE )
                        if DEBUG: print( 'data=', data )
                        if data == REQUEST_INFO:
                            info = controller.get_info( block=True )
                            if DEBUG: print( 'info=', info )
                            self.conn.send( bytes( str(i+1), 'utf-8' ) )
                        elif data == CANCEL_PROCESS:
                            controller.kill()
                            self.conn.send( ACK )
                            break
                    except Exception as exc:
                        print( exc )
                        break
            else:
                print( 'Unexpected data=', data )
                self.conn.send( UNEXPECTED )
                break

        self.conn.close()
        if DEBUG: print( 'SerialControllerThread: terminated' )
        # any other thing to do for closing?

class AutorgKekClient:
    def __init__( self, host=HOST, port=PORT ):
        self.host   = host
        self.port   = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect( ( self.host, self.port ) )

    def __del__( self ):
        self.socket.close()

    def request_impl( self, *args ):
        ret = None
        for message in args:
            if DEBUG: print( "client sending message:", message )
            self.socket.send( message )
            ret = self.socket.recv( BUFFER_SIZE )
            if DEBUG: print( "client received data:", ret )
        return ret

    def stop_server( self ):
        self.request_impl( STOP )

    def request( self, *args ):
        # print( 'request_str: args=', *args )
        b_args = list( map( lambda x: x.encode() if type(x) == str else str( x ).encode(), args ) )
        ret = self.request_impl( *b_args )
        retcode = 0 if ret == SUCCESS else -1
        return retcode
