"""

    TkTester.py

    Copyright (c) 2017-2023, Masatsuyo Takahashi, KEK-PF

"""

import os
import re
from time                   import sleep
import threading
from pyautogui              import typewrite
from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
from OurMessageBox          import dialog_queue

DEBUG               = False
COUNTER_CYCLE       = 10000
SHORT_INTERVAL      = 0.1
LONG_INTERVAL       = 0.5
LOOP_MAXNUM_CYCLE   = 360   # i.e., waits for LONG_INTERVAL * LOOP_MAXNUM_CYCLE seconds
GET_OBJ_MAX_RETRY   = 60
ENTER               = [ 'enter' ]
ESC                 = [ 'esc' ]

def reply_ok():
    # print( current_box_goemetry() )
    typewrite( ENTER )

def reply_yes():
    typewrite( 'Y' )

def reply_no():
    typewrite( 'N' )

lock = threading.Lock()

def messagebox_shown():
    return not dialog_queue.empty()

counter = 0
return_dict     = {}
attr_index_re   = re.compile( r'^(\w+)\[(\d+)\](.*)$' )
rest_index_re   = re.compile( r'^\[(\d+)\]' )

class Attribute(object):
    def __init__( self, parent, name ):
        self.__parent__ = parent
        self.__key__    = name

    def __getattribute__(self, key):
        if key.find( '__' ) == 0:
            if DEBUG: print( '(1) key=', key )
            return object.__getattribute__( self, key )
        else:
            attr = '.'.join( [ self.__key__, key ] )
            if DEBUG: print( 'Attribute:', attr )
            return Attribute( self.__parent__, attr )

    def __getitem__( self, n ):
        attr = self.__key__ + ( '[%d]' % n )
        return Attribute( self.__parent__, attr )

    def __call__( self, *args, __wait__=True, **wk_args ):
        sleep( SHORT_INTERVAL )
        if DEBUG: print( 'Attribute.__call__:', self.__key__,  *args )
        global counter
        if counter >= COUNTER_CYCLE:
            counter = 0

        counter += 1

        app = self.__parent__.__app__
        key = self.__key__
        log = self.__parent__.__log__

        success = False

        def get_obj():

            for retry in range( LOOP_MAXNUM_CYCLE ):
                if DEBUG: print( '(1) retry=', retry )
                try:
                    # this lock is probably necessary
                    if DEBUG: print( '(1) before lock' )
                    with lock:
                        if DEBUG: print( '(1-1) after lock' )
                        obj = app
                        for attr in key.split( '.' ):
                            if DEBUG: print( 'attr=', attr )
                            m = attr_index_re.match( attr )
                            if m:
                                attr_   = m.group(1)
                                n       = int( m.group(2) )
                                obj_    = object.__getattribute__( obj, attr_ )
                                obj     = obj_[n]
                                rest_index  = m.group(3)
                                if rest_index is None:
                                    pass
                                else:
                                    obj_ref = [obj]
                                    def apply_index( mobj ):
                                        i = int( mobj.group(1) )
                                        obj_ref[0] = obj_ref[0][i]
                                        return ''
                                    while rest_index != '':
                                        if DEBUG: print( 'rest_index=', rest_index )
                                        rest_index = re.sub( rest_index_re, apply_index, rest_index )
                                    obj = obj_ref[0]
                            else:
                                obj = object.__getattribute__( obj, attr )
                    success = True
                    break
                except AttributeError as exc:
                    print( 'Attribute: exc=', exc )
                    self.exception = exc
                    sleep( LONG_INTERVAL )
                    continue

            if DEBUG: print( '(1-2) after lock: obj=', obj )

            return success, obj

        success, obj = get_obj()
        if not success:
            print( 'key=', key, 'retry=', retry )
            raise self.exception

        should_wait = __wait__ is True or callable(__wait__)

        def invoke_closure( obj, counter ):
            if DEBUG: print( 'invoke_closure: counter=', counter )
            retry_count = 0
            while True:
                try:
                    if should_wait:
                        ckey = counter
                    else:
                        return_dict[ counter ] = counter
                        ckey = -counter     # negative key values will remain until clearance
                                            # without interfering positive key referencing
                    # ret = obj( *args )
                    ret = obj( *args, **wk_args )
                    return_dict[ ckey ] = ret
                    if DEBUG:
                        print( '---------- invoke_closure success', retry_count)
                    break
                except Exception as exc:
                    # one of the cases that cause this exception is a too long delay in
                    # DLL loading under, e.g, the real-time scan by anti-virus software.
                    print( exc )
                    if log is not None:
                        log( str(exc) + '\n' )
                    n = str(exc).find('invalid command name')
                    # n = str(exc).find('unknown')
                    print( 'ERROR at ' + key, n )
                    if n >= 0:
                        app.update()
                        sleep(1)
                        retry_count += 1
                        if DEBUG:
                            print("---------- retry_count=", retry_count)
                        if retry_count < GET_OBJ_MAX_RETRY:
                            success, obj = get_obj()
                            continue
                        else:
                            raise exc
                    else:
                        raise exc

        # not sure whethrer this lock is necessary
        if DEBUG: print( '(2) before lock' )
        with lock:
            if DEBUG: print( '(2-1) after lock' )
            app.after( 100, lambda c=counter: invoke_closure( obj, c ) )
        if DEBUG: print( '(2-2) after lock' )

        sleep( 0.2 )    # to probably succeed at the first try

        # note that "not __wait" cases wait for the invoke_closure call
        ret = None
        for retry in range( LOOP_MAXNUM_CYCLE ):
            if DEBUG: print( '(2) retry=', retry )
            try:
                ret = return_dict[ counter ]
                del return_dict[ counter ]
                break
            except KeyError:
                if DEBUG: print( 'retry=', retry, 'counter=', counter )
                try:
                    app.state()
                except Exception as exc:
                    print( exc )
                    break

                if callable( __wait__ ):
                    if not __wait__( retry ):
                        break

                sleep( LONG_INTERVAL )  # long or short?
                continue

        if DEBUG: print( '__call__: return' )
        return ret

class AppAgent(object):
    def __init__( self, app, log_func=None ):
        self.__app__    = app
        self.__log__    = log_func

    def __getattribute__( self, key ):
        if key.find( '__' ) == 0:
            if DEBUG: print( '(2) key=', key )
            return object.__getattribute__( self, key )
        else:
            print( 'AppAgent:',  key )
            return Attribute( self, key )

    def __is_alive__( self ):
        ret = False
        try:
            self.__app__.state()
            ret = True
        except Exception as exc:
            print( exc )
            pass
        return ret

class TestClient:
    def __init__( self, app, script, log_func=None ):
        self.agent = AppAgent( app, log_func=log_func )
        self.thread = threading.Thread(
                        target=script,
                        name='TestClientThread',
                        args=[ self, self.agent ]
                        )
        self.thread.start()

    def quit(self):
        self.thread.join()

    def __del__(self):
        # seems to be called in the current self.thread
        print('__del__')
