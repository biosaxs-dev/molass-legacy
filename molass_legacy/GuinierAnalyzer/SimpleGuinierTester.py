# coding: utf-8
"""
    SimpleGuinierTester.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
import re
from time           import sleep
from molass_legacy.KekLib.OurTkinter     import Tk
from TkTester       import TestClient, reply_ok, reply_yes, reply_no, messagebox_shown
from DataUtils      import serial_folder_walk

class SimpleGuinierTester:
    def __init__( self, app ):
        self.app        = app

    def control_suite( self, root_folder, targets={} ):
        self.root_folder    = root_folder
        self.compiled_targets   = {}
        for k, v in targets.items():
            regex = re.compile( k )
            self.compiled_targets[ regex ] = v

        self.results    = []
        self.test_count = 0
        self.ok_count   = 0
        self.client = TestClient( self.app, self.tester_cb )

        sleep( 1 )

        print( 'control_suite done' )

    def tester_cb( self, client, agent ):
        self.client = client
        self.agent  = agent
        print( 'tester_cb: root_folder=', self.root_folder )

        serial_folder_walk( self.root_folder, self.do_a_folder )

        # agent.quit( immediately=True )    # does not work
        # self.app.quit( immediately=True ) # does not work

        self.app.parent.quit()              # does work
        print( 'test results' )
        for result in self.results:
            print( result )

        print( 'test_count=', self.test_count )
        if self.ok_count == self.test_count:
            print( 'All tests were successful.' )
        else:
            failures = self.test_count - self.ok_count
            print( 'There were %d failures.' %  failures )

    def get_expected_results( self, in_folder ):

        ret = None
        for k, v in self.compiled_targets.items():
            if k.search( in_folder ):       # changed from find to re.search
                ret = v
                break

        return ret

    def do_a_folder( self, in_folder, uv_folder, plot=None, suppress_retry_call=False, analysis_name=None ):

        expected = self.get_expected_results( in_folder )
        if expected is None:
            return True, None

        print( 'expected=', expected )

        agent   = self.agent

        print( 'in_folder=', in_folder )
        print( 'uv_folder=', uv_folder )
        agent.in_folder_entry.focus_force()
        print( 'in_folder_entry.focus_force' )
        agent.in_folder_entry.delete( 0, Tk.END)
        agent.in_folder_entry.insert( 0, in_folder )
        agent.on_entry_in_folder()

        if uv_folder != in_folder:
            agent.uv_folder_entry.focus_force()
            agent.uv_folder_entry.delete( 0, Tk.END)
            agent.uv_folder_entry.insert( 0, uv_folder )
            agent.on_entry_uv_folder()

        for k, v in sorted( expected.items() ):
            self.test_count += 1
            try:
                result = agent.do_simple_guinier( k )
                ok = abs( result.Rg - v )/v < 0.01
                if ok:
                    self.ok_count += 1
                rg = result.Rg
            except:
                ok = None
                rg = None

            self.results.append( [ in_folder, k, v, float('%.3g' % rg), ok ] )

        sleep( 1 )

        ret_continue = True
        return ret_continue, None