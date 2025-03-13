# -*- coding: utf-8 -*-
"""

    ファイル名：   TestDataGenerator.py

    処理内容：

        テスト用の模擬データを生成する。

"""
from __future__ import division, print_function, unicode_literals

import sys
import os
import shutil
import re
from nose.tools import eq_

import PilatusCounter
from PilatusUtils import get_data_info

import glob
import shutil

num_part_re     = re.compile( '^[^\d]+(\d+)' )
image_file_re   = re.compile( '^([^_]+)_' )
counter_file_re = re.compile( '^PilatusCounter_([^_]+)_' )

counter_id      = 'None'
pilatus_counter = None
adj_folder      = None
syn_folder      = None

class TestDataGenerator:
    def __init__( self, in_folder ):
        self.in_folder = in_folder

        curdir = os.getcwd()

        os.chdir( in_folder )

        logfiles = glob.glob( 'measurement*.log' )
        bakfiles = glob.glob( '*.log.bak' )

        assert( len( logfiles ) == 1 )

        self.logfile = logfiles[0]

        if len( bakfiles ) == 0:
            bakfile = logfiles[0].replace( '.log', '.log.bak' )
            shutil.copy2( logfiles[0], bakfile )

        os.chdir( curdir )

    def restore_from_bak( self ):
        curdir = os.getcwd()

        os.chdir( self.in_folder )

        bakfiles = glob.glob( '*.log.bak' )
        assert( len( bakfiles ) == 1 )
        logfile = bakfiles[0].replace( '.log.bak', '.log' )
        shutil.copy2( bakfiles[0], logfile )

        data_array, text_dict = get_data_info( self.in_folder, adj_folder, syn_folder, pilatus_counter, counter_id, for_test_data=True )

        remove_count = 0
        for file in glob.glob( '*.tif' ):
            # print( file )
            m = image_file_re.match( file )
            if m:
                sample_id = m.group( 1 )
                if not sample_id in text_dict:
                    remove_count += 1
                    os.remove( file )

        for file in glob.glob( 'PilatusCounter_*.txt' ):
            # print( file )
            m = counter_file_re.match( file )
            if m:
                sample_id = m.group( 1 )
                if not sample_id in text_dict:
                    remove_count += 1
                    os.remove( file )

        if remove_count > 0:
            print( 'restore_from_bak: Rmoved %d files.' % ( remove_count ) )

        os.chdir( curdir )

    def get_data_array( self ):

        data_array, text_dict = get_data_info( self.in_folder, adj_folder, syn_folder, pilatus_counter, counter_id, for_test_data=True )
        self.data_array = data_array
        self.text_dict  = text_dict
 
        return data_array

    def incremented_id( self, name ):
        m = num_part_re.match( name )
        if m:
            oldnum = m.group( 1 )
            len_ = len( oldnum )
            fmt = '%0{0}d'.format( len_ )
            newnum = fmt % ( int( oldnum ) + 1 )
            # print( oldnum, '=>', newnum  )
            return name.replace( oldnum, newnum )
        else:
            assert( False )

    def add_mockcopy( self, num_copies=1 ):

        curdir = os.getcwd()
        os.chdir( self.in_folder )

        for i in range( num_copies ):
            data_array = self.get_data_array()
            oldname = data_array[-1][0]
            newname = self.incremented_id( oldname )

            tiffiles = glob.glob( oldname + '*.tif' )
            for oldtif in tiffiles:
                newtif = oldtif.replace( oldname, newname )
                shutil.copy2( oldtif, newtif )

            counterfiles = glob.glob( 'PilatusCounter_' + oldname + '*.txt' )
            assert( len( counterfiles ) == 1 )
            new_file = counterfiles[0].replace( oldname, newname )
            fh = open( counterfiles[0] )
            content = fh.read()
            fh.close()
            fh = open( new_file, 'w' )
            fh.write( content.replace( oldname, newname ) )
            fh.close()

            logtext = self.text_dict[ oldname ]
            # print( 'logtext=', logtext )
            fh = open( self.logfile, 'a' )
            fh.write( "\n" )
            fh.write( logtext.replace( oldname, newname ) )
            fh.close()
            print( 'Mock %s has been created.' % (newname) )

        os.chdir( curdir )

if __name__ == '__main__':
    mdg = MockDataGenerator()
    mdg.generate( 1 )
    