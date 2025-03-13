# coding: utf-8
"""

    ファイル名：   PilatusUtilsOldStyle.py

    処理内容：

       Pilatus 画像データ関連の共通処理

    Copyright (c) 2015-2020, 2025, SAXS Team, KEK-PF
"""
import os
import sys
import glob
import re
import pandas as pd
from DebugQueue             import debug_queue_get

separator_line  = re.compile( '^\s*$' )
exec_date_re    = re.compile( '^#Execution date:\s+(.+)$' )
prefix_re       = re.compile( '^#\s+Pilatus1 fileprefix:\s+(\w+)' )
num_cycles_re   = re.compile( '^#\s+Pilatus1 Number of cycles:\s+(\d+)' )
num_changes_re  = re.compile( '^#\s+Pilatus1 Detector position:\s+Change\s+(\d+)' )
position_re     = re.compile( '^#\s+Pilatus1 Detector position(\(original\)|\d+):\s+([+-]?\d+\.?\d*),([+-]?\d+\.?\d*)' )
wave_length_re  = re.compile( r'^#\s+Wavelength:\starget=(\d+\.\d+)' )
energy_re       = re.compile( r'^#\s+Energy:\s+target=(\d+)' )

image_ext_re = re.compile( '\.(tif|cbf)$', flags=re.I )

DEBUG = False

def get_ext( file ):
    m = image_ext_re.search( file )
    if m:
        return m.group( 1 )
    return None

def regex_glob( regex ):
    global globbed_extension_
    # return filter( lambda x: regex.search( x ), glob.glob( '*.*' ) )
    files = []
    for file in glob.glob( '*.*' ):
        m = regex.search( file )
        if m:
            globbed_extension_ = m.group( 1 )
            files.append( file )
    return files

def get_data_info( folder_info, log_file, mask_file,
                    adj_folder, syn_folder, pilatus_counter, counter_id,
                    log_file_path,
                    sample_complete=False, for_test_data=False,
                    logger=None ):

    in_folder = folder_info.path
    dict_info = get_dict_info( in_folder, adj_folder, syn_folder, pilatus_counter, counter_id )

    if debug_queue_get() == __name__ + '.die()': die()  # die(): undefined function

    prefix_info_list, text_dict = get_prefix_info_list( in_folder, log_file_path )
    prefix_dict = {}
    for info in prefix_info_list:
        # print(info[0])
        prefix_dict[info[0]] = info

    image_data_info = []

    for info in prefix_info_list:
        fkey_type_is_old = get_a_prefix_record_old_style( counter_id, dict_info, sample_complete, info, image_data_info )

        if not fkey_type_is_old:
            get_a_prefix_record_new_style( counter_id, dict_info, sample_complete, info, image_data_info )

    data_array = make_data_array( image_data_info, prefix_dict )

    if DEBUG:
        for i, rec in enumerate( prefix_info_list ):
            print( 'prefix_info_list[%d]=' % i, rec )
        if len(image_data_info) > 0:
            print( 'image_data_info[ 0 ]=', image_data_info[ 0 ] )
            print( 'image_data_info[ -1 ]=', image_data_info[ -1 ] )
        else:
            print( 'image_data_info is empty' )

    if for_test_data:
        return data_array, text_dict
    else:
        return log_file, mask_file, data_array, pilatus_counter

def get_prefix_info_list( in_folder, log_file_path ):
    prefix = None
    measurement_text = ''
    text_dict = {}
    exec_date = None

    fh = open( log_file_path )

    prefix_info_list = []

    for line in fh.readlines():
        # if DEBUG: print( line )

        if separator_line.match( line ):
            if prefix != '':
                text_dict[ prefix ] = measurement_text
            measurement_text = ''
            exec_date = None
        else:
            measurement_text += line

        me = exec_date_re.match( line )
        if me:
            exec_date = me.group(1)

        m0 = prefix_re.match( line )
        if m0:
            if prefix is not None:
                prefix_info_list.append( [ prefix, num_cycles, num_changes, pos_array, wave_lengths, energies, exec_date ] )

            prefix = m0.group( 1 )
            if prefix[-1] == '_':
                prefix = prefix[:-1]
            if DEBUG: print( 'prefix=', prefix )
            # pos_array = [ prefix ]

            pos_info_counter = 0
            num_cycles  = 0
            num_changes = 0
            pos_array = []
            wave_lengths = []
            energies = []
            continue

        m1 = num_cycles_re.match( line )
        if m1:
            num_cycles = int( m1.group( 1 ) )
            if DEBUG: print( 'num_cycles=', num_cycles )

        m2 = num_changes_re.match( line )
        if m2:
            # print line
            num_changes = int( m2.group( 1 ) )
            if DEBUG: print( 'num_changes=', num_changes )
            continue

        m3 = position_re.match( line )
        if m3:
            position = m3.group( 1 )
            x = m3.group( 2 )
            y = m3.group( 3 )
            if DEBUG: print( "\tposition_re.match: %s\t%s,%s" % (position, x, y) )
            pos_array.append( [ x, y ] )
            continue

        m4 = wave_length_re.match( line )
        if m4:
            wave_lengths.append( '_' + str( int( float( m4.group(1) ) * 1000 ) ) + 'A' )

        m5 = energy_re.match( line )
        if m5:
            energies.append( '_' + m5.group(1) + 'eV' )

    if prefix is not None:
        prefix_info_list.append( [ prefix, num_cycles, num_changes, pos_array, wave_lengths, energies, exec_date ] )

    if measurement_text != '':
        text_dict[ prefix ] = measurement_text

    fh.close()

    return prefix_info_list, text_dict

def get_dict_info( in_folder, adj_folder, syn_folder, pilatus_counter, counter_id ):
    orig_folder = os.getcwd()

    counter_dict = pilatus_counter.get_counter_dict( counter_id )
    if False:
        print( 'counter_id=', counter_id )
        for k, v in counter_dict.items():
            print( k, v )

    global globbed_extension_
    globbed_extension_ = None
    org_file_dict = {}
    if in_folder and os.path.exists( in_folder ):
        os.chdir( in_folder )
        for file in regex_glob( image_ext_re ):
            fkey = re.sub( r'_\d+\.\w+$', '', file, flags=re.I )
            # print( 'fkey=', fkey )
            org_file_dict[ fkey ] = file

    # if DEBUG: print( 'org_file_dict=', org_file_dict )
    # print( 'globbed_extension_=', globbed_extension_ )
    if globbed_extension_:
        restricted_image_ext_re = re.compile( '\.(' + globbed_extension_ + ')$' )
    else:
        restricted_image_ext_re = image_ext_re

    adj_file_dict = {}
    if adj_folder and os.path.exists( adj_folder ):
        os.chdir( adj_folder )
        for file in regex_glob( restricted_image_ext_re ):
            fkey = re.sub( r'_[^_]+\.\w+$', '', file, flags=re.I )
            adj_file_dict[ fkey ] = file

    # print( 'adj_file_dict=', adj_file_dict )

    syn_file_dict = {}
    if syn_folder and os.path.exists( syn_folder ):
        os.chdir( syn_folder )
        for file in regex_glob( restricted_image_ext_re ):
            fkey = re.sub( r'_[^_]+\.\w+$', '', file, flags=re.I )
            syn_file_dict[ fkey ] = file

    # print( 'syn_file_dict=', syn_file_dict )

    os.chdir( orig_folder )

    dict_info = [ counter_dict, org_file_dict, adj_file_dict, syn_file_dict ]
    return dict_info

def get_a_prefix_record_old_style( counter_id, dict_info, sample_complete, info, image_data_info ):
    old_result = True

    counter_dict, org_file_dict, adj_file_dict, syn_file_dict   = dict_info
    prefix, num_cycles, num_changes, pos_array, wave_lengths, energies, exec_date  = info

    if DEBUG:
        print( 'get_a_prefix_record_old_style' )
        print( 'prefix=', prefix, 'num_cycles=', num_cycles, 'num_changes=', num_changes, 'counter_id=', counter_id )

    info_array = []
    for i in range( num_changes ):
        fkey = '%s_%d' % ( prefix, i )
        if DEBUG: print( [i], 'fkey=', fkey )
        if i == 0:
            if num_cycles == 1:
                new_prefix_ = prefix
            else:
                new_prefix_ = '%s_%d' % ( prefix, i )
            new_fkey = '%s_d%d' % ( new_prefix_, i )

            orig_file_count = counter_dict.get( fkey, 0 )
            orig_file_count_new = counter_dict.get( new_fkey, 0 )
            if DEBUG:
                print( 'orig_file_count=', orig_file_count )
                print( 'orig_file_count_new=', orig_file_count_new )
            if orig_file_count == 0 or orig_file_count_new > 0:
                old_result = False
                if DEBUG: print( 'old_result is set to False' )
                break
            else:
                ratio = 1.0
        else:
            if counter_id == 'None':
                ratio = 1.0
            else:
                if orig_file_count > 0:
                    ratio = counter_dict.get( fkey ) / orig_file_count
                else:
                    ratio = None
        org_file = org_file_dict.get( fkey )
        syn_file = syn_file_dict.get( prefix )
        if not syn_file:
            # prefix_syn.tif がなければ prefix_{i}_syn.tif を設定する。
            syn_file = syn_file_dict.get( fkey )
        if org_file and syn_file:
            org_ext = get_ext( org_file )
            syn_ext = get_ext( syn_file )
            # TODO: avoid this failure
            assert( syn_ext == org_ext )

        info_array.append( [ org_file, pos_array[i], ratio, adj_file_dict.get( fkey ), syn_file ] )

    if len( info_array ):
        # print( 'sample_complete=', sample_complete, ', len( info_array )=', len( info_array ) )
        if not sample_complete or len( info_array ) == num_changes:
            # すなわち、 sample_complete == True の場合は num_changes に見たなければ
            # 採集しない。
            image_data_info.append( [ prefix, info_array ] )

    return old_result

def get_a_prefix_record_new_style( counter_id, dict_info, sample_complete, info, image_data_info ):
    counter_dict, org_file_dict, adj_file_dict, syn_file_dict   = dict_info
    prefix, num_cycles, num_changes, pos_array, wave_lengths, energies, exec_date  = info

    if DEBUG:
        print( 'fkey_type_is_new: prefix=', prefix, 'num_cycles=', num_cycles, 'counter_id=', counter_id )
        print( 'wave_lengths=', wave_lengths )
        print( 'energies=', energies )

    for suppress_limit in [ 2, 1 ]:
        # 命名規則としては suppress_limit == 2 であるが、
        # 20180221データのように、途中で止めた場合、suppress_limit == 1 での処理が必要。

        wave_length_list    = [ '' ] if len(wave_lengths) < suppress_limit else wave_lengths
        energy_list         = [ '' ] if len(energies) < suppress_limit else energies

        num_key_oks     = 0
        num_key_errors  = 0
        for wave_length in wave_length_list:
            for energy in energy_list:
                for j in range( num_cycles ):
                    if num_cycles == 1:
                        cycle = ''
                    else:
                        cycle = '_%d' % ( j )

                    info_array = []
                    for k in range( num_changes ):
                        new_prefix_ = prefix + wave_length + energy + cycle
                        fkey = '%s_d%d' % ( new_prefix_, k )
                        if DEBUG: print( [k], 'fkey=', fkey )
                        if k == 0:
                            orig_file_count = counter_dict.get( fkey, 0 )
                            if orig_file_count == 0:
                                if DEBUG: print( 'no count for fkey=', fkey )
                                num_key_errors += 1
                                break
                            else:
                                fkey_type_is_new = True
                                ratio = 1.0
                                num_key_oks += 1
                        else:
                            if counter_id == 'None':
                                ratio = 1.0
                            else:
                                if orig_file_count > 0:
                                    changed_file_count = counter_dict.get( fkey, 0 )
                                    if changed_file_count > 0:
                                        ratio = counter_dict.get( fkey ) / orig_file_count
                                        # print( 'ratio=', ratio )
                                    else:
                                        # stopped?
                                        continue
                                else:
                                    assert( False )
                        syn_file = syn_file_dict.get( new_prefix_ )
                        if not syn_file:
                            # TODO: 必要？
                            pass

                        info_array.append( [ org_file_dict.get( fkey ), pos_array[k], ratio, adj_file_dict.get( fkey ), syn_file ] )

                    if len( info_array ):
                        if not sample_complete or len( info_array ) == num_changes:
                            # すなわち、 sample_complete == True の場合は num_changes に満たなければ
                            # 採集しない。
                            image_data_info.append( [ new_prefix_, info_array, prefix ] )
                            # [ new_prefix_, info_array, prefix ] will be interpreted
                            # [ prefix, info_array, dict_key ] later in make_data_array below

        if num_key_oks > 0:
            break

def make_data_array( image_data_info, prefix_dict ):
    data_array = []

    for info in image_data_info:
        if len(info) == 2:
            prefix, info_rec = info
            dict_key = prefix
        elif len(info) == 3:
            prefix, info_rec, dict_key = info
        else:
            assert False
        # print('dict_key=', dict_key)

        is_lacking = False
        for rec in info_rec:
            if rec[0] is None:
                is_lacking = True

        if is_lacking:
            # いずれかの画像がないデータは無視する。
            continue

        info = prefix_dict.get(dict_key)
        num_changes = info[2]
        # print( prefix, info_rec )
        data_array.append( [ prefix, info_rec, num_changes] )
        # self.data_array は、表示された Table オブジェクトのセル（表示値）と
        # 単純な対応関係を持つように構築するが、
        # その平坦な構造上、prefix ごとのグルーピングが失われるため、
        # i によってそのグルーピングを把握できるようにしておく。
        """
                [ AgBh002,      [ AgBh002_0_00000.tif,  ['0.70000','-0.35000'], 1.0, None, None ],
                                [ AgBh002_1_00000.tif,  ['5', '3'],             1.0, AgBh002_1_adj.tif, AgBh002_1_syn.tif ],
                                [ AgBh002_2_00000.tif,  ['-5', '-3'],           1.0, AgBh002_2_adj.tif, AgBh002_2_syn.tif ],
                ]
        """

    return data_array
