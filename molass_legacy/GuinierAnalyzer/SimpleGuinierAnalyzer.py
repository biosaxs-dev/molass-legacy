# coding: utf-8
"""
    SimpleGuinierAnalyzer.py

    Copyright (c) 2017-2018, Masatsuyo Takahashi, KEK-PF
"""
import re
import numpy                as np
import matplotlib.pyplot    as plt
import matplotlib.gridspec  as gridspec
from mpl_toolkits.mplot3d   import Axes3D
import seaborn
from CanvasDialog           import CanvasDialog

NUM_ZX_POINTS   = 5

doing_re            = re.compile( r'Doing (\S+)' )
guinier_start_re    = re.compile( r'^guinier start' )
guinier_result_re   = re.compile( r'^guinier result: type=(\S+), Rg=\((\S+),\s*\S+\), basic_quality=(\S+)' )
extrapolation_re    = re.compile( r'^extrapolation start for the (\d)-th peak ad\((\d)\) with range[\(,\[](\d+), (\d+), (\d+)[\),\]]' )
result_type_re      = re.compile( r'(.+)\((.+)\)' )

class SimpleGuinierAnalyzer:
    def __init__( self, tester_log ):
        self.result_dict = {}
        self.result_type_dict = {}
        self.renumd_type_dict = {}
        self.folder_no = -1
        self.read_log( tester_log )
        self.create_xyz()

    def result_type_to_chart_value( self, type_ ):
        m = result_type_re.match( type_ )
        if m:
            branch  = m.group(1)
            try_ret = m.group(2)

            if branch == 'None':
                v = -20
            else:
                digits = [ int(d) for d in branch.split( '-' ) ]
                lacking = 5 - len(digits)
                if lacking > 0:
                    digits += [0] * lacking

                v = 0
                for d in digits:
                    v = v*2 + d

                v *= 20
                if try_ret != 'None':
                    v += int(try_ret)

        else:
            assert False

        self.result_type_dict[v] = type_

        return v

    def read_log( self, tester_log ):

        self.folder_no = -1
        self.folder_paths = []
        self.folder_infos = []

        fh = open( tester_log )

        # for cases where guinier_start_re does not match first
        before_guinier_start = True

        self.max_row_no = None
        folder_peaks = None

        for line in fh:
            m = doing_re.match( line )
            if m:
                if folder_peaks is not None:
                    self.folder_infos.append( folder_peaks )

                folder = m.group(1)
                self.folder_no += 1
                print( self.folder_no, folder )
                self.folder_paths.append( folder )
                row_no  = -1
                folder_peaks = []
                continue

            m = guinier_start_re.match( line )
            if m:
                zx_type = None
                zx_count = None
                file_no = -1
                before_guinier_start = False

            if before_guinier_start:
                # ignore results before guinier_start.
                # this is normally considered unnecessary,
                # but it did happen once for 20180204/Open01, though not reproducibly
                continue

            m = guinier_result_re.match( line )
            if m:
                type_   = m.group(1)
                rg      = m.group(2)
                bq      = m.group(3)
                row_no += 1
                if self.max_row_no is None or row_no > self.max_row_no:
                    self.max_row_no = row_no

                if zx_type is None:
                    file_no += 1
                else:
                    zx_count += 1
                    if zx_type == 0:
                        if zx_count < NUM_ZX_POINTS:
                            file_no = None
                        else:
                            file_no = zx_start + zx_count
                    else:
                        num_remain = ( zx_end - zx_start ) - zx_count
                        if num_remain >= 0:
                            file_no = zx_start - zx_count
                        else:
                            file_no = None
                print( row_no, file_no, zx_count, type_, rg, bq )
                file_id = str(file_no) + '-' + str(zx_count)
                chart_value = self.result_type_to_chart_value( type_ )
                self.result_dict[ (row_no, self.folder_no) ] = [ file_id, chart_value, rg, bq ]
                continue

            m = extrapolation_re.match( line )
            if m:
                type_   = None
                peak    = int( m.group(1) )
                ad      = int( m.group(2) )
                left    = int( m.group(3) )
                middle  = int( m.group(4) )
                right   = int( m.group(5) )
                row_no += 1
                if ad == 0:
                    folder_peaks.append( ( left, middle, right ) )
                print( row_no, None, type_, peak, ad )
                zx_type = ad
                zx_count    = -1
                if ad == 0:
                    zx_start    = left
                    zx_end      = middle
                else:
                    zx_start    = middle
                    zx_end      = right

        if folder_peaks is not None:
            self.folder_infos.append( folder_peaks )

        fh.close()

        print( 'max_row_no=', self.max_row_no )
        print( 'folder_no=', self.folder_no )

        self.type_names = []
        for n, v in enumerate( sorted( self.result_type_dict.keys() ) ):
            type_name = self.result_type_dict[v]
            self.renumd_type_dict[v] = [ n, type_name ]
            self.type_names.append( type_name )

    def write_csv( self ):
        fh_all  = open( 'result_summary_all.csv', 'w' )
        fh_id   = open( 'result_summary_id.csv', 'w' )
        fh_type = open( 'result_summary_type.csv', 'w' )
        fh_rg   = open( 'result_summary_rg.csv', 'w' )
        fh_bq   = open( 'result_summary_bq.csv', 'w' )

        hf_list = [ fh_id, fh_type, fh_rg, fh_bq ]

        header = ','.join( [ 'row no' ] + [ 'folder-%03d' % n for n in range( self.folder_no+1 ) ] ) + '\n'
        fh_all.write( header )
        for fh in hf_list:
            fh.write( header )

        for row in range( self.max_row_no + 1 ):
            result_row = []
            for col in range( self.folder_no + 1 ):
                result = result_dict.get( (row, col) )
                if result is None:
                    result_row.append( str(result) )
                else:
                    result[1] = renumd_type_dict[ result[1] ][0]
                    result_row.append( ':'.join( [ str(s) for s in result ] ) )

            fh_all.write( ','.join( [ str(row) ] + result_row ) + '\n' )

            for n in range(len(hf_list)):
                fh = hf_list[n]
                fh.write( ','.join( [ str(row) ] + [ '' if result == 'None' else str(result.split(':')[n]) for result in result_row ] ) + '\n' )

        fh_all.close()
        for fh in hf_list:
            fh.close()

    def create_xyz( self ):
        x_list = []
        y_list = []
        z_list = []
        r_list = []

        for col in range( self.folder_no + 1 ):
            for row in range( self.max_row_no + 1 ):
                result = self.result_dict.get( (row, col) )
                if result is None:
                    continue

                type_val    = self.renumd_type_dict[ result[1] ][0]
                x_list.append( row )
                y_list.append( col )
                z_list.append( type_val )
                rg = result[2]
                r_list.append( 0 if rg =='None' else float(rg)  )

        self.x = np.array( x_list )
        self.y = np.array( y_list )
        self.z = np.array( z_list )
        self.r = np.array( r_list )

    def plot_results( self, fig ):
        print( 'plot_results' )
        # fig = plt.figure( figsize=(18, 9) )
        gs  = gridspec.GridSpec( 1, 1 )
        self.ax1 = ax1 = fig.add_subplot( gs[0,0], projection='3d' )
        self.plot_redraw( fig, set_lims=False )
        self.xlim = np.array( ax1.get_xlim() )  # deep copy
        self.ylim = np.array( ax1.get_ylim() )  # deep copy
        self.zlim = np.array( ( 0, 200 ) )
        ax1.set_xlim( self.xlim )
        ax1.set_ylim( self.ylim )
        ax1.set_zlim( self.zlim )

        fig.tight_layout()

    def plot_redraw( self, fig, draw_flags=None, set_lims=True ):
        ax1 = self.ax1
        ax1.cla()
        ax1.set_title( 'Rg and type of peak processing' )
        ax1.set_xlabel( 'File No' )
        ax1.set_ylabel( 'Folder No' )
        ax1.set_zlabel( 'Rg' )

        max_z = np.max( self.z )

        for n in range(max_z + 1):
            if draw_flags is not None:
                if draw_flags[n] == 0:
                    continue

            index = self.z == n
            ax1.scatter( self.x[index], self.y[index], self.r[index], s=3, label='type %d' % n )

        if set_lims:
            ax1.set_xlim( self.xlim )
            ax1.set_ylim( self.ylim )
            ax1.set_zlim( self.zlim )
        ax1.legend()

    def show_dialog( self, parent ):
        dialog = CanvasDialog( "Debug", parent=parent )
        dialog.show( self.plot_results, figsize=(18,9), toolbar=True )
