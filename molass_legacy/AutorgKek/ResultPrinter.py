# coding: utf-8
"""
    ResultPrinter.py

    Copyright (c) 2016-2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy                as np

class ResultPrinter:
    def __init__( self, fh, csv_output, ATSAS=False ):
        self.fh     = fh
        self.format = None
        self.csv_output = csv_output
        self.separator = ',' if self.csv_output else '\t'
        self.ATSAS  = ATSAS

    def print_result( self, i, file, result, multiple=False ):
        file_ = file.replace( '/', '\\' )

        if multiple:
            self.print_result_multiple( i, file_, result )
        else:
            self.print_result_single( i, file_, result )

    def print_result_single( self, i, file, result ):
        if self.format is None and not self.csv_output:
            stdev_p = round( result.Rg_stdev / result.Rg * 100 )
            print( "Rg   = %.1f +/- %.2g (%d%%)" % ( result.Rg, result.Rg_stdev, stdev_p ) )
            print( "I(0) = %.3g +/- %.2g" % ( result.I0, result.I0_stdev ) )
            print( "Points %d to %d (%d total)" % ( result.From, result.To, result.To - result.From + 1 ) )
            print( "Quality %.2f" % result.Quality )
        else:
            self.print_result_multiple( i, file, result )

    def print_result_multiple( self, i, file, result ):
        if i == 0:
            self.print_header_line()
        self.print_detail_line( file, result )

    def print_header_line( self ):
        columns =  [ 'File',
                    'Rg', 'Rg StDev',
                    'I(0)', 'I(0) StDev',
                    'First point', 'Last point',
                    'Quality', 'Aggregated' ]

        if not self.ATSAS:
            columns += [ 'Min Q', 'Max Q', 'Remarks' ]
            # columns += [ 'Mono G', 'Mono Rg', 'Mono d' ]
            # columns += [ 'QF basic quality', 'QF positive score', 'QF fit consistency', 'QF stdev score', 'QF qRg score' ]
            # columns += [ 'IpI flag', 'BiCo flag', 'b/m ratio' ]
            # columns += [ 'BiCo G1', 'BiCo G2', 'BiCo Rg1', 'BiCo Rg2', 'BiCo d1', 'BiCo d2' ]

        self.fh.write( self.separator.join( columns ) + '\n' )

    def print_detail_line( self, file, result ):

        aggregated_ = '0'

        str_vars = []
        for v in [result.Rg, result.Rg_stdev, result.I0, result.I0_stdev,]:
            if v is None or np.isnan( v ):
                s_ = 'NA'
            else:
                s_ = '%g' % v
            str_vars.append( s_ )
        for v in [ result.From, result.To, ]:
            str_vars.append( 'NA' if v is None else '%d' % v )

        quality_ = 'NA' if result.Quality is None else '%.2g' % result.Quality 

        self.fh.write( ','.join( [  file, *str_vars, quality_, aggregated_ ] ) )

        if not self.ATSAS:
            # 
            str_vars = map( lambda x: 'NA' if x  is None else '%g' % x, [ result.min_q, result.max_q ] )
            self.fh.write( ',%s,%s,"%s"' % ( *str_vars, 'Unused' ) )

        if False:
            # Mono
            fit = result.fit
            str_vars = map( lambda x: 'NA' if x  is None else '%g' % x, [ fit.I0, fit.Rg, fit.degree ] )
            self.fh.write( ',' + ','.join( str_vars ) )

            # QF
            qobj = result.quality_object
            if qobj is None:
                str_vars = [ 'NA' ] * 5
            else:
                str_vars = map( lambda x: 'NA' if x  is None else '%g' % x, [ qobj.basic_quality, qobj.positive_score, qobj.fit_consistency_pure, qobj.stdev_score, qobj.q_rg_score ] )
            self.fh.write( ',' + ','.join( str_vars ) )

            # flags
            str_vars = map( lambda x: 'NA' if x  is None else '%g' % x, [ result.IpI, result.bicomponent, result.bico_mono_ratio ] )
            self.fh.write( ',' + ','.join( str_vars ) )

            # BiCo
            bico_params = [ result.bico_G1, result.bico_G2, result.bico_Rg1, result.bico_Rg2, result.bico_d1, result.bico_d2 ]
            str_vars = map( lambda x: 'NA' if x  is None else '%g' % x, bico_params )
            self.fh.write( ',' + ','.join( str_vars ) )

        self.fh.write( '\n' )
