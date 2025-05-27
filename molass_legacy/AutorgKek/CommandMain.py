"""
    CommandMain.py

    Copyright (c) 2016-2018, Masatsuyo Takahashi, KEK-PF
"""
import sys
import warnings
import traceback
import numpy                as np
from CommandLineOptions     import parser
# from KekToolsGP             import AutorgKek
# import KekLib, molass_legacyGuinierAnalyzer, SerialAnalyzer, Decomposer
from molass_legacy.GuinierAnalyzer.AutorgKekAdapter import AutorgKekAdapter as AutorgKek
from ResultPrinter          import ResultPrinter

DEBUG = False

def autorg_func( in_file, out_file, ASTSAS, robust ):
    if DEBUG: print( 'autorg_func: ', ASTSAS, robust )

    command_ = CommandMain()
    atsas_  = ASTSAS == 1
    robust_ = robust == 1

    try:
        result = command_.execute( in_file, out_file, ATSAS=atsas_, robust=robust_ )
    except Exception as e:
        if True:
            ( exc_type_type_, exc_value, exc_traceback ) = sys.exc_info()
            e_seq = traceback.format_exception( exc_type_type_, exc_value, exc_traceback )
            print( ''.join( e_seq ) )
        print( e )
        return -1

    if DEBUG: print( 'autorg_func: before return' )

    if robust_:
        return 0
    else:
        return -1 if result is None or result.Rg is None else 0

class CommandMain:
    def __init__( self ):
        self.opts = None
        warnings.filterwarnings("ignore")

    def run( self ):
        if self.opts is None:
            self.opts, self.args = parser.parse_args()

        if self.opts.version:
            from molass_legacy.AutorgKek.AppVersion import get_com_version_string
            print( get_com_version_string() )
            sys.exit()

        if self.opts.server:
            self.run_as_a_server()
            sys.exit()

        if len( self.args ) == 0:
            parser.print_help()
            sys.exit()

        if self.opts.output is None:
            self.fh = sys.stdout
        else:
            self.fh = open( self.opts.output, "w" )

        self.csv_output = ( self.opts.format == 'csv' or self.opts.output.find( '.csv' ) > 0 )

        self.separator = ',' if self.csv_output else '\t'
        self.ATSAS  = self.opts.ATSAS
        self.format = self.opts.format

        outfile = self.opts.output
        printer = ResultPrinter(
            fh = self.fh,
            csv_output = outfile is not None and outfile.find( '.csv' ) > 0,
            ATSAS=self.ATSAS,
            )

        multiple = len( self.args ) > 1
        for i, file in enumerate( self.args ):
            # print( file )
            autorg_ = AutorgKek( file )

            try:
                result = autorg_.run( robust=self.opts.robust )
            except Exception as e:
                sys.exit( str(e) )

            if not self.opts.robust and ( result is None or result.Rg is None ):
                sys.exit( "Can't determine Rg." )

            printer.print_result( i, file, result, multiple=multiple )

        if self.fh != sys.stdout:
            self.fh.close()

    def execute( self, in_file, out_file, ATSAS=False, robust=False, optimize=True ):
        self.autorg = AutorgKek( in_file )
        result = self.autorg.run( robust=robust, optimize=optimize )

        if result is None: return result

        if robust or result.Rg is not None:
            fh = open( out_file, "w" )

            printer = ResultPrinter(
                fh = fh,
                csv_output = out_file.find( '.csv' ) > 0,
                ATSAS=ATSAS,
                )
            printer.print_result( 0, in_file, result )

            fh.close()

        return result

    def run_as_a_server( self ):
        from ClientServer   import AutorgKekServer
        server = AutorgKekServer( self )
        server.run()
