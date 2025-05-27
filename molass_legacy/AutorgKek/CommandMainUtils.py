
"""
    CommandMainUtils.py
"""
import sys
from subprocess import call
from .AtsasTools import autorg       as autorg_atsas
from .CommandMain import CommandMain
# import SerialAnalyzer
from molass_legacy.Reports.ReportUtils import make_record

sys_argv_init = sys.argv

def do_CommandMain( folder, file, out_file, subprocess=False, atsas=False, fh=None, run_dir=None, bico_mono_ratio=True ):
    path    = folder + '/' + file

    if subprocess:
        exe_name = 'autorg_kek.bat'
        atsas_arg = '-A' if atsas else ''
        ret = call( '%s/%s %s -o %s %s' % ( run_dir, exe_name, atsas_arg, out_file, path ), shell=True )
        # eq_( ret, 0 )
    else:
        sys.argv = sys_argv_init + [ '-o', out_file, path ]
        command = CommandMain()
        result = command.execute( path, out_file, robust=True, optimize=True )

        orig_result, eval_result    = autorg_atsas( path )

        rec = make_record( path, result, orig_result, eval_result )

        fh.write( rec + '\n' )
