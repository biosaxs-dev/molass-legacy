"""
    ResultGui.py

    Copyright (c) 2016-2022, SAXS Team, KEK-PF
"""

import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy.KekLib.TkUtils import adjusted_geometry
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.AutorgKek.AtsasTools import autorg as autorg_atsas
# from KekToolsGP             import autorg   as autorg_kekpf
from molass_legacy.GuinierAnalyzer.AutorgKekAdapter import AutorgKekAdapter
from molass_legacy.SerialAnalyzer.ProofPlot import Plotter
from molass_legacy.SerialAnalyzer.ResultTable import ResultTable
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.TextShowDialog import TextShowDialog
from molass_legacy.KekLib.NumpyUtils import np_savetxt

class ResultGui( Tk.Toplevel ):
    def __init__( self, parent, folder, file, np_array=None, atsas_exe_index=0,
                    orig_info=None,
                    q_limit_index=None,
                    title_filename=None,
                    atsas_np_array=None ):
        Tk.Toplevel.__init__( self, parent )

        self.orig_info      = orig_info
        self.q_limit_index  = q_limit_index
        self.atsas_np_array = atsas_np_array

        self.withdraw()
        tk_set_icon_portable( self )

        self.oopt_fit_consistency   = get_setting( 'oopt_fit_consistency' )
        # self.oopt_qrg_limits        = get_setting( 'oopt_qrg_limits' )
        self.oopt_qrg_limits        = 'Limited'
        # self.oopt_qrg_limits_apply  = get_setting( 'oopt_qrg_limits_apply' )
        self.oopt_qrg_limits_apply  = 1
        # self.oopt_optimize_only     = get_setting( 'oopt_optimize_only' )
        self.oopt_optimize_only     = 0

        rec_array, intensity = self.run_autorg(  folder, file, np_array=np_array, atsas_exe_index=atsas_exe_index )
        self.intensity = intensity

        atsas_exe_paths = get_setting( 'atsas_exe_paths' )
        if len( atsas_exe_paths ) > 0:
            text_ = 'KEK autorg result of %s, compared with that of %s' % (
                        folder + '/' + file,
                        atsas_exe_paths[atsas_exe_index]
                        )

        else:
            text_ = 'KEK autorg result of %s' % ( folder + '/' + file )

        frame = Tk.Frame( self )
        frame.pack( fill=Tk.BOTH, expand=1, padx=10, pady=10 )

        frame1 = Tk.Frame( frame )
        frame1.pack( fill=Tk.BOTH, expand=1 )

        """
        if self.oopt_fit_consistency == 1:
            text_ += '; optimized'
            if self.oopt_qrg_limits=='Limited' and self.oopt_qrg_limits_apply:
                text_ += ' with q*Rg limits'
            if self.oopt_optimize_only == 1:
                text_ += '; optimized only'
        """

        label = Tk.Label( frame1, text=text_ )
        label.pack()

        frame2 = Tk.Frame( frame )
        frame2.pack( fill=Tk.BOTH, expand=1 )

        self.show_results( frame1, rec_array )

        if title_filename is None:
            title_filename = file
        self.show_proof_plot( frame2, title_filename, rec_array, intensity )

        # it seemd that '<Control-c>' should be bound from the top-level.
        self.bind( "<Control-c>", self.control_c )

        # print( 'ResultGui: geometry=', self.geometry() )
        self.update()
        self.geometry( adjusted_geometry( '1800x700' ) )

        if intensity.comments is not None:
            self.create_popup_menu()
            self.bind( '<Button-3>', self._on_right_click )

        self.protocol( "WM_DELETE_WINDOW", self.at_exit )
        self.deiconify()

    def create_popup_menu( self ):
        # create a popup menu
        self.menu = Tk.Menu( self, tearoff=0 )
        self.menu.add_command( label='Show Comments',  command=self.show_comments )

    def _on_right_click( self, event ):
        self.popup_position = ( event.x, event.y )
        # print( 'popup_position=', self.popup_position )

        self.menu.post( event.x_root, event.y_root )

    def show_comments( self ):
        dialog = TextShowDialog( self, "Intensity File Header",
                message=self.intensity.comments,
                width=93, height=53,
                )

    def run_autorg( self, folder, file, np_array=None, atsas_exe_index=0 ):
        path = folder + '/' + file

        optimize_           = self.oopt_fit_consistency==1

        if self.oopt_qrg_limits == 'Unlimited':
            qrg_limits_ = None
            qrg_limits_apply_   = False
        else:
            qrg_limits_ = get_setting( 'oopt_qrg_limits_vals' )
            qrg_limits_apply_   = self.oopt_qrg_limits_apply == 1

        # print( 'ResultGui.run_autorg: qrg_limits_=', qrg_limits_ )

        rec_array = []
        atsas_exe_paths = get_setting( 'atsas_exe_paths' )
        if len( atsas_exe_paths ) > 0 and atsas_exe_index is not None:
            if self.atsas_np_array is None:
                path_for_atsas  = path
            else:
                path_for_atsas  = get_setting( 'temp_folder' ) + '/' + file
                np_savetxt( path_for_atsas, self.atsas_np_array )
            rg_info_a, rg_info_a_   = autorg_atsas( path_for_atsas, exe_index=atsas_exe_index, smaxrg=qrg_limits_[1] )
            if self.atsas_np_array is not None:
                os.remove( path_for_atsas )
        else:
            rg_info_a, rg_info_a_   = None, None

        rec_array.append( rg_info_a )
        rec_array.append( rg_info_a_ )

        if np_array is None:
            path_for_autorg_kek = path
        else:
            path_for_autorg_kek = np_array

        """
        rg_info_k, intensity = autorg_kekpf( path_for_autorg_kek, robust=True,
                                optimize=optimize_,
                                qrg_limits=qrg_limits_,
                                qrg_limits_apply=qrg_limits_apply_ )
        """

        autorg_kek  = AutorgKekAdapter( path_for_autorg_kek )
        result_kek  = autorg_kek.run( robust=True, optimize=True, fit_result=True )

        rec_array.append( result_kek )
        intensity = autorg_kek.intensity

        return rec_array, intensity

    def show_proof_plot( self, frame, file, rec_array, intensity ):
        rg_info_a   = rec_array[1]
        rg_info_k   = rec_array[2]
        plotter = Plotter( intensity, file, rg_info_a, rg_info_k, orig_info=self.orig_info, q_limit_index=self.q_limit_index )
        plotter.show_prepare()

        self.mpl_canvas = FigureCanvasTkAgg( plotter.fig, frame )
        self.mpl_canvas.draw()
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack( fill=Tk.BOTH, expand=1 )

    def show_results( self, frame, rec_array ):
        self.result = ResultTable( frame, rec_array)

    def at_exit( self ):
        # print( 'at_exit' )
        self.destroy()

    def control_c( self, event ):
        w = self.focus_get()
        # print( 'table=', self.file_info_table )
        # print( 'w=', w )
        if str(w).find( str( self.result ) ) == 0:
            self.result.table.control_c( event )
        else:
            # TODO:  w.selection
            pass
