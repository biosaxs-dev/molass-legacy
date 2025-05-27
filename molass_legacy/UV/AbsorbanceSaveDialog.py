# coding: utf-8
"""
    AbsorbanceSaveDialog.py

    Copyright (c) 2018-2019, SAXS Team, KEK-PF
"""
import os
import numpy                as np
from bisect                 import bisect_right
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog
from molass_legacy.KekLib.BasicUtils             import get_caller_module
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy._MOLASS.SerialSettings         import get_setting
from molass_legacy.KekLib.TkUtils                import split_geometry
from molass_legacy.KekLib.TkCustomWidgets        import FileEntry
from molass_legacy.KekLib.NumpyUtils             import np_savetxt

class AbsorbanceSaveDialog( Dialog ):
    def __init__( self, parent ):
        self.parent = parent
        self.caller_module = get_caller_module( level=2 )

    def show( self ):
        title = "Absorbance Data Save"
        Dialog.__init__( self, self.parent, title, auto_geometry=False, geometry_cb=self.adjust_geometry )

    def adjust_geometry( self ):
        w, h, x, y = split_geometry( self.parent.geometry() )
        self.geometry("+%d+%d" % (self.parent.winfo_rootx() + w//2,
                                  self.parent.winfo_rooty() + int(h*0.7) ))

    def body( self, body_frame ):   # overrides parent class method
        tk_set_icon_portable( self, module=self.caller_module )

        label_frame = Tk.Frame(body_frame)
        label_frame.pack( padx=50, pady=10 )

        guide = Tk.Label(label_frame, text='Select what and where (file name) to save and press "OK"')
        guide.pack()

        detail_frame = Tk.Frame(body_frame)
        detail_frame.pack( padx=50, pady=10 )

        texts = [ "Save Data", "Save Figure" ]
        files = [ "UV_data.txt", "UV_figure.png" ]
        out_dir = get_setting( 'analysis_folder' )
        cb_vars = []
        file_vars = []
        row = 0
        for i, t in enumerate(texts):
            cb_var = Tk.IntVar()
            cb_vars.append( cb_var )
            cb = Tk.Checkbutton( detail_frame, text=t,
                                    variable=cb_var )
            cb.grid( row=row, column=0, sticky=Tk.W )
            file = Tk.StringVar()
            path = os.path.join( out_dir , files[i] ).replace( '\\', '/' )
            file.set( path )
            file_vars.append( file )
            file_entry = FileEntry( detail_frame, textvariable=file, width=60 )
            file_entry.grid( row=row, column=1, sticky=Tk.W, padx=10 )
            row += 1
            if i == 0:
                restrict_frame = Tk.Frame( detail_frame )
                restrict_frame.grid( row=row, column=1 )
                row += 1

                restrict_var = Tk.IntVar()
                cb = Tk.Checkbutton( restrict_frame, text="restrict data so that wavelength(λ) is within", variable=restrict_var )
                cb.grid( row=0, column=0, sticky=Tk.W )

                range_vars = []
                col = 1
                for k in range(2):
                    var = Tk.DoubleVar()
                    var.set( int( self.parent.wl_vector[ 0  if k==0 else -1] ) )
                    range_vars.append( var )
                    entry = Tk.Entry( restrict_frame, textvariable=var, width=6, justify=Tk.CENTER )
                    entry.grid( row=0, column=col, padx=5 )
                    if k == 0:
                        col += 1
                        label = Tk.Label( restrict_frame, text="and" )
                        label.grid( row=0, column=col )
                    col += 1

                space = Tk.Frame( detail_frame, height=10 )
                space.grid( row=row, column=0 )
                row += 1

        self.cb_vars    = cb_vars
        self.file_vars  = file_vars
        self.restrict_var   = restrict_var
        self.range_vars = range_vars

    def validate( self ):
        if self.restrict_var.get() == 1:
            f = self.range_vars[0].get()
            t = self.range_vars[1].get()
            range_ = np.where( self.parent.wl_vector >= f and self.parent.wl_vector <= t )[0]
            slice_ = slice(range_[0], range_[-1]+1)
        else:
            slice_ = slice(None, None)

        for k, cb_var in enumerate(self.cb_vars):
            if cb_var.get() == 0:
                continue

            file = self.file_vars[k].get()
            if k == 0:
                wl_vector = self.parent.wl_vector[slice_]
                wl_vector.reshape( ( 1, len(wl_vector )) )
                absorbance  = self.parent.absorbance
                corrected_data  = absorbance.get_corrected_data()
                col_header      = absorbance.col_header

                data    = corrected_data[slice_,:].T

                if False:
                    from mpl_toolkits.mplot3d   import Axes3D
                    import molass_legacy.KekLib.DebugPlot as dplt

                    dplt.push()
                    i = np.arange(data.shape[1])
                    y = np.arange(data.shape[0])
                    ii, yy = np.meshgrid(i, y)
                    zz = data[yy,ii]
                    w = self.parent.wl_vector[slice_]
                    xx, yy = np.meshgrid(w, y)
                    fig = dplt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    ax.plot_surface(xx, yy, zz, alpha=0.3)
                    fig.tight_layout()
                    dplt.show()
                    dplt.pop()

                data_   = np.vstack( [ wl_vector, data ] ).T
                col_header = np.array( [ '' ] + absorbance.col_header )
                # print( 'len(col_header)=', len(col_header) )
                # print( 'data_.shape=', data_.shape )

                with open(file, 'w', encoding='cp932') as fh:
                    fh.write('>>>>>>>>>>>>>> 連続計測（領域波長）Data Start<<<<<<<<<<<<\n')
                    delimter = ',' if file.lower().find( '.csv' ) > 0 else '\t'
                    fh.write( delimter.join(col_header) + '\n' )

                np_savetxt(file, data_, mode='a')

                with open(file, 'a', encoding='cp932') as fh:
                    fh.write('>>>>>>>>>>>>>> Data End <<<<<<<<<<<<\n')

            else:
                self.parent.fig.savefig( file )

        return 1
