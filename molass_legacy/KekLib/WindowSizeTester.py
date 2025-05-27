# coding: utf-8
"""

    WindowSizeTester.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""
import sys
import os
import time
from pyautogui              import screenshot
from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker
from molass_legacy.KekLib.BasicUtils             import mkdirs_with_retry, clear_dirs_with_retry
from molass_legacy.KekLib.OurTkinter             import Tk, Dialog, Font
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
from molass_legacy.KekLib.TkUtils                import adjusted_geometry, split_geometry, join_geometry, get_max_monitor
import OurMessageBox        as MessageBox
from ChangeableLogger       import Logger, arg_join

class TestDialog( Dialog ):
    def __init__( self, parent, title, width, height ):
        self.grab = 'local'     # used in grab_set
        self.parent     = parent
        self.title_     = title
        self.width      = width
        self.height     = height
        self.fixed_font = Font.Font( family="Courier", size=32 )

    def show( self ):
        Dialog.__init__( self, self.parent, self.title_, auto_geometry=False ) # this calls body

    def body( self, body_frame ):
        body_frame.pack( fill=Tk.BOTH, expand=Tk.Y )

        frame = Tk.Frame( body_frame, width=self.width, height=self.height )
        frame.pack( fill=Tk.BOTH, expand=Tk.Y )

        if self.parent.auto:
            escape_label = Tk.Label( frame, text='Press <escape> to stop' )
            escape_label.grid( row=1, column=1 )

        label_nw = Tk.Label( frame, text='NW', font=self.fixed_font )
        label_nw.grid( row=0, column=0, sticky=Tk.NW )
        label_ne = Tk.Label( frame, text='NE', font=self.fixed_font )
        label_ne.grid( row=0, column=2, sticky=Tk.NE )
        label_sw = Tk.Label( frame, text='SW', font=self.fixed_font )
        label_sw.grid( row=2, column=0, sticky=Tk.SW )
        label_se = Tk.Label( frame, text='SE', font=self.fixed_font )
        label_se.grid( row=2, column=2, sticky=Tk.SE )

        for x in range(3):
          Tk.Grid.columnconfigure(frame, x, weight=1)
        for y in range(3):
          Tk.Grid.rowconfigure(frame, y, weight=1)

        # W, H, X, Y = split_geometry( self.parent.geometry() )
        self.x = self.parent.X + self.parent.W // 2 - self.width // 2
        self.y = self.parent.Y + self.parent.H // 2 - self.height // 2
        geom = join_geometry( self.width, self.height, self.x, self.y )
        print( 'geom=', geom )
        self.geometry( geom )

    def buttonbox(self):
        '''add standard button box.

        override if you do not want the standard buttons
        '''

        box = Tk.Frame(self)

        w = Tk.Button(box, text="Shot", width=10, command=self.shot, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Cancel", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

        box.pack()

    def shot( self ):
        file = os.path.join( self.parent.img_folder, self.title_ + '.png' )
        # screenshot( file, region=( self.x, self.y, self.width, self.height) )
        screenshot( file )


class WindowSizeTester( Tk.Toplevel ):
    def __init__( self, parent ):
        Tk.Toplevel.__init__( self, parent )

        parent.report_callback_exception = self.report_callback_exception
        self.parent = parent
        self.auto   = False

        wrk_folder = __file__
        for i in range(3):
            wrk_folder = os.path.dirname( wrk_folder )
        log_folder= wrk_folder + r'\log'
        if not os.path.exists( log_folder ):
            mkdirs_with_retry( log_folder )

        log_file = log_folder + r'\test.log'
        if os.path.exists( log_file ):
            os.remove( log_file )

        self.app_logger = Logger( log_file )
        img_folder = wrk_folder + r'\img'
        clear_dirs_with_retry( [ img_folder ] )
        self.img_folder = img_folder

        self.withdraw()
        self.build_window()
        self.update()

        max_monitor = get_max_monitor()
        print( 'max_monitor=', max_monitor )
        W, H, X, Y = max_monitor.width, max_monitor.height, max_monitor.x, max_monitor.y
        self.geometry( join_geometry( 920, 650, X + 100, Y + 100 ) )
        self.W  = W     # 1920
        self.H  = H     # 1080
        self.X  = X
        self.Y  = Y

        self.W_min  = 240
        self.W_inc  = ( self.W - self.W_min ) / 9
        self.H_min  = 180
        self.H_inc  = ( self.H - self.H_min ) / 9

        # self.resizable(width=False, height=False)
        self.deiconify()
        self.bind( '<Escape>', self._on_escape )

    def report_callback_exception(self, exc, val, tb):
        # This method is to override the Tk method to be able
        # to report to the spawned console in windows application mode.
        et = ExceptionTracebacker()
        msg = 'Overridden Tk report method: ' + str( et )
        if self.app_logger is None:
            print( msg )
        else:
            self.app_logger.error( msg )

    def build_window( self ):
        global radio_button_text_length

        tk_set_icon_portable( self )

        label = Tk.Label( self, text="This is a window size tester." )
        label.pack( pady=30 )
        test_frame = Tk.Frame( self )
        # test_frame.pack( anchor=Tk.CENTER, fill=Tk.BOTH, expand=1 )
        test_frame.pack()

        btn_matrix = []
        for i in range(10):
            btn_row = []
            for j in range(10):
                btn = Tk.Button( test_frame, text='%d,%d' % (i, j), command=lambda i_=i, j_=j: self.test_dialog( j_, i_ ) )
                btn.grid( row=i, column=j, padx=10, pady=10 )
                btn_row.append( btn )
            btn_matrix.append( btn_row )
        self.btn_matrix = btn_matrix

        auto_frame = Tk.Frame( self )
        auto_frame.pack()
        btn = Tk.Button( auto_frame, text='Auto Test', command=self.auto_test )
        btn.pack( pady=20 )

        self.protocol( "WM_DELETE_WINDOW", self.quit )

    def quit( self ):
        self.parent.quit()

    def test_dialog( self, i, j ):
        print( i, j )
        w = self.W_min + int( i * self.W_inc )
        h = self.H_min + int( j * self.H_inc )
        title = '%dx%d' % ( w, h )
        self.dialog = TestDialog( self, title, w, h )
        self.dialog.show()

    def auto_test( self ):
        self.auto = True
        for i in range(10):
            for j in range(10):
                if not self.auto:
                    break
                self.update()
                self.after(  500, lambda: self.dialog.shot() )
                self.after( 1000, lambda: self.dialog.ok() )
                self.btn_matrix[i][j].invoke()
                time.sleep( 1 )

        self.auto = False
        MessageBox.showinfo( "Notify", "Auto Test has been finished.", parent=self )

    def _on_escape( self, *args ):
        self.update()
        yn = MessageBox.askyesno( "Confirmation", "Terminate testing?", parent=self )
        if yn:
            self.auto = False
            # self.dialog.destroy()
            # self.after( 1000, lambda: self.quit() )
