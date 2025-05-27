# coding: utf-8
"""

    ファイル名：   DeveloperOptions.py

    処理内容：

        開発者用の設定変更ダイアログ

    Copyright (c) 2017-2022, SAXS Team, KEK-PF

"""

import os
import platform
import wmi
import re
import warnings
import time

from molass_legacy.KekLib.OurTkinter             import Tk, Dialog, ttk, Font, FileDialog
from molass_legacy.KekLib.TkSupplements          import tk_set_icon_portable
import OurMessageBox        as MessageBox
from DevSettings            import get_dev_setting, set_dev_setting
from molass_legacy.KekLib.TkCustomWidgets        import FolderEntry
from molass_legacy.KekLib.BasicUtils             import get_home_folder
from molass_legacy._MOLASS.SerialSettings         import clear_settings, get_setting, set_setting
from MultiMonitor           import get_max_monitor, get_monitor_list, PYTHON_DEMO_MONITOR
from molass_legacy.KekLib.TkUtils                import adjusted_geometry

from molass_legacy.KekLib.ReadOnlyText           import ReadOnlyText
from MachineTypes           import is_on_virtual_machine, get_cpuid

ENABLE_HIRES_MODE_CHANGE    = False

class PlatformDialog(Dialog):
    def __init__(self, parent):
        Dialog.__init__(self, parent, "PlatformDialog" )

    def body( self, body_frame ):
        pinfo = ReadOnlyText(body_frame)
        pinfo.pack( fill=Tk.BOTH, expand=1, pady=10 )

        # Platform Info
        pinfo_list = [
            platform.machine(),
            platform.version(),
            str(platform.uname()),
            platform.system(),
            platform.processor(),
            ]

        for info in pinfo_list:
            pinfo.insert(Tk.END, info + '\n')

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

class ProcessorDialog(Dialog):
    def __init__(self, parent):
        Dialog.__init__(self, parent, "ProcessorDialog" )

    def body( self, body_frame ):
        pinfo = ReadOnlyText(body_frame)
        pinfo.pack( fill=Tk.BOTH, expand=1, pady=10 )

        # Win32_Processor Info
        pinfo_list = [ str(s) for s in wmi.WMI().Win32_Processor()]

        for info in pinfo_list:
            pinfo.insert(Tk.END, info + '\n')

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

class GpuDialog(Dialog):
    def __init__(self, parent):
        Dialog.__init__(self, parent, "GpuDialog" )

    def body( self, body_frame ):
        from Env.GpuInfo import DxdiagInfo

        ginfo = ReadOnlyText(body_frame)
        ginfo.pack( fill=Tk.BOTH, expand=1, pady=10 )

        if False:
            # DxdiagInfo
            # this takes some time
            dxinfo = DxdiagInfo()
            ginfo_list = dxinfo.get_info_list()

            for info in ginfo_list:
                ginfo.insert(Tk.END, info + '\n')

        # Win32_VideoController Info
        ginfo_list = [ str(s) for s in wmi.WMI().Win32_VideoController()]

        for info in ginfo_list:
            ginfo.insert(Tk.END, info + '\n')

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)


class DeveloperOptionsDialog( Dialog ):
    def __init__( self, parent, title ):
        self.grab = 'local'     # used in grab_set
        self.parent             = parent
        self.title_             = title
        self.applied            = False

    def show( self ):
        self.parent.config( cursor='wait' )
        self.parent.update()

        Dialog.__init__(self, self.parent, self.title_ )

        self.parent.config( cursor='' )

    def body( self, body_frame_ ):   # overrides parent class method

        body_frame = Tk.Frame( body_frame_ )
        body_frame.pack( padx=40, pady=10 )

        tk_set_icon_portable( self )

        clear_button = Tk.Button( body_frame, text="Clear Setting", command=self.clear_settings )
        clear_button.pack( anchor=Tk.NW )

        # Caveate
        caveate = ( "Please note that those options below are temporary and not saved as permanent setting.\n"
                    + "In other words, you have to specify everytime after restart."
                  )
        msg = Tk.Label( body_frame, text=caveate, bg='white', fg='red' )
        msg.pack( fill=Tk.BOTH, expand=1, pady=10 )

        platform_frame = Tk.Frame(body_frame)
        platform_frame.pack()

        platform_btn = Tk.Button(platform_frame, text='Platform Info', command=lambda: PlatformDialog(self))
        platform_btn.grid(row=0, column=0, padx=10)

        processor_btn = Tk.Button(platform_frame, text='Processor Info', command=lambda: ProcessorDialog(self))
        processor_btn.grid(row=0, column=1, padx=10)

        cpuid_label = Tk.Label(platform_frame, text='ProcessorId: ' + str(get_cpuid()))
        cpuid_label.grid(row=0, column=2, padx=10)

        vitual_label = Tk.Label(platform_frame, text='Virtual Machine: ' + str(is_on_virtual_machine()))
        vitual_label.grid(row=0, column=3, padx=10)

        gpu_btn = Tk.Button(platform_frame, text='Gpu Info', command=lambda: GpuDialog(self))
        gpu_btn.grid(row=1, column=0, padx=10, pady=5, sticky=Tk.W)

        base_frame = Tk.Frame( body_frame );
        base_frame.pack( expand=1, fill=Tk.BOTH, padx=10, pady=10 )

        grid_row = -1

        # Display Selection
        grid_row += 1
        display_select_frame = Tk.Frame( base_frame )
        display_select_frame.grid( row=grid_row, column=0, sticky=Tk.W )

        display_select_label = Tk.Label( display_select_frame, text="Display Selection: " )
        display_select_label.grid( row=0, column=0, sticky=Tk.W )
        max_monitor = get_max_monitor()
        monitors = get_monitor_list()
        # print( 'monitors=', monitors )
        self.monitor = Tk.StringVar()
        env_monitor = os.environ.get( PYTHON_DEMO_MONITOR )
        if env_monitor is not None:
            self.monitor.set( env_monitor )
        for k, m in enumerate(monitors):
            m_ = str(m)
            if env_monitor is None and m is max_monitor:
                self.monitor.set( str(k) )
            rb = Tk.Radiobutton( display_select_frame, text=m_, variable=self.monitor, value=k )
            rb.grid( row=k+1, column=1, sticky=Tk.W )

        if ENABLE_HIRES_MODE_CHANGE:
            grid_row += 1
            self.hiresolution   = Tk.IntVar()
            cb = Tk.Checkbutton( base_frame, text="Hi-resolution mode for screenshots",
                                    variable=self.hiresolution  )
            cb.grid( row=grid_row, column=0, sticky=Tk.W )

        self.monitor_previous = self.monitor.get()

        # Taking Screenshots
        grid_row += 1
        self.take_screenshots = Tk.IntVar()
        self.take_screenshots.set( get_dev_setting( 'take_screenshots' ) )

        cb = Tk.Checkbutton( base_frame, text="Take screenshots automatically for window size problem investigation",
                                variable=self.take_screenshots  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        grid_row += 1
        folder_frame = Tk.Frame( base_frame )
        folder_frame.grid( row=grid_row, column=0, sticky=Tk.W, padx=40 )

        folder_label = Tk.Label( folder_frame, text="Save Folder" )
        folder_label.grid( row=0, column=0, sticky=Tk.W )

        self.screenshot_folder = Tk.StringVar()
        folder = get_dev_setting( 'screenshot_folder' )
        if folder is None:
            folder = get_home_folder() + '/img'
        self.screenshot_folder.set( folder )
        folder_entry = FolderEntry( folder_frame, textvariable=self.screenshot_folder, width=60 )
        folder_entry.grid( row=0, column=1, sticky=Tk.W, padx=5 )

        grid_row += 1
        space = Tk.Label( base_frame, text="" )
        space.grid( row=grid_row, column=0 )

        # Intensity Data Reduction
        grid_row += 1
        self.intensity_reduction = Tk.IntVar()
        self.intensity_reduction.set( get_dev_setting( 'intensity_reduction' ) )

        cb = Tk.Checkbutton( base_frame, text="Reduce intensity data",
                                variable=self.intensity_reduction  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        reduction_state = Tk.NORMAL

        grid_row += 1
        method_frame = Tk.Frame( base_frame )
        method_frame.grid( row=grid_row, column=0, sticky=Tk.W, padx=40 )

        method_grid_row = 0
        cycle_label = Tk.Label( method_frame, text="leaving one out of every " )
        cycle_label.grid( row=method_grid_row, column=0, sticky=Tk.E )

        self.reduction_cycle = Tk.IntVar()
        self.reduction_cycle.set( get_dev_setting( 'reduction_cycle' ) )
        self.reduction_cycle_entry = Tk.Spinbox( method_frame, textvariable=self.reduction_cycle,
                                            from_=2, to=4, increment=1, 
                                            justify=Tk.CENTER, width=6, state=reduction_state )
        self.reduction_cycle_entry.grid( row=method_grid_row, column=1, sticky=Tk.E )

        by_label = Tk.Label( method_frame, text=" Q-points by" )
        by_label.grid( row=method_grid_row, column=2, sticky=Tk.W )

        method_grid_row += 1
        self.reduction_method_buttons = []
        self.reduction_method = Tk.StringVar()
        self.reduction_method.set( get_dev_setting( 'reduction_method' ) )

        b = Tk.Radiobutton( method_frame, text='thinning out',
                    # font=self.fixed_font,
                    variable=self.reduction_method, value='THIN-OUT',
                    state=reduction_state,
                    )
        b.grid( row=method_grid_row, column=0, sticky=Tk.W, padx=20  )
        self.reduction_method_buttons.append( b )

        start_at_label = Tk.Label( method_frame, text="starting at " )
        start_at_label.grid( row=method_grid_row, column=1, sticky=Tk.W )

        self.reduction_start = Tk.IntVar()
        self.reduction_start.set( get_dev_setting( 'reduction_start' ) )
        self.reduction_start_entry = Tk.Spinbox( method_frame, textvariable=self.reduction_start,
                                            from_=0, to=3, increment=1, 
                                            justify=Tk.CENTER, width=6, state=reduction_state )
        self.reduction_start_entry.grid( row=method_grid_row, column=2, sticky=Tk.W )

        method_grid_row += 1
        b = Tk.Radiobutton( method_frame, text='averaging',
                    # font=self.fixed_font,
                    variable=self.reduction_method, value='AVERAGE',
                    state=Tk.DISABLED,
                    )
        b.grid( row=method_grid_row, column=0, sticky=Tk.W, padx=20 )
        self.reduction_method_buttons.append( b )

        # Show number of iterations entry for xray-scattering baseline corection
        grid_row += 1
        self.show_num_iterations = Tk.IntVar()
        self.show_num_iterations.set( get_dev_setting( 'show_num_iterations' ) )

        cb = Tk.Checkbutton( base_frame, text="Show number of iterations entry for xray-scattering baseline corection",
                                variable=self.show_num_iterations  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Suppress usable_limit cutting
        grid_row += 1
        self.no_usable_q_limit = Tk.IntVar()
        self.no_usable_q_limit.set( get_dev_setting( 'no_usable_q_limit' ) )

        cb = Tk.Checkbutton( base_frame, text="Suppress usable_q_limit cutting",
                                variable=self.no_usable_q_limit  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Log Memory Usage
        grid_row += 1
        self.log_memory_usage = Tk.IntVar()
        self.log_memory_usage.set( get_dev_setting( 'log_memory_usage' ) )

        cb = Tk.Checkbutton( base_frame, text="Log memory usage",
                                variable=self.log_memory_usage  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Log Xray LPM Params
        grid_row += 1
        self.log_xray_lpm_params = Tk.IntVar()
        self.log_xray_lpm_params.set( get_dev_setting( 'log_xray_lpm_params' ) )

        cb = Tk.Checkbutton( base_frame, text="Log Xray LPM Params",
                                variable=self.log_xray_lpm_params  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Show number of iterations entry for xray-scattering baseline corection
        grid_row += 1
        self.show_num_iterations = Tk.IntVar()
        self.show_num_iterations.set( get_dev_setting( 'show_num_iterations' ) )

        cb = Tk.Checkbutton( base_frame, text="Show number of iterations entry for xray-scattering baseline corection",
                                variable=self.show_num_iterations  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Suppress boundary test deferring to allow boundaries in Guinier interval
        grid_row += 1
        self.suppress_defer_test = Tk.IntVar()
        self.suppress_defer_test.set( get_dev_setting( 'suppress_defer_test' ) )

        cb = Tk.Checkbutton( base_frame, text="Suppress boundary test deferring to allow boundaries to enter in the Guinier interval",
                                variable=self.suppress_defer_test  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Use DATGNOM to fill [from P(r)] items in the Summary for Puiblication
        grid_row += 1
        self.use_datgnom = Tk.IntVar()
        self.use_datgnom.set( get_dev_setting( 'use_datgnom' ) )

        cb = Tk.Checkbutton( base_frame, text="Use DATGNOM to fill [from P(r)] items in the Summary for Puiblication",
                                variable=self.use_datgnom  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Enable baseline drift simulation
        grid_row += 1
        self.enable_drift_simulation = Tk.IntVar()
        self.enable_drift_simulation.set( get_setting( 'enable_drift_simulation' ) )

        cb = Tk.Checkbutton( base_frame, text="Enable baseline drift simulation",
                                variable=self.enable_drift_simulation  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Constant Term in Regression
        grid_row += 1
        self.zx_add_constant = Tk.IntVar()
        self.zx_add_constant.set( get_dev_setting( 'zx_add_constant' ) )

        cb = Tk.Checkbutton( base_frame, text="Add a constant term in Q-iterated WLS for A(q), B(a)",
                                variable=self.zx_add_constant  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Constant Row in Concentration Matrix
        grid_row += 1
        self.add_conc_const = Tk.IntVar()
        self.add_conc_const.set( get_dev_setting( 'add_conc_const' ) )

        cb = Tk.Checkbutton( base_frame, text="Add a constant row to concentration matrix",
                                variable=self.add_conc_const )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Individual B(q) Ignore
        grid_row += 1
        self.individual_bq_ingore = Tk.IntVar()
        self.individual_bq_ingore.set( get_dev_setting( 'individual_bq_ingore' ) )

        cb = Tk.Checkbutton( base_frame, text="Enable individual bq_ingore",
                                variable=self.individual_bq_ingore )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Excel Visibility
        grid_row += 1
        self.make_excel_visible = Tk.IntVar()
        self.make_excel_visible.set( get_dev_setting( 'make_excel_visible' ) )

        cb = Tk.Checkbutton( base_frame, text="Make Excel visible",
                                variable=self.make_excel_visible  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # Remaining Excel
        grid_row += 1
        self.keep_remaining_excel = Tk.IntVar()
        self.keep_remaining_excel.set( get_dev_setting( 'keep_remaining_excel' ) )
        cb = Tk.Checkbutton( base_frame, text="Keep remaining Excel instances instead of killing",
                                variable=self.keep_remaining_excel  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # DnD Debug
        grid_row += 1
        self.enable_dnd_debug = Tk.IntVar()
        self.enable_dnd_debug.set( get_dev_setting( 'enable_dnd_debug' ) )
        cb = Tk.Checkbutton( base_frame, text="enable DnD Debug",
                                variable=self.enable_dnd_debug  )
        cb.grid( row=grid_row, column=0, sticky=Tk.W )

        # global grab cannot be set befor windows is 'viewable'
        # and this happen in mainloop after this function returns
        # Thus, it is needed to delay grab setting of an interval
        # long enough to make sure that the window has been made
        # 'viewable'
        if self.grab == 'global':
            self.after(100, self.grab_set_global )
        else:
            pass # local grab is set by parent class constructor

        self.update()

    def buttonbox( self ):
        box = Tk.Frame( self )
        box.pack( pady=10 )
        Dialog.buttonbox( self, frame=box )

    def clear_settings( self ):
        ok = MessageBox.askyesno( 'Warninig', 'This operations is not cancelable. OK?' )
        if ok:
            clear_settings()

    def apply( self ):  # overrides parent class method
        take_screenshots    = self.take_screenshots.get()
        screenshot_folder   = self.screenshot_folder.get()
        if ENABLE_HIRES_MODE_CHANGE:
            set_dev_setting( 'hiresolution',        self.hiresolution.get() )
        set_dev_setting( 'take_screenshots',    take_screenshots )
        set_dev_setting( 'screenshot_folder',   screenshot_folder )
        set_dev_setting( 'intensity_reduction', self.intensity_reduction.get() )
        set_dev_setting( 'reduction_cycle',     self.reduction_cycle.get() )
        set_dev_setting( 'reduction_start',     self.reduction_start.get() )
        set_dev_setting( 'no_usable_q_limit',   self.no_usable_q_limit.get() )
        set_dev_setting( 'log_memory_usage',    self.log_memory_usage.get() )
        set_dev_setting( 'log_xray_lpm_params', self.log_xray_lpm_params.get() )
        set_dev_setting( 'show_num_iterations', self.show_num_iterations.get() )
        set_dev_setting( 'suppress_defer_test', self.suppress_defer_test.get() )
        set_dev_setting( 'use_datgnom',         self.use_datgnom.get() )
        set_setting( 'enable_drift_simulation', self.enable_drift_simulation.get() )
        set_dev_setting( 'zx_add_constant',     self.zx_add_constant.get() )
        set_dev_setting( 'add_conc_const',      self.add_conc_const.get() )
        set_dev_setting( 'individual_bq_ingore', self.individual_bq_ingore.get() )
        set_dev_setting( 'make_excel_visible',   self.make_excel_visible.get() )
        set_dev_setting( 'keep_remaining_excel', self.keep_remaining_excel.get() )
        set_dev_setting( 'enable_dnd_debug',    self.enable_dnd_debug.get() )

        if take_screenshots:
            from OurScreenShot import screenshot
            self.parent.after( 500, lambda: screenshot( widget=self.parent, log=True ) )
        os.environ[ PYTHON_DEMO_MONITOR ]   = monitor = self.monitor.get()
        if monitor != self.monitor_previous:
            new_geometry = adjusted_geometry(self.parent.geometry())
            self.parent.geometry( new_geometry )
            print( 'changed geometry according to monitor selection', monitor, new_geometry )
        self.applied    = True
