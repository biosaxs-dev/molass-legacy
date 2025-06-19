"""
    ExcelCOM.py - a minimal Python wrapper for Excel ole object

    Copyright (c) 2016-2025, Masatsuyo Takahashi, KEK-PF
"""
import os
import re
import time
import numpy as np
from win32com.client import Dispatch, DispatchEx, gencache
from win32process import GetWindowThreadProcessId
import pythoncom
from shutil import copyfile
import logging
import psutil

DEBUG = False

WAIT_TIME_BEFORE_KILL           = 10

msoTrue                         = -1
msoFalse                        =  0
msoConnectorStraight            =  1
msoShapeOval                    =  9
msoShapeRightBrace              = 32
msoTextOrientationHorizontal    =  1
msoThemeColorAccent1            = 5
msoThemeColorText1              = 13
xlCategory                      = 1
xlValue                         = 2
xlLine                          = 4
xlUpward                        = -4171
xlLow                           = -4134
xlNone                          = -4142
xlXYScatter                     = -4169
xlMaximum                       = 2

worksheet_path_re = re.compile('^(.+\.xls[xm])\((.+)\)$')

def CoInitialize():
    pythoncom.CoInitialize()

def CoUninitialize():
    pythoncom.CoUninitialize()

def ColAddr( col ):
    col_ = col - 1
    addr = ''
    while True:
        r = col_ % 26
        col_ //= 26
        addr = chr(65+r) + addr
        if col_ == 0:
            break
        else:
            col_ -= 1
    return addr

def CellAddr( cell, absref='$$' ):
    row_abs = absref[0].replace( ' ', '' )
    col_cabs = absref[1].replace( ' ', '' )
    return row_abs + ColAddr( cell[1] ) + col_cabs + str(cell[0])

def RGB( hex_color_str ):
    return int( hex_color_str[4:6] + hex_color_str[2:4] + hex_color_str[0:2], 16 )

CLSID = 'Excel.Application'
INVOKE_EXCEL_SIMPLE_DISPATCH = True

pid_list = []

def cleanup_created_excels():
    logger = logging.getLogger( __name__  )
    while len(pid_list) > 0:
        pid = pid_list.pop(0)
        try:
            proc    = psutil.Process( pid )
            if proc.is_running():
                proc.kill()
                logger.warning( 'Killed the remaining process pid=' + str(pid) + ' in cleanup.' )
        except Exception as exc:
            # psutil.NoSuchProcess no process found with pid ...
            logger.debug( str(exc) + '; ok' )

class ExcelComClient:
    def __init__( self, new_instance=True, excel_id=None, keep_remaining=False, visible=False, display_alerts=False ):
        self.clsid = None
        self.excel = None
        self.new_instance = None
        self.keep_remaining = keep_remaining
        self.visible = visible
        self.logger = logging.getLogger( __name__  )
        if excel_id is None:
            if new_instance:
                self.excel  = DispatchEx(CLSID)
                self.new_instance = True
            else:
                self.excel  = Dispatch(CLSID)
                self.new_instance = False
        else:
            self.new_instance = False
            self.excel  = Dispatch(
                            pythoncom.CoGetInterfaceAndReleaseStream( excel_id, pythoncom.IID_IDispatch )
                            )

        if self.visible:
            self.excel.Visible = True

        self.excel.DisplayAlerts = display_alerts

        self.pid = GetWindowThreadProcessId( self.excel.Hwnd )[1]
        if self.new_instance:
            pid_list.append( self.pid )
        self.logger.debug( 'Excel started with pid %d' % self.pid )

    def __del__( self ):
        if self.excel is not None:
            # self.logger.info( 'Excel Quit in ExcelComClient.__del__' )
            self.quit()

        if self.new_instance:
            time.sleep(1)
            try:
                proc    = psutil.Process( self.pid )
                n = 0
                while n < WAIT_TIME_BEFORE_KILL and proc.is_running():
                    time.sleep(1)
                    n += 1
                if proc.is_running():
                    if self.keep_remaining:
                        self.logger.warning( 'Kept the remaining process pid=' + str(self.pid) )
                    else:
                        proc.kill()
                        self.logger.warning( 'Killed the remaining process pid=' + str(self.pid) )
            except Exception as exc:
                # psutil.NoSuchProcess no process found with pid ...
                self.logger.debug( str(exc) + '; ok' )

            try:
                pid_list.remove( self.pid )
            except ValueError:
                # this happens in cases after cleanup_created_excels
                pass

    def get_pid( self ):
        self.pid = GetWindowThreadProcessId(self.excel.Hwnd)[1]
        return self.pid

    def get_comid( self ):
        excel_id    = pythoncom.CoMarshalInterThreadInterfaceInStream( pythoncom.IID_IDispatch, self.excel )
        return excel_id

    def get_version( self ):
        return self.excel.Version

    def get_thread_mode(self):
        return self.excel.MultiThreadedCalculation.ThreadMode

    def get_thread_count(self):
        return self.excel.MultiThreadedCalculation.ThreadCount

    def create_thread( self, target, args=[], kwargs={}, name='ExcelComClientThread' ):
        import threading
        thread      = threading.Thread(
                                    target=target,
                                    name=name,
                                    args=args,
                                    kwargs=kwargs
                                    )
        return thread

    def selection( self ):
        return self.excel.Selection

    def quit( self ):
        # self.excel.DisplayAlerts = alert
        try:
            self.excel.Quit()
            self.logger.debug( 'Excel Quit ok' )
        except AttributeError:
            # AttributeError: Excel.Application.Quit
            self.logger.warning( 'Excel with pid %d seems to have died' % self.pid )
        self.excel = None

    def openWorkbook( self, path, save=True ):
        path_ = path.replace( '/', '\\' )
        wb = Workbook_( self, path_, save )
        return wb

    def openWorksheet( self, path ):
        path_ = path.replace( '/', '\\' )
        ws = Worksheet( self, path_ )
        return ws

    def merge_books( self, input_books, merged_book, progress_cb=None, merged_cb=None, delete_target=None, default_font=None ):
        if merged_book in input_books:
            raise Exception( "input_books must not include merged_book" )

        if os.path.exists( merged_book ):
            os.remove( merged_book )

        # Opening a new book causes a loss of book parameters
        # which are not copied by ws.Copy method.
        # So, it is better to filecopy to preserve ones in the
        # first book.
        copyfile( input_books[-1], merged_book )

        if progress_cb is not None:
            progress_cb(0)

        if len( input_books ) == 1:
            return

        out_wb = self.openWorkbook( merged_book ).workbook
        if default_font is not None:
            out_wb.Styles("Normal").Font.Name = default_font

        # num_sheets = out_wb.Sheets.Count
        target_ws = out_wb.Sheets(1)
        # self.logger.info( 'debug: target_ws.Name=' + target_ws.Name )

        for i, book in enumerate(input_books[:-1]):
            try:
                in_wb = self.openWorkbook( book ).workbook
            except:
                self.logger.error( 'Skipped because of an open error: ' + book )
                continue

            for ws in in_wb.Sheets:
                # print( "ws.Name", ws.Name )
                # self.logger.info( 'debug: copying ' + ws.Name )
                # ? After results in a new book
                ws.Copy( Before=target_ws )
            in_wb.Close( SaveChanges=False )
            if progress_cb is not None:
                progress_cb(i)

        if target_ws.Name == delete_target:
            target_ws.Delete()

        if merged_cb is not None:
            merged_cb( self, out_wb )

        out_wb.Sheets(1).Activate()
        out_wb.Save()
        out_wb.Close()

def merge_into_a_book( input_books, merged_book, excel_client=None, progress_cb=None, delete_target=None, default_font=None ):
    if excel_client is None:
        CoInitialize()  # required in thread
        excel_client_ = ExcelComClient()
    else:
        excel_client_ = excel_client

    """
    TODO: this doen't work!
    def freeze_pane_sheet1( excel_client, wb ):
        # workaourd to fix broken FreezePanes setting
        ws = wb.Worksheets(1)
        ws.Activate()
        excel_client.excel.Range( "D2" ).Select()
        aw = excel_client.excel.ActiveWindow
        aw.FreezePanes  = True
        aw.ScrollColumn = 16

    excel_client_.merge_books( input_books, merged_book, progress_cb=progress_cb, merged_cb=freeze_pane_sheet1 )
    """
    excel_client_.merge_books( input_books, merged_book, progress_cb=progress_cb, delete_target=delete_target, default_font=default_font )

    if excel_client is None:
        excel_client_ = None
        CoUninitialize()

"""
    Naming this class as Workbook_ is to distinguish Workbook  from openpyxl
"""
class Workbook_:
    def __init__( self, excel_client, path, save=True ):
        self.excel_client   = excel_client
        self.excel          = excel_client.excel
        if os.path.exists( path ):
            wb_ = self.excel.Workbooks.Open( path )
        else:
            wb_ = self.excel.Workbooks.Add()
            if save:
                wb_.SaveAs( path )

        self.workbook = wb_

class Worksheet:
    def __init__( self, excel_client, path ):
        self.excel_client   = excel_client
        self.excel          = excel_client.excel
        m = worksheet_path_re.match( path )
        if not m: raise Exception('Invalid path: ' + path )

        wb_path = m.group(1)
        ws_name = m.group(2)
        # print( 'Worksheet:', wb_path, ws_name )

        self.workbook   = wb_ = self.excel_client.openWorkbook( wb_path ).workbook
        try:
            ws_ = wb_.Worksheets(ws_name)
        except:
            ws_ = wb_.Worksheets(1)
            ws_.Name = ws_name

        # print( 'ws_.Name=', ws_.Name )
        self.worksheet  = ws_

    def draw_line( self, p1, p2 ):
        line = self.worksheet.Shapes.AddConnector( msoConnectorStraight, p1[0], p1[1], p2[0], p2[1] )
        line.Select()
        return line

    def draw_oval( self, p, a, b, color=None ):
        oval = self.worksheet.Shapes.AddShape( msoShapeOval, p[0], p[1], a, b )
        oval.Select()
        if color is not None:
            selection = self.excel.Selection
            selection.ShapeRange.Fill.ForeColor.RGB = color
        return oval

    def draw_right_brace( self, p, width, hight, rotation ):
        """
        Function AddShape(Type As MsoAutoShapeType, Left As Single, Top As Single, Width As Single, Height As Single) As Shape
        """
        brace = self.worksheet.Shapes.AddShape( msoShapeRightBrace, p[0], p[1], width, hight )
        brace.Select()
        self.excel.Selection.ShapeRange.Rotation = rotation
        return brace

    def draw_textbox( self, text, p, width, hight, visible=True ):
        textbox = self.worksheet.Shapes.AddTextbox( msoTextOrientationHorizontal, p[0], p[1], width, hight )
        textbox.Select()
        selection = self.excel.Selection
        selection.ShapeRange.Line.Visible = visible
        selection.ShapeRange(1).TextFrame2.TextRange.Characters.Text = text
        return textbox

    def freeze_panes( self, range_, scroll_column=None ):
        self.excel.Range( range_ ).Select()
        self.excel.ActiveWindow.FreezePanes = True
        if scroll_column is not None:
            self.excel.ActiveWindow.ScrollColumn = scroll_column

    def zoom( self, percentage ):
        self.excel.ActiveWindow.Zoom = percentage

    def get_num_charts( self ):
        return self.worksheet.ChartObjects().Count

    def get_chart(self, i, debug=False):
        if debug:
            for chart in self.worksheet.ChartObjects():
                print( chart.Name )

        chartobj = self.worksheet.ChartObjects()[i]

        if debug:
            print( 'chartobj.Chart=', chartobj.Chart )
            for ser in chartobj.Chart.SeriesCollection():
                print( ser.Name )

        return chartobj.Chart

    def get_chart_data_point( self, chart, i, j ):
        # <class 'numpy.int64'> can cause 'MemoryError: CreatingSafeArray'
        i_, j_ = int(i), int(j)
        point  = chart.SeriesCollection()[i_].Points()[j_]

        if DEBUG:
            print( 'point.Left=', point.Left )
            print( 'point.Top=', point.Top )

        return point

def compute_axis_max_value( max_value, num_labels=5 ):
    log10_  = int( np.floor( np.log10( max_value / num_labels ) ) )
    scale   = np.power( 10.0, log10_ )
    scaled_value = max_value/scale
    # print( 'scale=', scale, 'scaled_value=', scaled_value )
    return np.ceil( scaled_value ) * scale

def excel_availability(debug=False):
    """
    Note that this test may require rebooting after changing the Excel abailability.
    """
    CoInitialize()
    version = None
    try:
        excel = ExcelComClient()
        version  = excel.get_version()
        excel.quit()
    except:
        if debug:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            print(etb)
    CoUninitialize()
    return version
