"""
    Console.py
"""
import sys
import time
import win32console, win32con
import psutil

class Console:
    def __init__( self, windows_app_only=False, console_parent_only=False ):
        self.activeted = False

        proc    = psutil.Process()
        pproc   = psutil.Process(proc.ppid())
        ppproc  = psutil.Process(pproc.ppid())
        pname   = pproc.name()
        ppname  = ppproc.name()
        parent_is_console = 'cmd.exe' in [ pname, ppname ]
        if console_parent_only and not parent_is_console:
            return

        free_console=True
        try:
            win32console.AllocConsole()
        except win32console.error as exc:
            if exc.winerror!=5:
                raise
            ## only free console if one was created successfully
            free_console=False

        if windows_app_only:
            if not free_console: return

        self.activeted = True

        self.free_console = free_console
        self.stdout     = win32console.GetStdHandle(win32console.STD_OUTPUT_HANDLE)
        self.stdin      = win32console.GetStdHandle(win32console.STD_INPUT_HANDLE)

        self.newbuffer  = newbuffer = win32console.CreateConsoleScreenBuffer()

        newbuffer.SetConsoleActiveScreenBuffer()
        """
        newbuffer.SetConsoleTextAttribute(win32console.FOREGROUND_RED|win32console.FOREGROUND_INTENSITY
                |win32console.BACKGROUND_GREEN|win32console.BACKGROUND_INTENSITY)
        """
        newbuffer.WriteConsole('This is a new screen buffer\n')

        ## test setting screen buffer and window size
        ## screen buffer size cannot be smaller than window size
        window_size=newbuffer.GetConsoleScreenBufferInfo()['Window']
        coord=win32console.PyCOORDType(X=window_size.Right+20, Y=window_size.Bottom+20)
        newbuffer.SetConsoleScreenBufferSize(coord)

        window_size.Right+=10
        window_size.Bottom+=10

        newbuffer.SetConsoleWindowInfo(Absolute=True,ConsoleWindow=window_size)
        self.stdout_save = sys.stdout
        sys.stdout      = self

        # self.write( 'free_console=' + str( self.free_console ) + '\n' )

    def __del__( self ):
        if not self.activeted: return

        print( 'press <Control> to close.' )

        breakout=False
        while not breakout:
            input_records = self.stdin.ReadConsoleInput(1)
            for input_record in input_records:
                if input_record.VirtualKeyCode==win32con.VK_END:
                    breakout=True

            time.sleep(0.1)

        self.stdout.SetConsoleActiveScreenBuffer()

        if self.free_console:
             win32console.FreeConsole()
        sys.stdout = self.stdout_save

    def write( self, buffer ):
        self.newbuffer.WriteConsole( buffer )

    def flush( self, *args ):
        pass

"""
    https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
    http://vivi.dyndns.org/tech/cpp/setColor.html
"""
import ctypes

STD_OUTPUT_HANDLE = -11
COLOR_WHITE = 0x0f
COLOR_GRAY  = 0x07

def reset_text_color():
    handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
    ctypes.windll.kernel32.SetConsoleTextAttribute(handle, COLOR_GRAY)
