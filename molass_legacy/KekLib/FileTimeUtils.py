"""

    FileTimeUtils.py

    Copyright (c) 2019, Masatsuyo Takahashi, KEK-PF

    References

    How do I change the file creation date of a Windows file from Python?
    https://stackoverflow.com/questions/4996405/how-do-i-change-the-file-creation-date-of-a-windows-file-from-python

    
    
"""
import os
import win32file, win32con, pywintypes

def getFileDateTimes( filePath ):        
    return ( os.path.getctime( filePath ), 
             os.path.getmtime( filePath ), 
             os.path.getatime( filePath ) )

def setFileDateTimes( filePath, datetimes ):
    ctime, mtime, atime = [pywintypes.Time(t) for t in datetimes]
    winfile = win32file.CreateFile(
        filePath, win32con.GENERIC_WRITE,
        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE | win32con.FILE_SHARE_DELETE,
        None, win32con.OPEN_EXISTING,
        win32con.FILE_ATTRIBUTE_NORMAL, None)
    win32file.SetFileTime( winfile, ctime, atime, mtime )
    winfile.close()
