# tk common message boxes
#
# this module provides an interface to the native message boxes
# available in Tk 4.2 and newer.
#
# written by Fredrik Lundh, May 1997
#

#
# options (all have default values):
#
# - default: which button to make default (one of the reply codes)
#
# - icon: which icon to display (see below)
#
# - message: the message to display
#
# - parent: which window to place the dialog on top of
#
# - title: dialog title
#
# - type: dialog type; that is, which buttons to display (see below)
#
from __future__ import division, print_function, unicode_literals
from OurCommonDialog import Dialog
import queue

#
# constants

# icons
ERROR = "error"
INFO = "info"
QUESTION = "question"
WARNING = "warning"

# types
ABORTRETRYIGNORE = "abortretryignore"
OK = "ok"
OKCANCEL = "okcancel"
RETRYCANCEL = "retrycancel"
YESNO = "yesno"
YESNOCANCEL = "yesnocancel"

# replies
ABORT = "abort"
RETRY = "retry"
IGNORE = "ignore"
OK = "ok"
CANCEL = "cancel"
YES = "yes"
NO = "no"

class LastMessage:
    def __init__( self ):
        pass

    def set( self, title, message, reply=None ):
        self.title      = title
        self.message    = message
        self.reply      = reply

    def set_reply( self, reply ):
        self.reply      = reply

last_message = LastMessage()

def check_last_message( message ):
    assert( last_message.message.find( message ) >= 0 )

#
# message dialog class

class Message(Dialog):
    "A message box"

    command  = "tk_messageBox"

dialog_queue = queue.Queue()

#
# convenience stuff

# Rename _icon and _type options to allow overriding them in options
def _show(title=None, message=None, _icon=None, _type=None, **options):
    if _icon and "icon" not in options:    options["icon"] = _icon
    if _type and "type" not in options:    options["type"] = _type
    if title:   options["title"] = title
    if message: options["message"] = message
    options["grab"] = True
    last_message.set( title, message )  # to be checked before reply
    dialog =  Message(**options)
    dialog_queue.put( dialog )
    res = dialog.show()
    dialog_queue.get()

    if isinstance(res, bool):
        # In some Tcl installations, yes/no is converted into a boolean.
        if res:
            res_ = YES
        else:
            res_ = NO
    else:
        # In others we get a Tcl_Obj.
        res_ = str(res)

    last_message.set_reply( res_ )
    return res_

def showinfo(title=None, message=None, **options):
    "Show an info message"
    return _show(title, message, INFO, OK, **options)

def showwarning(title=None, message=None, **options):
    "Show a warning message"
    return _show(title, message, WARNING, OK, **options)

def showerror(title=None, message=None, **options):
    "Show an error message"
    return _show(title, message, ERROR, OK, **options)

def askquestion(title=None, message=None, **options):
    "Ask a question"
    return _show(title, message, QUESTION, YESNO, **options)

def askokcancel(title=None, message=None, **options):
    "Ask if operation should proceed; return true if the answer is ok"
    s = _show(title, message, QUESTION, OKCANCEL, **options)
    return s == OK

def askyesno(title=None, message=None, icon=None, **options):
    "Ask a question; return true if the answer is yes"
    icon_ = QUESTION if icon is None else icon
    s = _show(title, message, icon_, YESNO, **options)
    return s == YES

def askyesnocancel(title=None, message=None, **options):
    "Ask a question; return true if the answer is yes, None if cancelled."
    s = _show(title, message, QUESTION, YESNOCANCEL, **options)
    # s might be a Tcl index object, so convert it to a string
    s = str(s)
    if s == CANCEL:
        return None
    return s == YES

def askretrycancel(title=None, message=None, **options):
    "Ask if operation should be retried; return true if the answer is yes"
    s = _show(title, message, WARNING, RETRYCANCEL, **options)
    return s == RETRY

# --------------------------------------------------------------------
# test stuff

if __name__ == "__main__":

    print("info", showinfo("Spam", "Egg Information"))
    print("warning", showwarning("Spam", "Egg Warning"))
    print("error", showerror("Spam", "Egg Alert"))
    print("question", askquestion("Spam", "Question?"))
    print("proceed", askokcancel("Spam", "Proceed?"))
    print("yes/no", askyesno("Spam", "Got it?"))
    print("yes/no/cancel", askyesnocancel("Spam", "Want it?"))
    print("try again", askretrycancel("Spam", "Try again?"))
