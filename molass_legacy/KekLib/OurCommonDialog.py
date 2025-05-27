# base class for tk common dialogues
#
# this module provides a base class for accessing the common
# dialogues available in Tk 4.2 and newer.  use filedialog,
# colorchooser, and messagebox to access the individual
# dialogs.
#
# written by Fredrik Lundh, May 1997
#

"""
  mofified by Masatsuyo Takahashi, December 2015,
    1) make Dialog derive from Tk.Toplevel
    2) self.transient( self.parent )
    3) self.grab_set()
    4) self.withdraw()
    5) self.grab_release()
    6) changed self.master to self.parent
"""
from molass_legacy.KekLib.OurTkinter import Tk, TkVersion

class Dialog( Tk.Toplevel ):

    command  = None

    def __init__(self, master=None, **options):

        # FIXME: should this be placed on the module level instead?
        if TkVersion < 4.2:
            raise TclError("this module requires Tk 4.2 or newer")

        Tk.Toplevel.__init__( self )

        """
        changed self.master to self.parent, because
        self.master produced the following message during destruction.
          File "C:\Program Files\Python 3.5\lib\tkinter\__init__.py", line 2145, in destroy
        AttributeError: 'NoneType' object has no attribute 'children'
        """
        self.parent  = master
        self.options = options
        if not master and options.get('parent'):
            self.parent = options['parent']

        self.grab = False
        if 'grab' in options:
            self.grab = options.get('grab')
            del options[ 'grab' ]

        if self.parent and self.grab:
            self.transient( self.parent )
            self.grab_set()

        self.withdraw()

    def _fixoptions(self):
        pass # hook

    def _fixresult(self, widget, result):
        return result # hook

    def show(self, **options):

        # update instance options
        for k, v in options.items():
            self.options[k] = v

        self._fixoptions()

        # we need a dummy widget to properly process the options
        # (at least as long as we use Tkinter 1.63)
        w = Tk.Frame(self.master)

        try:

            s = w.tk.call(self.command, *w._options(self.options))

            s = self._fixresult(w, s)

        finally:

            try:
                # get rid of the widget
                w.destroy()
                if self.parent and self.grab:
                    self.grab_release()
            except:
                pass

        return s
