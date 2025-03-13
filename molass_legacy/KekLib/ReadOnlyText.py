# coding: utf-8
"""

    ReadOnlyText.py

    copied from
    Readonly tkinter text widget
    https://stackoverflow.com/questions/21873195/readonly-tkinter-text-widget/21882093

"""
import tkinter as Tk

# This is the list of all default command in the "Text" tag that modify the text
commandsToRemove = (
"<Control-Key-h>",
"<Meta-Key-Delete>",
"<Meta-Key-BackSpace>",
"<Meta-Key-d>",
"<Meta-Key-b>",
"<<Redo>>",
"<<Undo>>",
"<Control-Key-t>",
"<Control-Key-o>",
"<Control-Key-k>",
"<Control-Key-d>",
"<Key>",
"<Key-Insert>",
"<<PasteSelection>>",
"<<Clear>>",
"<<Paste>>",
"<<Cut>>",
"<Key-BackSpace>",
"<Key-Delete>",
"<Key-Return>",
"<Control-Key-i>",
"<Key-Tab>",
"<Shift-Key-Tab>"
)

class ReadOnlyText(Tk.Text):
    tagInit = False

    def init_tag(self):
        """
        Just go through all binding for the Text widget.
        If the command is allowed, recopy it in the ROText binding table.
        """
        for key in self.bind_class("Text"):
            if key not in commandsToRemove:
                command = self.bind_class("Text", key)
                self.bind_class("ReadOnlyText", key, command)
        ReadOnlyText.tagInit = True

    def __init__(self, *args, **kwords):
        Tk.Text.__init__(self, *args, **kwords)
        if not ReadOnlyText.tagInit:
            self.init_tag()

        # Create a new binding table list, replace the default Text binding table by the ROText one
        bindTags = tuple(tag if tag!="Text" else "ReadOnlyText" for tag in self.bindtags())
        self.bindtags(bindTags)

class CopyableLabel(ReadOnlyText):
    def __init__( self, parent, text, *args, **kwargs ):
        ReadOnlyText.__init__( self, parent, *args, **kwargs )
        self.insert( Tk.END, text )
        lable = Tk.Label( parent )
        bg = lable.cget( 'bg' )
        self.config( relief=Tk.FLAT, height=1, width=len(text), bg=bg )

    def config( self, text=None, *args, **kwargs ):
        if text is not None:
            self.delete( '1.0', Tk.END )
            self.insert( Tk.END, text )
            self.config( width=len(text) )
        ReadOnlyText.config( self, *args, **kwargs )
