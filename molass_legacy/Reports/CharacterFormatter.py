"""

    CharacterFormatter.py

    Copyright (c) 2023, SAXS Team, KEK-PF

"""
from Unicode import SUPERSCRIPT, SUBSCRIPT

SUPERSCRIPT_INV = {}
for k,v in SUPERSCRIPT.items():
    SUPERSCRIPT_INV[v] = k

SUBSCRIPT_INV = {}
for k,v in SUBSCRIPT.items():
    SUBSCRIPT_INV[v] = k

def to_superscript_font(range_):
    n = range_.Characters.Count
    for i in range(1, n+1):
        c = range_.GetCharacters(i,1)
        t = SUPERSCRIPT_INV.get(c.Text)
        if t is not None:
            c.Text = t
            c.Font.Superscript = True

def to_subscript_font(range_):
    n = range_.Characters.Count
    for i in range(1, n+1):
        c = range_.GetCharacters(i,1)
        t = SUBSCRIPT_INV.get(c.Text)
        if t is not None:
            c.Text = t
            c.Font.Subscript = True

def to_italic_font(range_, targets):
    n = range_.Characters.Count
    c = range_.GetCharacters(1,n)
    text = c.Text
    for t in targets:
        i = text.find(t)
        if i >= 0:
            j = t.find('(')
            if j >= 0:
                # parentheses should not be italic
                c_ = range_.GetCharacters(i+1,j)
                c_.Font.Italic = True
                k = t.find(')')
                assert k > 1
                if t[j+1:k].isnumeric():
                    # numeric chars should not be italic
                    pass
                else:
                    c_ = range_.GetCharacters(i+j+2,k-1-j)
                    c_.Font.Italic = True
            else:
                c_ = range_.GetCharacters(i+1,len(t))
                c_.Font.Italic = True

def to_italic_subscipt_font(range_, targets):
    n = range_.Characters.Count
    c = range_.GetCharacters(1,n)
    text = c.Text
    for t in targets:
        i = text.find(t)
        if i >= 0:
            c_ = range_.GetCharacters(i+1,len(t))
            c_.Font.Italic = True
            cs = range_.GetCharacters(i+2,len(t)-1)
            cs.Font.Subscript = True
