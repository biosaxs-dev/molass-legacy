"""

    Unicode.py

    Copyright (c) 2017-2024, Masatsuyo Takahashi

"""

SUPERSCRIPT = {
    '+' : '⁺',
    '-' : '⁻',
    '0' : '⁰',
    '1' : '¹',
    '2' : '²',
    '3' : '³',
    '4' : '⁴',
    '5' : '⁵',
    '6' : '⁶',
    '7' : '⁷',
    '8' : '⁸',
    '9' : '⁹',
    'i' : 'ⁱ',
    'n' : 'ⁿ',
}

def to_superscripts( a ):
    s = ''
    for c in a:
        s += SUPERSCRIPT[c]
    return s

SUBSCRIPT = {
    '0' : '₀',
    '1' : '₁',
    '2' : '₂',
    '3' : '₃',
    '4' : '₄',
    '5' : '₅',
    '6' : '₆',
    '7' : '₇',
    '8' : '₈',
    '9' : '₉',
}

def to_subscripts( a ):
    s = ''
    for c in a:
        s += SUBSCRIPT[c]
    return s

"""
    ≡
    ≈
    ⇔
    ‖
    ︙
    ⧵   Backslash (REVERSE)
    Å
"""
