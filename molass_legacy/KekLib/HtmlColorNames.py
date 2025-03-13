# coding: utf-8
"""

    HtmlColorNames.py

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF

"""
# HTML Color Names from http://www.w3schools.com/colors/colors_names.asp
black       = '000000'
red         = 'FF0000'
blue        = '0000FF'
gold        = 'FFD700'
orange      = 'FFA500'
green       = '008000'
seagreen    = '2E8B57'
darkseagreen= '8FBC8F'
limegreen   = '32CD32'
lightgreen  = '90EE90'
steelblue   = '4682B4'
deepskyblue = '00BFFF'
lightskyblue= '87CEFA'
dodgerblue  = '1E90FF'
royalblue   = '4169E1'
lightgray   = 'D3D3D3'
darkgray    = 'A9A9A9'
darkorange  = 'FF8C00'

"""
    to convert to int do like as follows

    int_color = int( red, 16 )

    or

    from ExcelCOM import RGB
    xl_rgb_color = RGB( red )

    note that in Excel VBA
        0x50B000 == RGB(0, 176, 80)
"""
