# coding: utf-8
"""

    ResultTable.py

    Copyright (c) 2016-2017, Masatsuyo Takahashi, KEK-PF

"""

import os
from molass_legacy.KekLib.OurTkinter             import Tk, ToolTip
import OurMessageBox        as     MessageBox
from TkMiniTable            import TkMiniTable
from molass_legacy._MOLASS.SerialSettings               import get_setting

NUM_RESULT_COLUMNS  = 28
QUALITY_SIGNAL      =  7
MONO_RG             = 12
IPI_FLAG            = 19
BICO_FLAG           = 20

class ResultTable:
    def __init__( self, parent, rec_array ):
        self.parent = parent
        self.table  = None

        self.refresh( rec_array )

    def refresh( self, rec_array ):

        if self.table is not None:
            self.table.destroy()

        columns = list( map( lambda n: '%02d' % n, range( NUM_RESULT_COLUMNS ) ) )
        self.table = table = TkMiniTable( self.parent, columns=columns, default_colwidth=8 )
        table.pack( fill=Tk.BOTH, anchor=Tk.NW  )

        table.heading( '00', 'Tool ID.' )
        table.column ( '00', width=20 )
        table.heading( '01', 'Rg' )
        table.heading( '02', 'Rg StDev' )
        table.heading( '03', 'I(0)' )
        table.heading( '04', 'I(0) StDev' )
        table.heading( '05', 'First Point' )
        table.heading( '06', 'Last Point' )
        table.heading( '07', 'Quality' )
        table.heading( '08', 'Aggregated' )
        table.heading( '09', 'min q*Rg' )
        table.heading( '10', 'max q*Rg' )
        table.heading( '11', 'Mono G' )
        table.column ( range( 11, 14 ), bg = 'ivory')
        table.heading( '12', 'Mono Rg' )
        table.heading( '13', 'Mono d' )
        table.heading( '14', 'basic qual' )
        table.column ( range( 14, 19 ), bg = 'honeydew')
        table.heading( '15', 'positive s' )
        table.heading( '16', 'consistenc' )
        table.heading( '17', 'stdev scor' )
        table.heading( '18', 'qRg score' )
        table.heading( '19', 'IpI flag' )
        table.column ( range( 19, 28 ), bg = 'seashell')
        table.heading( '20', 'BiCo flag' )
        table.heading( '21', 'b/m ratio' )
        table.heading( '22', 'BiCo G1' )
        table.heading( '23', 'BiCo G2' )
        table.heading( '24', 'BiCo Rg1' )
        table.heading( '25', 'BiCo Rg2' )
        table.heading( '26', 'BiCo d1' )
        table.heading( '27', 'BiCo d2' )

        rec_a = rec_array[0]
        try:
            atsas_signal = 'white' if rec_a is None else rec_a.get_atsas_signal()
        except:
            # ErrorResult
            atsas_signal = 'white'
        rec_k = rec_array[2]
        try:
            kekpf_signal = 'white' if rec_k is None else rec_k.get_kekpf_signal()
        except:
            # ErrorResult
            kekpf_signal = 'white'

        signals     = [ atsas_signal, None, kekpf_signal ]
        tool_list   = [ 'ATSAS original', 'ATSAS re-evaluated', 'KEK' ]

        for row, rec in enumerate( rec_array ):
            val_array = []
            val_array.append( tool_list[row] )

            if rec is None:
                val_array += [ '' ] * ( NUM_RESULT_COLUMNS - 1 )
            else:
                for i in list( range(NUM_RESULT_COLUMNS - 1) ):
                    try:
                        if rec[i] is None:
                            strval = 'NA'
                        else:
                            strval = '%.4g' % rec[i]
                    except:
                        # ErrorResult
                        strval = 'NA'

                    val_array.append( strval )

            # print( val_array )
            table.insert_row( val_array )

            color = signals[ row ]
            if color is not None:
                fg = 'black' if color == 'yellow' else 'white'
                table.cells_array[ row+1, QUALITY_SIGNAL ].widget.config( bg=color, fg=fg )

        for col, color in [ [ MONO_RG, 'light blue' ], [ IPI_FLAG, 'yellow' ], [ BICO_FLAG, 'yellow' ] ]:
            cell = table.cells_array[ 3, col ]
            if col == MONO_RG or cell.svar.get() == '1':
                fg = 'white' if color == 'red' else 'black'
                cell.widget.config( bg=color, fg=fg )
