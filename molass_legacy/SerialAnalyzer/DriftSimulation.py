# coding: utf-8
"""
    DriftSimulation.py

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF
"""
import numpy as np
from molass_legacy.KekLib.OurTkinter         import Tk, Dialog
from molass_legacy.KekLib.TkUtils            import is_low_resolution
from CanvasFrame        import CanvasFrame
from DriftAnalyzer      import DriftAnalyzer
from DataModels         import _GuinierPorod

class DriftSimulation:
    def __init__( self, sd, mapper ):
        self.sd     = sd
        self.mapper = mapper
        self.data   = sd.intensity_array
        self.ecurve = mapper.x_curve

    def create_sim_data( self ):

        analyzer = DriftAnalyzer( self.sd, self.mapper )
        drift_params, gp_params = analyzer.solve()
        self.drift_params = drift_params

        if True:
            G_bg    = drift_params['Gbg']
            Rg_bg   = drift_params['Rg']
            d_bg    = drift_params['d']
            D_bg    = drift_params['slope']
            E_bg    = drift_params['intercept']
        else:
            # manually worked values for OA_Ald_Fer
            G_bg, Rg_bg, d_bg   = 0.793, 254, 3.5
            D_bg, E_bg = -0.000107, 0.779

        data    = self.data
        ecurve  = self.ecurve
        print( data.shape )
        num_elutions = data.shape[0]
        x   = data[0,:,0]

        y_sub_bg    = _GuinierPorod( G_bg, Rg_bg, d_bg, x )

        data_list = []
        base_list = []

        start   = 0
        for k, info in enumerate( ecurve.peak_info ):
            if k < len(ecurve.boundaries):
                stop    = ecurve.boundaries[k]
            else:
                stop    = num_elutions

            print( (start, stop) )
            p   = int(info[1]+0.5)
            top = ecurve.y[p]
            G, Rg, d    = gp_params[k]

            for j in range(start, stop):
                s   = ecurve.y[j]/top
                y   = _GuinierPorod( G*s, Rg, d, x )
                G_  = D_bg * j + E_bg
                y_drift = _GuinierPorod( G_, Rg_bg, d_bg, x )
                y_  = y + y_drift
                y_sub   = y_ - y_sub_bg
                data_list.append( np.vstack( [ x, y_sub, data[j,:,2 ] ] ).T )
                base_list.append( y_drift - y_sub_bg )

            if k > 0:
                print( ecurve.boundaries[k-1] )
            print( k, info )
            start = stop

        return np.array( data_list ), np.array( base_list )

def apply_simulated_baseline_correction( drift_params, data, progress_cb=None ):
    x   = data[0,:,0]

    print( 'drift_params=', drift_params )

    Gbg = drift_params['Gbg']
    Rg  = drift_params['Rg']
    d   = drift_params['d']
    slope   = drift_params['slope']
    intercept   = drift_params['intercept']

    num_elutions = data.shape[0]

    stain_bg    = _GuinierPorod( Gbg, Rg, d, x )

    for j in range( num_elutions ):
        G   = slope * j + intercept
        stain = _GuinierPorod( G, Rg, d, x )
        data[j,:,1] += stain_bg - stain
        if progress_cb is not None:
            progress_cb( j )

class BicomponentSimulator( Dialog ):
    def __init__( self, parent ):
        self.parent = parent
        self.x  = np.linspace( 0.003, 0.02, 50 )
        self.xx = self.x**2

    def show( self ):
        title   = "BicomponentSimulator"
        Dialog.__init__( self, self.parent, title )

    def body( self, body_frame ):   # overrides parent class method

        figsize = ( 18, 6 ) if is_low_resolution() else ( 21, 7 )
        self.canvas_frame = canvas_frame = CanvasFrame( body_frame, figsize=figsize )
        canvas_frame.pack()
        fig = canvas_frame.fig
        ax1  = fig.add_subplot( 131 )
        ax2  = fig.add_subplot( 132 )
        ax3  = fig.add_subplot( 133 )
        fig.tight_layout()

        self.draw( ax1, ax2, ax3 )

    def draw( self, ax1, ax2, ax3 ):

        ax1.set_title( "Guinier Plot of Protein with Stain" )

        yb  = _GuinierPorod( 0.01, 300, 3.5, self.x )
        y   = _GuinierPorod( 1.0, 35, 3, self.x )

        ax1.plot( self.xx, np.log( y + yb ),  label='protein + stain' )
        ax1.plot( self.xx, np.log( y ), label='protein'  )
        ax1.plot( self.xx, np.log( yb ), label='stain' )
        ax1.legend()

        ax2.set_title( "Guinier Plot of Protein with Stain" )
        ax2.plot( self.xx, np.log( y + yb ),  label='protein + stain' )
        ax2.plot( self.xx, np.log( y ), label='protein'  )
        ax2.legend()

        ax3.set_title( "Linear Plot of Protein with Stain" )
        ax3.plot( self.x, y + yb,  label='protein + stain' )
        ax3.plot( self.x, y, label='protein'  )
        ax3.legend()

        self.canvas_frame.show()
