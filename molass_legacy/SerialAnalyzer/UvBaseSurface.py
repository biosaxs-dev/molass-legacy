# coding: utf-8
"""
    UvBaseSurface.py

    Copyright (c) 2017-2020, SAXS Team, KEK-PF
"""
import numpy        as np
from molass_legacy.Test.TesterLogger import write_to_tester_log

USE_CONSISTENT_BASE         = True
MIN_USABLE_SIZE_IN_SIGMA    = 3
PENALTY_RATE                = 100
ACCEPTABLE_OFFSSET          = 0.05  #  0.053 for Dec05, 0.07 for SUB_TRN1
SAFE_ENOUGH_POS_SIGMA       = 5

class UvBaseSurface:
    def __init__( self, rail_points, tailcurve, rails, rail_points_sigma, max_y,
                    estimeted_rails_base,
                    rail_points_offset=None ):
        assert len(rails) == 2, 'len(rails) != 2'

        # print( 'slope=', slope )
        self.tailcurve  = tailcurve
        self.rails      = rails

        """
            y = A*x + B
        """
        self.jsize  = rail_points[1] - rail_points[0]
        j   = np.arange( self.jsize + 1 )

        if USE_CONSISTENT_BASE:
            B0  = estimeted_rails_base[0] - tailcurve[0]
            B1  = estimeted_rails_base[1] - tailcurve[-1]
        else:

            rail_points_ok_pre  = [ abs( rail_points_sigma[i] ) >= MIN_USABLE_SIZE_IN_SIGMA  for i in range(2) ]
            if rail_points_offset is None:
                rail_points_offset = [ abs( rail[0] - estimeted_rails_base[i] )/max_y for i, rail in enumerate(rails) ]

            print( 'rail_points_offset=', rail_points_offset )
            rail_points_ok  = [ abs( rail_points_sigma[i] ) - max( 0, rail_points_offset[i] - ACCEPTABLE_OFFSSET )* PENALTY_RATE  >= SAFE_ENOUGH_POS_SIGMA
                                or (  b and rail_points_offset[i] <= ACCEPTABLE_OFFSSET ) for i, b in enumerate(rail_points_ok_pre) ]
            write_to_tester_log( 'rail_points_ok=' + str(rail_points_ok) + '\n' )
            print( 'rail_points_ok=', rail_points_ok )
            print( 'estimeted_rails_base=', estimeted_rails_base )

            if rail_points_ok[0] and rail_points_ok[1]:
                B0  = rails[0][0] - rails[0][-1]
                B1  = rails[1][0] - rails[1][-1]
            elif rail_points_ok[0]:
                B0  = rails[0][0] - rails[0][-1]
                B1  = estimeted_rails_base[1] - rails[1][-1]
            elif rail_points_ok[1]:
                B0  = estimeted_rails_base[0] - rails[0][-1]
                B1  = rails[1][0] - rails[1][-1]
            else:
                B0  = estimeted_rails_base[0] - rails[0][-1]
                B1  = estimeted_rails_base[1] - rails[1][-1]

        intercept = B0
        slope = ( B1 - B0 ) / self.jsize
        self.baseshift  = slope*j + intercept

        if False:
            import molass_legacy.KekLib.DebugPlot as plt
            fig = plt.figure()
            ax = fig.gca()
            ax.plot(self.baseshift)
            ax.plot(self.tailcurve)
            ax.plot(self.tailcurve + self.baseshift)
            fig.tight_layout()
            plt.show()

    def get_basecurve( self ):
        basecurve   = self.tailcurve + self.baseshift
        return basecurve

    def debug_plot( self ):
        pass
