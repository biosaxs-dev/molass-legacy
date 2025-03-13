"""
    Affine.py

    Copyright (c) 2018-2023, Masatsuyo Takahashi, KEK-PF
"""

import numpy                as np

NUM_POINTS  = 3

class Affine:
    def __init__( self, src_points, tgt_points, raise_=False):
        """
        M   [
                [ a, b ]
                [ c, d ]
            ]

        v   [ e, f ]

        p -> q ; q = np.dot( M, p ) + v

        a * p[0] + b * p[1] + c *  0   + d *  0   + e * 1 + f * 0 = q[0]
        a *  0   + b *  0   + c * p[0] + d * p[1] + e * 0 + f * 1 = q[1]

        """
        assert len(src_points) == NUM_POINTS
        assert len(tgt_points) == NUM_POINTS
        A_list  = []
        b_list  = []
        for i, p in enumerate( src_points ):
            p0, p1  = p
            q0, q1  = tgt_points[i]
            #                 a   b   c   d   e   f
            A_list.append( [ p0, p1,  0,  0,  1,  0 ] )
            b_list.append( q0 )
            A_list.append( [  0,  0, p0, p1,  0,  1 ] )
            b_list.append( q1 )

        A   = np.array( A_list )
        b   = np.array( b_list )

        try:
            x   = np.linalg.solve( A, b )
        except Exception as exc:
            import logging
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            logger = logging.getLogger(__name__)
            if raise_:
                msg = "Affine Error: "
            else:
                msg = "later affine.transform(x, y) will return the same x, y for singular cases: "
            log_exception(logger, msg)
            logger.info('src_points=' + str(src_points))
            logger.info('tgt_points=' + str(tgt_points))

            if raise_:
                raise( exc )
            else:
                self.is_singlar = True
                return

        self.is_singlar = False
        self.matrix = np.array( [ [ x[0], x[1] ], [ x[2], x[3] ] ] )
        self.vector = np.array( [ x[4], x[5] ] )
        self.matrix_1 = np.array( [ [ x[0], x[1], x[4] ], [ x[2], x[3], x[5] ] ] )

    def transform_list( self, points ):
        return [ np.dot( self.matrix, point ) + self.vector for point in points ]

    def transform( self, x, y ):
        if self.is_singlar:
            return x, y

        tp = np.dot( self.matrix_1, np.array( [ x, y, np.ones(len(x)) ] ) )
        return tp[0,:], tp[1,:]
