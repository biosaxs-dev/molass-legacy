# coding: utf-8
"""
    PlateauRegions.py

    平坦な領域の判別

    Copyright (c) 2016-2017, Masatsuyo Takahashi, KEK-PF
"""
import numpy    as np

NUM_POINTS_PLATEAU      = 10
NON_PLATEAU_VALUE       = -1
ACCEPTABLE_DIFF_RATE    = 0.2
ACCEPTABLE_RG_DIFF      = 3
SPLIT_TEST_GRAD_RATE    = 0.05
ACCEPTABLE_DEV_RATE     = 0.05
ACCEPTABLE_STD_RATIO    = 0.2
MINIMUM_HALF_POINTS     = 5

DEBUG   = False
continuous_rule10 = np.arange(10)

class PlateauRegions:
    def __init__( self, array, average=None ):
        self.array      = np.array( [ NON_PLATEAU_VALUE if x is None else x for x in array ] )
        self.gradient   = np.gradient( self.array )
        # print( 'self.gradient.shape=', self.gradient.shape )
        plateaus        = np.array( self.array  )
        if average is None:
            acceptable_diff = ACCEPTABLE_RG_DIFF
        else:
            acceptable_diff = average * ACCEPTABLE_DIFF_RATE

        print( 'acceptable_diff=', acceptable_diff )

        plateaus[ np.abs( self.gradient ) > acceptable_diff ] = NON_PLATEAU_VALUE

        size = len(self.array)
        for i in range( size ):
            ok = False
            for j in range( i-NUM_POINTS_PLATEAU+1, i+1 ):
                if j < 0 or j > size - NUM_POINTS_PLATEAU:
                    continue
                if ( plateaus[j:j+NUM_POINTS_PLATEAU] > 0 ).all():
                    ok = True
                    break
            if not ok:
                plateaus[j] = NON_PLATEAU_VALUE

        if DEBUG: print( 'np.where( plateaus > 0 )=',np.where(  plateaus > 0 )[0] )

        def check_region( start, stop ):
            checked_intervals = []
            if stop - start < NUM_POINTS_PLATEAU:
                plateaus[start:stop] = NON_PLATEAU_VALUE
                return checked_intervals

            region  = plateaus[start:stop]
            if DEBUG: print( (start, stop ), 'region=', region[0:5], region[-5:] )
            # pmin, p50, pmax = np.percentile( region, [ 0, 50, 100 ] )
            # print( 'pmin, p50, pmax=', pmin, p50, pmax )

            # check the compactness of primary values

            grad    = self.gradient[start:stop]
            if True: print( 'grad=', grad )
            w40 = start + np.where( grad < -SPLIT_TEST_GRAD_RATE*acceptable_diff )[0]
            w60 = start + np.where( grad > +SPLIT_TEST_GRAD_RATE*acceptable_diff )[0]
            # print( 'w40=', w40 )
            # print( 'w60=', w60 )
            split_candedates = []
            for wnn in [ w40, w60 ]:
                for j in range(0,len(wnn)-10):
                    wj = wnn[j]
                    # print( 'wnn[%d]=%d' % ( j, wj ) )
                    if ( wnn[j:j+10] == wj + continuous_rule10 ).all():
                        for k in range( j+11, len(wnn) ):
                            if not (  k <= len(wnn) and wnn[j:k] == wj + np.arange(k-j) ).all():
                                split_candedates.append( (wnn[j], wnn[k]) )
                                break
                        break

            if True: print( 'split_candedates=', split_candedates )
            if len( split_candedates ) == 1:
                f, t = split_candedates[0]
                if f - start >= NUM_POINTS_PLATEAU and stop - t >= NUM_POINTS_PLATEAU:
                    intervals = [ [ start, f ], [ t, stop ] ]
                    plateaus[f:t]   = NON_PLATEAU_VALUE
                else:
                    intervals = [ [ start, stop ] ]
            else:
                intervals = [ [ start, stop ] ]

            if True: print( 'intervals=', intervals )
            for start_, stop_ in intervals:
                if len( intervals ) == 2:
                    region_ = plateaus[start_:stop_]
                elif len( intervals ) == 1:
                    region_ = region
                else:
                    assert( False )

                if DEBUG: print( (start_, stop_ ), 'region_=',  region_[0:5], region_[-5:] )
                # remove inappropriate points at both ends
                p50 = np.percentile( region_, 50 )
                if DEBUG: print( 'p50=', p50 )
                # print( 'region_ - p50=', region_ - p50 )
                in_appropriate =  np.where( np.abs( region_ - p50 ) > p50 * ACCEPTABLE_DEV_RATE )[0]
                if True: print( 'in_appropriate=', in_appropriate )

                exclude_rule_lower = np.arange( len( in_appropriate ) )
                region_size = len(region_)
                exclude_rule_upper = np.arange( region_size-len( in_appropriate ), region_size )
                interval = [ start_, stop_ ]
                for i, rule in enumerate( [ exclude_rule_lower, exclude_rule_upper ] ):
                    if DEBUG: print( 'rule=',  rule )
                    index = np.where( in_appropriate == rule )[0]
                    if True: print( 'index=',  in_appropriate[index] )
                    if len( index ) > 0:
                        plateaus[ start_ + in_appropriate[index] ] = NON_PLATEAU_VALUE
                        if i == 0:
                            interval[i] = start_ + in_appropriate[ index[-1] ] + 1
                        else:
                            interval[i] = start_ + in_appropriate[ index[0] ]
                    else:
                        # ok. no change
                        pass

                f, t = interval
                if t - f < NUM_POINTS_PLATEAU:
                    ok = False
                else:
                    std = np.std( plateaus[f:t] )
                    std_ratio = std/p50
                    print( (f, t), 'std=', std, 'std_ratio=', std_ratio )
                    if std_ratio <= ACCEPTABLE_STD_RATIO:
                        ok = True
                    else:
                        ok = False

                if ok:
                    checked_intervals.append( interval )
                else:
                    plateaus[f:t] = NON_PLATEAU_VALUE

            print( 'checked_intervals=', checked_intervals )
            return checked_intervals

        def get_regions( y ):
            r = []
            start = None
            for i in range( size ):
                if start is None:
                    if y[i] >= 0:
                        start = i
                else:
                    if y[i] < 0:
                        checked_intervals = check_region( start, i )
                        for interval in checked_intervals:
                            r.append( interval )
                        start = None

            if start is not None:
                checked_intervals = check_region( start, size )
                for interval in checked_intervals:
                    r.append( interval )

            return r

        regions = get_regions( plateaus )

        self.plateaus   = plateaus
        self.regions    = regions

    def is_consistent( self, other ):
        for f, t  in other.regions:
            included = False
            m = ( f + t ) // 2
            for ff, tt in self.regions:
                if ff <= m and m < tt:
                    included = True
                    break
            if not included:
                return False

        return True

    def adjust( self, interval, positive_curve=None ):
        lower, middle, upper = interval
        #  middle  = ( lower + upper ) // 2
        print( 'adjust regions=', self.regions, 'middle=', middle )

        for region in self.regions:
            lb, ub = region
            if upper <= lb or ub <= lower:
            # if lb < middle or middle > ub:
                continue
            else:
                lower = min( lb,        middle - MINIMUM_HALF_POINTS )
                upper = max( ub - 1,    middle + MINIMUM_HALF_POINTS )
                if positive_curve is None:
                    return ( lower, upper )
                else:
                    # print( 'len(self.plateaus)=', len(self.plateaus) )
                    # print( 'len(positive_curve)=', len(positive_curve) )
                    assert( len(positive_curve) == len(self.plateaus) )

                    where_non_positive = np.where( positive_curve[lower:middle] <= 0 )[0]
                    # print( 'lower where_non_positive', where_non_positive )
                    lower_ = None
                    adjusted = False
                    if len( where_non_positive ) == 0:
                        lower_ = lower
                    else:
                        lower_ = lower + where_non_positive[-1] + 1
                        adjusted = True
                        # TODO: where positive_curve[middle] <= 0

                    where_non_positive = np.where( positive_curve[middle:upper] <= 0 )[0]
                    # print( 'upper where_non_positive', where_non_positive )
                    upper_ = None
                    if len( where_non_positive ) == 0:
                        upper_ = upper
                    else:
                        # TODO: untested
                        upper_ = middle + where_non_positive[0] - 1
                        adjusted = True
                        # TODO: where positive_curve[middle-1] <= 0

                    if adjusted:
                        print( 'adjusted by positive_curve: ', ( lower, upper ), '=>', ( lower_, upper_ ) )
                    return ( lower_, upper_ )

        return None
