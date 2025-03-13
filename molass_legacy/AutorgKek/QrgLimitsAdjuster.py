# coding: utf-8
"""
    QrgLimitsAdjuster.py

    Copyright (c) 2016, Masatsuyo Takahashi, KEK-PF
"""

import sys
import traceback
import time
import numpy                as np
from Quality                import fit_consistency_with_stdev

DEBUG = False

class Struct:
    def __init__( self, **entries ): 
        self.__dict__.update(entries)

class ResultEvaluator:
    def __init__( self, result ):
        self.result = result
        self.has_max_qRg = False

    def is_valid( self, qrg_limits ):
        if self.result is None: return False
        if self.result.Rg is None: return False
        self.has_max_qRg = True
        if self.result.max_qRg > qrg_limits[1] :  return False
        return True

    def can_be_improved( self, qrg_limits ):
        if not self.has_max_qRg: return True
        if self.result.max_qRg < qrg_limits[1] - 0.1:
            return True
        else:
            return False

class QrgLimitsAdjuster:
    def __init__( self, fit, guinier, qrg_limits, result, min_num_points=10, max_num_points=40 ):
        self.fit                = fit
        self.f0                 = fit.f0
        self.guinier            = guinier
        self.qrg_limits         = qrg_limits
        self.result             = result
        self.modified           = False
        self.previous_interval  = None
        self.previous_wls       = None
        self.min_num_points     = min_num_points
        self.max_num_points     = max_num_points

        # print( 'QrgLimitsAdjuster: qrg_limits=', qrg_limits )

        if result is None:
            if DEBUG: print( 'QrgLimitsAdjuster: None return' )
            return

        if DEBUG:
            Rg = 0 if result.Rg is None else result.Rg
            print( 'QrgLimitsAdjuster: Rg=%.4g, fit.Rg=%.4g, Rg/fit.Rg=%.4g' % ( Rg, fit.Rg, Rg /fit.Rg ) )

        if ( result.Rg is not None
            and ( result.Aggregated is not None and abs( result.Aggregated ) > 0.5 )
            and result.max_qRg <= qrg_limits[1]
            ):
            # it is better not to try to extend max_qRg when anomalies exist.
            if DEBUG: print( 'QrgLimitsAdjuster: Anomaly return' )
            return

        if result.Rg is None or result.Rg > fit.Rg or result.Rg / fit.Rg > 0.98:
            # Rg / fit.Rg == 0.989 for 20160628/AIMdim01_00231_sub.dat
            # print( 'max_qRg=', result.max_qRg )
            self.try_loop( 'L', self.try_shifting_left )

        else:
            if DEBUG: print('TBD!!!: not ( result.Rg > fit.Rg or result.Rg / fit.Rg > 0.98 ) ')
            self.try_loop( 'R', self.try_extending_right, previous_ok=True )

        """
        try:
            self.minimize_lower_limit()
        except Exception as exc:
            print( exc )
        """

    def try_loop( self, direction, method, previous_ok=False ):
        result = self.result

        if DEBUG:
            print( 'try_loop: result=', result )
            if result is not None:
                print( 'try_loop: result.Rg=', result.Rg, ', result.max_qRg=', result.max_qRg )

        num_trials = self.max_num_points if result.Rg is None else 100

        n = 0

        last_better_result      = None
        last_better_consitency  = None
        evaluator = ResultEvaluator( result )
        if evaluator.is_valid( self.qrg_limits ):
            last_better_result      = result
            consistency             = fit_consistency_with_stdev( self.fit.Rg, result.Rg, result.Rg_stdev )
            last_better_consitency  = consistency

        while ( result is not None
                and ( result.Rg is None and n < num_trials
                    or result.max_qRg is not None
                        and (  result.max_qRg > self.qrg_limits[1]
                            or result.max_qRg < self.qrg_limits[1] - 0.1
                            or (   last_better_consitency is not None
                               and last_better_consitency < 0.7
                               )
                        )
                    )
                ):
            if DEBUG: print( 'try_loop: max_qRg=', result.max_qRg )
            if result.max_qRg is not None:
                stdev_ratio = result.Rg_stdev / result.Rg
                if DEBUG: print( 'stdev_ratio=', stdev_ratio )
                if stdev_ratio < 0.3:
                    if direction == 'L':
                        if result.max_qRg < self.qrg_limits[1] - 0.1:
                            break
                    else:
                        if result.max_qRg > self.qrg_limits[1] + 0.1:
                            break

            try:
                result = method( result )
                evaluator = ResultEvaluator( result )
                if evaluator.is_valid( self.qrg_limits ):
                    consistency = fit_consistency_with_stdev( self.fit.Rg, result.Rg, result.Rg_stdev )
                    if last_better_consitency is None or consistency > last_better_consitency:
                        last_better_result      = result
                        last_better_consitency  = consistency
                        if DEBUG: print( '---- fit.Rg=%.4g, Rg=%.4g, consistency=%.4g' % ( self.fit.Rg, result.Rg, consistency ) )
            except Exception as e:
                if DEBUG: print( 'Exception:', e.args )
                if len( e.args[0] ) != 2:
                    raise Exception( 'len( e.args[0] ) > 2' )
                    break
                    # TODO: do it better.

                f_, t_ = e.args[0][1:]
                result = Struct( Rg=None, max_qRg=None, From=f_, To=t_, result_type=-1, Quality=0 )
                if DEBUG: print( 'result.Rg=', result.Rg, ', result.From=', result.From, ', result.To=', result.To )
            n += 1

        result = last_better_result

        if self.modified and not hasattr( result, 'Quality' ):
            result = self.guinier.get_result( fit=self.fit, result=result )

        if result is not None:
            if DEBUG:
                if result.Rg is not None:
                    print(  'result_type=%s, Rg=%.4g, max_qRg=%.4g, Quality=%.4g'
                                % ( result.result_type, result.Rg, result.max_qRg, result.Quality )
                         )

            if result.result_type is not None and result.result_type >= 0:
                self.result = result

    def try_shifting_left( self, result ):
        f_, t_ = result.From, result.To
        size = t_ - f_ + 1
        if DEBUG: print( 'try_shifting_left: [%d, %d], size=%d' % ( f_, t_, size ) )
        if size < 10:
            f_ = t_ - 9
            if f_ < self.f0:
                return None
            size = t_ - f_ + 1
            assert( size >= self.min_num_points )
        else:
            if f_ > self.f0:
                f_ -= 1
                if size >= self.max_num_points:
                    # has enough points, so don't widen any more
                    t_ -= 1
            else:
                if size > self.min_num_points:
                    t_ -= 1
                else:
                    return None

        self.guinier.estimate_rg( [ f_, t_ ] )
        if self.guinier.wls.Rg is None:
            # restore the previous state
            self.guinier.estimate_rg( [ result.From, result.To ] )
            raise Exception( [ 'failed to calculate Rg', f_, t_ ] )

        Rg = self.guinier.wls.Rg
        Rg_stdev = self.guinier.wls.sigmaRg
        max_qRg = np.sqrt( self.guinier.x[t_] ) * Rg
        if DEBUG: print( 'try_shifting_left: [%d, %d], max_qRg=%.4g' % ( f_, t_, max_qRg ) )
        self.modified   = True
        result_type_ = self.result.result_type * 10 + 8
        return Struct( Rg=Rg, Rg_stdev=Rg_stdev, From=f_, To=t_, wls=self.guinier.wls, max_qRg=max_qRg, result_type=result_type_ )

    def try_extending_right( self, result ):
        f_, t_ = result.From, result.To
        size = t_ - f_ + 1
        if DEBUG: print( 'try_extending_right: [%d, %d], size=%d' % ( f_, t_, size ) )
        if size < 10:
            t_ = f_ + 9
            if t_ > self.fit.t1:
                return None
            size = t_ - f_ + 1
            assert( size >= self.min_num_points )
        else:
            if t_ < self.fit.t1 + self.min_num_points:
                if size >= self.max_num_points:
                    # has enough points, so don't widen any more
                    f_ += 1
                t_ += 1
            else:
                if f_ < self.fit.t1 and size > self.min_num_points:
                    f_ += 1
                else:
                    return None

        self.guinier.estimate_rg( [ f_, t_ ] )
        if self.guinier.wls.Rg is None:
            # restore the previous state
            self.guinier.estimate_rg( [ result.From, result.To ] )
            raise Exception( [ 'failed to calculate Rg', f_, t_ ] )

        Rg = self.guinier.wls.Rg
        Rg_stdev = self.guinier.wls.sigmaRg
        max_qRg = np.sqrt( self.guinier.x[t_] ) * Rg
        if DEBUG: print( 'try_extending_right: [%d, %d], max_qRg=%.4g' % ( f_, t_, max_qRg ) )
        self.modified   = True
        result_type_ = self.result.result_type * 10 + 9
        return Struct( Rg=Rg, Rg_stdev=Rg_stdev, From=f_, To=t_, wls=self.guinier.wls, max_qRg=max_qRg, result_type=result_type_ )

    def minimize_lower_limit( self ):
        print( 'minimize_lower_limit' )
        Rg  = self.result.Rg
        f   = self.result.From
        t   = self.result.To
        min_i = None
        start = time.time()
        for i in range( f ):
            try:
                self.guinier.estimate_rg( [ i, t ] )
                Rg_ = self.guinier.wls.Rg
                if Rg_ is None:
                    continue
                if abs( Rg_/Rg - 1 ) < 0.01:
                    min_i = i
                    break
            except:
                continue
        print( 'f0, f, min_i=', self.f0, f, min_i, time.time() - start )

    def get_adjusted_result( self ):
        if self.result is None:
            return None
        else:
            return self.result
            """
            if self.modified:
                result_type = self.result.result_type * 10 + 7
                return self.guinier.get_result( self.fit, result_type=result_type )
            else:
                # TODO: why is this different from the above?
                return self.result
            """
