"""
    ExtrapolationSolver.py

    Copyright (c) 2018-2025, SAXS Team, KEK-PF
"""
import os
import copy
import time
import numpy as np
import logging
from scipy import optimize
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.SerialAnalyzer.DevSettings import get_dev_setting
from molass_legacy.DataStructure.SvdDenoise import get_denoised_data, get_denoised_error
from .KnownPeakPenalty import compute_known_error_impl, compute_known_error_grad_impl
from molass_legacy.Conc.ConcDepend import compute_distinct_cd
from molass_legacy.LRF.LrfInfo import LrfInfo
import molass_legacy.KekLib.DebugPlot as plt

# SA_DEBUGGING = os.environ.get('SA_DEBUGGING', None)
SA_DEBUGGING = False

DEBUG   = False
DEBUG_WEIGHT_MATRIX = False
DEBUG_LRF_RANK = False

USE_TILTED_MODEL        = True
USE_PENALTY_PEAK_ERROR  = False
USE_LOGARITHMIC_RATIO   = False
LOGARITHMIC_RATIO       = 0.5
VERY_SMALL_VALUE        = 1e-5
BASE_ERROR_NORM_DEGREE  = 2
assert BASE_ERROR_NORM_DEGREE == 2
MAX_ITERATION_FMIN_CG   = 10000
USE_UNIFORMLY_SCALED_UV = True
ERROR_PROPAGATION_DENOISE   = False
DISABLE_IGNORE_BQ_LIST  = True
USE_DICTINCT_CD         = True

class KnownInfo:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return str(self.data.shape)

class ExtrapolationSolver:
    def __init__( self, pdata, popts):
        self.rank_control = get_setting('rank_control')
        self.concentration_datatype = get_setting('concentration_datatype')

        self.judge_holder = pdata.judge_holder
        self.mapper = pdata.mapper          # self.mapper will be used in ElutionMatrix

        if pdata.is_for_sec:
            ecurve = self.mapper.x_curve
            self.sd = sd = pdata.sd
            self.qv = sd.qvector
            self.Rg = sd.pre_recog.get_rg()
            data = sd.intensity_array[:,:,1].T
            error = sd.intensity_array[:,:,2].T
            mc_vector = sd.mc_vector
            conc = None
            self.sd_id_info = sd.get_id_info()
            self.cd_slice = sd.get_cd_slice()
            self.xray_scale = sd.get_xray_scale()
        else:
            xdata = pdata.xdata
            self.sd = None
            self.qv = xdata.vector
            self.Rg = None
            ecurve = xdata.e_curve
            data = xdata.data
            error = xdata.error.data
            mc_vector = None
            conc = pdata.decomp_info.conc
            self.sd_id_info = None
            self.rank_control = 0   # not yet supported
            self.cd_slice = None
            self.xray_scale = None

        print( 'data.shape=', data.shape )

        if SA_DEBUGGING:
            from .ExtrapolationDebugger import ExtrapolationDebugger
            self.debugger = ExtrapolationDebugger()
        else:
            self.debugger = None

        self.logger = logging.getLogger( __name__ )
        self.ecurve = ecurve
        self.data = data
        self.error = error
        self.mc_vector = mc_vector  # already scaled with conc_factor
        self.cnv_ranges = pdata.cnv_ranges
        self.num_ranges = pdata.num_ranges
        self.conc_factor = pdata.conc_factor
        rg_str = "None" if self.Rg is None else "%.3g" % self.Rg
        self.logger.info('lrf solver has been constructed with conc_factor %g, Rg=%s.', self.conc_factor, rg_str)
        self.DEFAULT_WEIGHTS = get_setting('penalty_weighting')

        self.popts = popts
        self.aq_smoothness = popts.aq_smoothness
        self.aq_positivity = popts.aq_positivity
        self.use_elution_models = popts.use_elution_models
        self.conc = conc
        self.add_conc_const = get_dev_setting('add_conc_const')
        self.conc_dependence = get_setting('conc_dependence')
        self.doing_sec = self.conc is None
        if self.doing_sec:
            self.conc_depend = self.conc_dependence
        else:
            denat_dependent = get_setting('denat_dependent')
            self.conc_depend = 2 if denat_dependent else 1

        self.j0 = pdata.sd.xr_j0

        self.synthesized_lrf = get_setting('synthesized_lrf')
        self.logger.info('conc_depend has been set to %d' % self.conc_depend)

    def extrapolate_wiser( self, start, stop, peakset_info,
                            ignore_bq=0,
                            penalty_weights=None,
                            stop_event=None,
                            animation=False,
                            demo_info=None,
                            surplus_rank=0,
                            suppress_rank_control=False,
                            lrf_rank=None,
                            conc_depend=None,
                            save_cmatrix=False,
                            debug=False,
                            ):
        if debug:
            from importlib import reload
            import molass_legacy.LRF.ConcMatrix
            reload(molass_legacy.LRF.ConcMatrix)       
        from molass_legacy.LRF.ConcMatrix import ConcMatrix

        paired_ranges = self.cnv_ranges

        if penalty_weights is None:
            penalty_weights = copy.deepcopy(self.DEFAULT_WEIGHTS)

        self.penalty_weights = np.array(penalty_weights)
        self.stop_event = stop_event
        self.animation = animation
        self.demo_info = demo_info
        if animation:
            self.anim_data1 = []
            self.anim_data2 = []
            self.anim_counter = 0

        pno, nth, peakset, known_peak_info = peakset_info
        """
        NOTE:
            known_peak_info has been later added to [pno, nth, peakset],
            which has been appended to PeakSetSelector.required_peakset_infos.
            this differece in peakset_info, i.e., the difference between
            [pno, nth, peakset] and [pno, nth, peakset, known_peak_info],
            can be confusing.
        """
        paired_ranges_ = [paired_ranges[i] for i in peakset]
        # print('self.use_elution_models=', self.use_elution_models)
        # print('paired_ranges_=', paired_ranges_, 'ignore_bq=', ignore_bq)

        x = self.ecurve.x[start:stop]
        if self.mc_vector is None:
            mc_vector = None
        else:
            mc_vector = self.mc_vector[start:stop]

        if suppress_rank_control:
            rank_control = 0
        else:
            rank_control = self.rank_control

        if conc_depend is None:
            if rank_control:
                if lrf_rank is None:
                    if USE_DICTINCT_CD:
                        cmatrix = ConcMatrix(x, self.conc, conc_depend=1,
                                                paired_ranges=paired_ranges_, mc_vector=mc_vector,
                                                conc_factor=self.conc_factor,
                                                ecurve=self.ecurve, j0=self.j0,
                                                )
                        M = self.data[:,start:stop]
                        conc_depend = compute_distinct_cd(M, cmatrix.data, self.cd_slice, self.xray_scale, self.logger)
                    else:
                        conc_depend = self.judge_holder.get_cd_degree_from_range(start, stop)
                else:
                    if lrf_rank == 1:
                        conc_depend = 1
                    else:
                        conc_depend = 2
            else:
                conc_depend = self.conc_depend

        self.ignore_bq = 1 if conc_depend < 2 else 0

        cmatrix = ConcMatrix(x, self.conc, conc_depend=conc_depend,
                                paired_ranges=paired_ranges_, mc_vector=mc_vector,
                                conc_factor=self.conc_factor,
                                ecurve=self.ecurve, j0=self.j0,
                                )

        self.logger.info('pno=%d, nth=%d, cdl_list=%s, use_elution_models=%s', pno, nth, str(cmatrix.cdl_list), str(self.use_elution_models))
        if not self.use_elution_models:
            if nth > 0:
                nth = 0
                self.logger.info("nth has been changed to %d as a temporary fix in non-elution-model mode.", nth)

        C_ = cmatrix.data
        if save_cmatrix:
            np.savetxt("C-%d-%d.dat" % (start, stop), C_)
        self.ret_C = C_

        D = self.data[:,start:stop]
        E = self.error[:,start:stop]

        rank = C_.shape[0] + surplus_rank
        if not rank_control:
            if get_setting('allow_rank_variation'):
                rank_increment = get_setting('rank_increment')
                rank += rank_increment
                self.logger.info('rank incremented by %d to %d.' % (rank_increment, rank))

        if rank_control:
            svd_reconstruct = 1
        else:
            svd_reconstruct = get_setting('svd_reconstruct')

        j0 = self.j0
        synthesized_solution = False
        boundary = None
        if svd_reconstruct:
            if self.synthesized_lrf and rank > 1 and conc_depend > 1:
                if rank == 2 and conc_depend == 2:
                    from Theory.Rg import compute_corrected_Rg
                    from Theory.SynthesizedLRF import synthesized_lrf, get_reduced_conc
                    Rg, gf, gt, _ = compute_corrected_Rg(self.sd, self.ecurve, pno, self.qv, D, E, range_=(start, stop))
                    Cred = get_reduced_conc(C_, cmatrix.cdl_list)
                    red_rank = len(Cred)
                    P, opt_info, D_, boundary = synthesized_lrf(self.qv, D, C_, rank, Cred, Rg)
                    synthesized_solution = True
                else:
                    from Rank.Synthesized import get_reduced_rank, synthesized_data
                    D_, boundary  = synthesized_data(self.qv, D, self.Rg, rank=rank, cd=conc_depend)
                    red_rank = get_reduced_rank(rank, conc_depend)
                rank_phrase = 'with ranks=(%d, %d)' % (rank, red_rank)
                by_phrase = ' by synthesized LRF'
            else:
                D_ = get_denoised_data(D, rank=rank)
                rank_phrase = 'with rank=%d' % rank
                by_phrase = ''
            self.logger.info('SVD-reconstructed %s in range [%d, %d)%s.', rank_phrase, j0+start, j0+stop, by_phrase)

        else:
            D_ = D
            self.logger.info('skipping SVD-reconstruction in range [%d, %d).', j0+start, j0+stop)

        weight_matrix_type = get_setting('weight_matrix_type')
        if weight_matrix_type == 1:
            from .WeightedLRF import compute_reciprocally_weighted_matrices
            Cr, Dr = compute_reciprocally_weighted_matrices(self.conc_dependence, C_, D_)
            self.logger.info("using concentration-reciprocal weights.")
            if DEBUG_WEIGHT_MATRIX:
                from molass_legacy.DataStructure.MatrixData import simple_plot_3d
                fig = plt.figure(figsize=(21,7))
                ax1 = fig.add_subplot(131, projection='3d')
                ax2 = fig.add_subplot(132, projection='3d')
                ax3 = fig.add_subplot(133)
                simple_plot_3d(ax1, D_)
                simple_plot_3d(ax2, Dr)
                ax3.plot(Cr[0,:])
                ax3.plot(Cr[1,:])
                fig.tight_layout()
                plt.show()
            C_, D_ = Cr, Dr

        self.C_ = C_

        if synthesized_solution:
            pass
        else:
            try:
                Cpinv = np.linalg.pinv( C_ )
            except Exception as exc:
                from molass_legacy.KekLib.ExceptionTracebacker import log_exception
                log_exception(self.logger, "pinv failed: ")
                # np.savetxt("C-%d-%d.dat" % (start, stop), C_)
                raise exc

            if ERROR_PROPAGATION_DENOISE:
                E_  = get_denoised_error( D_, D, E )

            self.D  = D_
            self.P_init = np.dot( D_, Cpinv )

            if demo_info is not None:
                self.demo_data = []
                # self.P_init[:,0] += np.random.uniform(size=D.shape[0]) * 0.01
                self.P_init[demo_info[0],0] += 0.002
                self.P_init[demo_info[1],0] -= 0.002
                # self.demo_data.append(self.P_init[demo_info,0])

            known_log_str = '' if known_peak_info is None else ' and known ' + str(known_peak_info)
            self.logger.info( 'solving rank %d factorization of M into P, C with shapes %s = %s @ %s%s'
                                 % ( rank, str(self.D.shape), str(self.P_init.shape), str(C_.shape), known_log_str ) )

            self.set_known_peak_info(known_peak_info, conc_depend)

            P, opt_info = self.fit()

        opt_info += [(nth, cmatrix.cdl_list), self.qv, boundary, D_]

        if self.ignore_bq:
            A   = P[:,nth]
            B   = np.zeros(P.shape[0])
            Z   = B
        else:
            A   = P[:,nth*conc_depend]
            B   = P[:,nth*conc_depend+1]
            if conc_depend > 2:
                Z   = P[:,nth*conc_depend+2]
            else:
                Z   = np.zeros(P.shape[0])

        if self.concentration_datatype < 2:
            j = paired_ranges_[nth].top_x
            top_c = self.mc_vector[j]
            top_y = self.ecurve.y[j]/top_c
            i = self.sd.xray_index
            ab_scale = top_y/A[i]
            A *= ab_scale
            B *= ab_scale**2    # ok?
            Z *= ab_scale**3    # ok?

        if ERROR_PROPAGATION_DENOISE:
            Ae, Be, Ze = self.compute_propagated_errors_from_denoised_error( P, D_, E_, nth, self.ignore_bq )
        else:
            Ae, Be, Ze = self.compute_propagated_errors( P, D, E, nth, self.ignore_bq )

        if self.concentration_datatype < 2:
            Ae *= ab_scale
            Be *= ab_scale**2   # ok?
            Ze *= ab_scale**3   # ok?

        if DEBUG_LRF_RANK:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21,7))
            C = self.ret_C
            M = P@C
            svd_results = []
            s_list = []
            for matrix in [P, C, M]:
                U, s, VT = np.linalg.svd(matrix)
                svd_results.append(s)
                s_list.append(s[0])

            ymax = np.max(s_list)*1.1
            for s, ax, name in zip(svd_results, axes, ['$P$', '$C$', r'$P \cdot C$']):
                ax.set_title('Singular Values of ' + name, fontsize=30)
                ax.plot(s[0:5], ':', marker='o', markersize=10)
                w = np.where(s>0.2)[0]
                print('s=', s)
                print('w=', w)
                rank = len(w)
                ax.text(2, ymax*0.7, name, fontsize=100, alpha=0.2, ha='center', va='center')
                ax.text(2, ymax*0.4, "Rank %d" % rank, fontsize=100, alpha=0.2, ha='center', va='center')
                ax.set_xlim(-0.5, 4.5)
                ax.set_ylim(-0.1, ymax)
            fig.tight_layout()
            plt.show()

        if self.debugger is not None:
            from collections import OrderedDict
            param_info = OrderedDict()
            param_info['rank'] = rank
            param_info['popts'] = str(self.popts)
            param_info['sd_id_info'] = self.sd_id_info
            self.debugger.save_info(D, C_, A, param_info)

        lrfE = [Ae, Be, Ze]
        return A, B, Z, lrfE, LrfInfo(opt_info, A, lrfE), self.ret_C

    def set_known_peak_info(self, known_peak_info, conc_depend):
        if known_peak_info is None:
            self.solver_known_info = known_peak_info
        else:
            self.solver_known_info = []
            for k, info in enumerate(known_peak_info):
                if info is None:
                    data = None
                else:
                    data = info.data
                    n = k*conc_depend
                    self.P_init[:,n] = data[:,1]
                self.solver_known_info.append(data)

    def get_anim_data( self ):
        return self.anim_data1, self.anim_data2

    def compute_propagated_errors_from_denoised_error( self, P, D_, E_, nth, ignore_bq ):
        D_pinv = np.linalg.pinv(D_)
        W   = np.dot(D_pinv, P)
        Pe  = np.sqrt( np.dot(E_**2, W**2) )
        if ignore_bq:
            Ae  = Pe[:,nth]
            Be  = np.zeros(P.shape[0])
            Ce  = Be
        else:
            Ae  = Pe[:,nth*self.conc_depend]
            Be  = Pe[:,nth*self.conc_depend+1]
            if self.conc_depend < 3:
                Ce  = np.zeros(P.shape[0])
            else:
                Ce  = Pe[:,nth*self.conc_depend+2]
        return Ae, Be, Ce

    def compute_propagated_errors( self, P, M, E, nth, ignore_bq ):
        M_pinv = np.linalg.pinv(M)
        W   = np.dot(M_pinv, P)
        Pe  = np.sqrt( np.dot(E**2, W**2) )
        if ignore_bq:
            Ae  = Pe[:,nth]
            Be  = np.zeros(P.shape[0])
            Ce  = Be
        else:
            Ae  = Pe[:,nth*self.conc_depend]
            Be  = Pe[:,nth*self.conc_depend+1]
            if self.conc_depend < 3:
                Ce  = np.zeros(P.shape[0])
            else:
                Ce  = Pe[:,nth*self.conc_depend+2]
        return Ae, Be, Ce

    def compute_error( self, P_flat ):
        if self.stop_event is not None and self.stop_event.is_set():
            raise Exception( 'Stop Event is set' )

        P = P_flat.reshape( self.P_init.shape )
        reconstructed = np.dot( P, self.C_ )

        # compute a squared Frobenius norms
        rec_error = np.linalg.norm( reconstructed - self.D )**2

        if self.aq_positivity:
            A   = np.dot( P, self.P2A )
            negA = np.min( [ self.Zero, A ], axis=0 )
            neg_error = np.linalg.norm( negA )**2
        else:
            neg_error = 0

        if self.add_conc_const:
            base_error = np.linalg.norm( np.dot( P, self.P2E ) )**BASE_ERROR_NORM_DEGREE
        else:
            base_error = 0

        if self.solver_known_info is None:
            known_error = 0
        else:
            known_error = self.compute_known_error(P)

        if self.animation:
            self.anim_counter += 1
            anim_rec1 = [ self.anim_counter, P ]
            anim_rec2 = [ rec_error,
                            neg_error*self.penalty_weights[0],
                            base_error*self.penalty_weights[1] ]

        if self.demo_info is not None:
            self.demo_data.append(P[self.demo_info,0])

        # print( 'ratio=', base_error/rec_error )
        error   = rec_error + neg_error*self.penalty_weights[0] + base_error*self.penalty_weights[1] + known_error

        if self.aq_smoothness:
            if self.ignore_bq:
                penalties = self.sp.get_penalties_bq_ignore(P_flat, self.penalty_weights[2])
                error += np.sum(penalties)
                if self.animation:
                    aq_penalty_sum = np.sum(penalties)
                    anim_rec2 += [ aq_penalty_sum, 0 ]
            else:
                penalties = self.sp.get_penalties(P_flat, self.penalty_weights[2:])
                error += np.sum(penalties)
                if self.animation:
                    aq_penalty_sum = np.sum( penalties[0::2] )
                    bq_panalty_sum = np.sum( penalties[1::2] )
                    anim_rec2 += [ aq_penalty_sum, bq_panalty_sum ]

        if self.animation:
            self.last_anim_rec1 = anim_rec1
            self.last_anim_rec2 = anim_rec2
            nth = max(0, int(np.log10(self.anim_counter+0.5))-1)
            self.anim_pick_data = self.anim_counter % (10 ** nth) == 0
            if self.anim_pick_data:
                self.anim_data1.append( anim_rec1 )
                self.anim_data2.append( anim_rec2 )

        return error

    def compute_known_error(self, P):
        return compute_known_error_impl(self.conc_depend, self.solver_known_info, P)

    def compute_error_grad( self, P_flat ):
        P = P_flat.reshape( self.P_init.shape )
        reconstructed = np.dot( P, self.C_ )

        # compute the derivative of the squared Frobenius norm
        # || P@C - D ||**2 ==> 2*( P@C - D )@C.T
        rec_grad = 2 * np.dot( reconstructed - self.D, self.C_.T ).flatten()

        if self.aq_positivity:
            A   = np.dot( P, self.P2A )
            negA = np.min( [ self.Zero, A ], axis=0 )
            neg_grad = 2 * negA.flatten()
        else:
            neg_grad = 0

        if self.add_conc_const:
            base_grad = 2 * np.dot( np.dot( P, self.P2E ), self.P2E.T ).flatten()
        else:
            base_grad = 0

        if self.solver_known_info is None:
            known_error_grad = 0
        else:
            known_error_grad = self.compute_known_error_grad(P)

        error_grad  = rec_grad + neg_grad * self.penalty_weights[0] + base_grad * self.penalty_weights[1] + known_error_grad

        if self.aq_smoothness:
            if self.ignore_bq:
                error_grad += self.sp.get_penalty_diff_bq_ignore(P_flat, self.penalty_weights[2])
            else:
                error_grad += self.sp.get_penalty_diff(P_flat, self.penalty_weights[2:])

        return error_grad

    def compute_known_error_grad(self, P):
        return compute_known_error_grad_impl(self.conc_depend, self.solver_known_info, P)

    def fit( self ):
        t0 = time.time()
        # return self.P_init

        if self.aq_smoothness:
            from SmoothnessPenalty import SmoothnessPenalty
            self.sp = SmoothnessPenalty(self.P_init.shape[0], self.conc_depend, add_const=self.add_conc_const)

        self.Zero = np.zeros( self.P_init.shape )
        n = self.P_init.shape[1]
        self.P2A = np.zeros( (n, n) )
        for i in range(0, n, 2):
            self.P2A[i,i] = 1

        self.P2E = np.zeros( (n, n) )
        self.P2E[n-1,n-1] = 1

        self.num_iterations = 0
        def num_iterations_count(x):
            self.num_iterations += 1

        ret = optimize.fmin_cg( self.compute_error, self.P_init.flatten(), fprime=self.compute_error_grad,
                maxiter=MAX_ITERATION_FMIN_CG,
                full_output=True,
                callback=num_iterations_count )
        P_flat = ret[0]
        func_calls = ret[2]
        self.P_fit = P = P_flat.reshape( self.P_init.shape )

        if self.animation:
            self.anim_data1.append( self.last_anim_rec1 )
            self.anim_data2.append( self.last_anim_rec2 )

        t1 = time.time()
        time_elapsed = t1 - t0
        print( 'It took ', time_elapsed, 'num_iterations=', self.num_iterations )

        self.logger.info('solved with aq_smoothness=%s, aq_positivity=%s resulting into iterations=%d',
                            str(self.aq_smoothness), str(self.aq_positivity), self.num_iterations)

        return P, [self.num_iterations, func_calls, time_elapsed,]

def run_extrapolation_solver_impl(
        solver, selector, conc_depend=None,
        exe_queue=None, stop_event=None ):

    ignore_all_bqs = get_setting('ignore_all_bqs')
    if DISABLE_IGNORE_BQ_LIST:
        ignore_bq_list = None
    else:
        ignore_bq_list = get_setting('ignore_bq_list')

    results = []
    peak_range_infos = []
    row = 0
    if exe_queue is not None:
        exe_queue.put( [0, row ] )
        if stop_event.is_set():
            return

    # previous_penalty_matrix = get_setting('zx_penalty_matrix')
    # TODO: re-consider on previous_penalty_matrix
    previous_penalty_matrix = None

    for pno, paired_range in enumerate(solver.cnv_ranges):
        fromto_list = paired_range.get_fromto_list()
        for rno, range_ in enumerate(fromto_list):
            start   = range_[0]
            stop    = range_[1] + 1
            peakset_info = selector.select_peakset(row)
            penalty_weights = None if previous_penalty_matrix is None else previous_penalty_matrix[row]
            if ignore_bq_list is None:
                ignore_bq = ignore_all_bqs
            else:
                ignore_bq = ignore_bq_list[row]
            try:
                A, B, Z, E, lrf_info, C = solver.extrapolate_wiser( start, stop, peakset_info,
                                        stop_event=stop_event,
                                        penalty_weights=penalty_weights,
                                        ignore_bq=ignore_bq,
                                        conc_depend=conc_depend,
                                        )
            except:
                from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
                etb = ExceptionTracebacker()
                print(etb)
                if exe_queue is not None:
                    exe_queue.put( [-1, str(etb) ] )
                raise RuntimeError('in the solver')

            if stop_event is not None and stop_event.is_set():
                print( "solver stopped by stop_event." )
                return None

            results.append( [A, B, Z, E, lrf_info, C] )
            peak_range_infos.append( [pno, rno, paired_range] )
            row += 1
            if exe_queue is not None:
                exe_queue.put( [0, row ] )

    print( "solver finished." )
    return results, peak_range_infos

def run_extrapolation_solver_with_progress(
        solver, selector, parent, conc_depend=None):

    import queue
    import threading
    from ProgressMinDialog import ProgressMinDialog
    from molass_legacy.KekLib.TkUtils import split_geometry

    w, h, x, y = split_geometry( parent.winfo_geometry() )
    geometry_info = "+%d+%d" % (parent.winfo_rootx() + int(w*0.7), parent.winfo_rooty() + int(h*0.8))

    exe_queue = queue.Queue()
    max_iter = len( selector.required_peakset_infos ) + 1

    stop_event = threading.Event()
    def proc():
        ret_tuple = run_extrapolation_solver_impl(
            solver, selector,conc_depend=conc_depend,
            exe_queue=exe_queue, stop_event=stop_event )
        exe_queue.put( [1, ret_tuple] )

    thread = threading.Thread( target=proc, name='PreviewExtrapolationgThread', args=[] )
    thread.start()

    progress_dialog = [None]

    counter = [0]
    ret_info = [None]
    def progress_cb():
        error_info = None
        try:
            info = exe_queue.get(block=False)
            print('progress_cb:', info[0])
            if info[0] == 1:
                ret_info[0] = info[1]
                counter[0] = max_iter + 1
            elif info[0] == 0:
                counter[0] = info[1] + 1
            else:
                # get exception info
                error_info = info[1]
        except:
            pass

        if error_info is not None:
            pdialog = progress_dialog[0]
            pdialog.cancel()
            thread.join()
            # raise the exception here in the parent thread
            # which occrured in the child thread
            raise RuntimeError(error_info)
            """
                this will be caught in Tk with the overriden repoter
                GuiMain.report_callback_exception if this code is running
                under GuiMain
            """

        return counter[0]

    progress = ProgressMinDialog(parent,
                    title="Preview Extrapolation with Smoothness Constraints",
                    message="It may take several minutes. Please be patient.",
                    num_steps=max_iter + 1, length=360,
                    progress_cb=progress_cb, geometry_info=geometry_info,
                    cancelable=True, visible=False )

    progress_dialog[0] = progress
    progress.show()

    if progress.canceled:
        stop_event.set()

    thread.join()

    ret_tuple = ret_info[0]
    return ret_tuple

def run_extrapolation_solver(
        solver, selector, conc_depend=None, parent=None, use_gui=True ):

    if parent is None or not use_gui:
        ret_tuple = run_extrapolation_solver_impl(
            solver, selector, conc_depend=conc_depend)
    else:
        ret_tuple = run_extrapolation_solver_with_progress(
            solver, selector, parent, conc_depend=conc_depend)

    return ret_tuple
