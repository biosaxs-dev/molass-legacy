"""
    LrfResultPool.py

    Copyright (c) 2021-2025, SAXS Team, KEK-PF
"""
import logging
from molass_legacy._MOLASS.SerialSettings import get_setting

LRF_INFO = 4
USE_LRF_SOLVER = True

class LrfResultPool:
    def __init__(self, pdata, popts):
        self.logger = logging.getLogger(__name__)
        self.lrf_bound_correction = get_setting("lrf_bound_correction")
        self.pdata = pdata
        self.popts = popts
        self.rank_control = get_setting('rank_control')
        self.conc_dependence = get_setting('conc_dependence')

    def run_solver(self, parent=None, debug=False):
        if debug:
            from importlib import reload
            import molass_legacy.Extrapolation.PeakSetSelector
            reload(molass_legacy.Extrapolation.PeakSetSelector)  
        from molass_legacy.Extrapolation.PeakSetSelector import PeakSetSelector

        if USE_LRF_SOLVER:
            if debug:
                from importlib import reload
                import molass_legacy.LRF.LrfSolver
                reload(molass_legacy.LRF.LrfSolver)       
            from .LrfSolver import LrfSolver as Solver
        else:
            from molass_legacy.Extrapolation.ExtrapolationSolver import ExtrapolationSolver as Solver

        self.logger.info('cnv_ranges=%s, num_ranges=%s', str(self.pdata.cnv_ranges), str(self.pdata.num_ranges))

        self.solver_results = None
        self.solver = Solver(self.pdata, self.popts)
        self.pno_map = self.solver.get_pno_map()
        self.selector = PeakSetSelector(self.pdata.cnv_ranges, self.pdata.mapper.x_curve, debug=debug)

        if self.rank_control:
            ret_tuple = run_lrf_solver_impl(self.solver, self.selector, lrf_rank=1)
            self.rank1_results = ret_tuple[0]
            ret_tuple = run_lrf_solver_impl(self.solver, self.selector, lrf_rank=2)
            self.rank2_results = ret_tuple[0]
        else:
            lrf_rank = self.conc_dependence
            ret_tuple = run_lrf_solver_impl(self.solver, self.selector, lrf_rank=lrf_rank)
            self.rankx_results = ret_tuple[0]

        self.peak_range_infos = ret_tuple[1]
        self.get_better_results()

    def ok( self ):
        return self.solver_results is not None

    def show_dialog(self, parent, editor_frame=None, last_change_id=None, debug=True):
        if debug:
            from importlib import reload
            import molass_legacy.Extrapolation.ExtrapolSolverDialog
            reload(molass_legacy.Extrapolation.ExtrapolSolverDialog)
        from molass_legacy.Extrapolation.ExtrapolSolverDialog import ExtrapolSolverDialog
        self.dialog = ExtrapolSolverDialog(parent, self, editor_frame=editor_frame, last_change_id=last_change_id)
        self.dialog.show()

    def get_dialog( self ):
        return self.dialog

    def get_solver_ret_tuple( self ):
        return self.solver_results, self.peak_range_infos

    def get_better_results(self):

        if self.solver_results is None:
            if self.rank_control:
                rank_args = self.get_better_rank_args()
                solver_results = []

                for rank_pair, rank1_result, rank2_result in zip(rank_args, self.rank1_results, self.rank2_results):
                    if rank_pair == 1:
                        solver_results.append(rank1_result)
                    else:
                        solver_results.append(self.synthesize(rank2_result, rank1_result))
            else:
                solver_results = self.rankx_results

            self.solver_results = solver_results

        return self.solver_results

    def get_better_rank_args(self):
        self.peak_judgements = []
        rank_args = []
        for k, range_info in enumerate(self.peak_range_infos):
            num_ad_ranges = len(range_info[2].get_fromto_list())
            print([k], range_info, num_ad_ranges)
            if num_ad_ranges == 1:
                rank_args.append(self.get_minor_rank_info(k, range_info))
            else:
                rank_args.append(self.get_major_rank_info(k, range_info))
        return rank_args

    def get_minor_rank_info(self, k, range_info):
        pno = range_info[0]
        if pno < len(self.peak_judgements):
            pass
        else:
            self.peak_judgements.append(1)
        return self.peak_judgements[pno]

    def get_major_rank_info(self, k, range_info):
        pno, ad = range_info[0:2]
        if pno < len(self.peak_judgements):
            assert ad == 1
        else:
            assert ad == 0

            asc_rank1_result = self.rank1_results[k]
            dsc_rank1_result = self.rank1_results[k+1]
            asc_rank2_result = self.rank2_results[k]
            dsc_rank2_result = self.rank2_results[k+1]

            if self.lrf_bound_correction or True:
                color = self.pno_map.get_color(pno)
                lrf_rank = 2 if color in ["yellow", "red"] else 1
                self.logger.info("[%d-%d] color=%s, lrf_rank=%d", pno, k, color, lrf_rank)
            else:
                asc_Rg1 = asc_rank1_result[LRF_INFO].Rg
                dsc_Rg1 = dsc_rank1_result[LRF_INFO].Rg
                asc_Rg2 = asc_rank2_result[LRF_INFO].Rg
                dsc_Rg2 = dsc_rank2_result[LRF_INFO].Rg

                rg_list = tuple([str(rg) if rg is None else "%.3g" % rg for rg in [asc_Rg1, dsc_Rg1, asc_Rg2, dsc_Rg2]])
                print([k], "(%s, %s), (%s, %s)" % rg_list)

                def judge_as_same(rg1, rg2, allow=0.02):
                    if rg1 is None or rg2 is None:
                        return False

                    ratio = abs((rg1-rg2)/(rg1+rg2))
                    return ratio < allow

                if judge_as_same(asc_Rg1, asc_Rg2, allow=0.01) and judge_as_same(dsc_Rg1, dsc_Rg2, allow=0.01):
                    lrf_rank = 1
                    how = "safely"
                else:
                    if judge_as_same(asc_Rg2, dsc_Rg2, allow=0.02):
                        lrf_rank = 2
                        how = "safely"
                    else:
                        lrf_rank = 1
                        how = "unsurely"

                self.logger.info("LRF rank has been %s judged as %d for the %d-th peak.", how, lrf_rank, pno)

            self.peak_judgements.append(lrf_rank)
        return self.peak_judgements[pno]

    def synthesize(self, rank2_result, rank1_result):
        return rank2_result

def run_lrf_solver_impl(solver, selector, lrf_rank=None,
            exe_queue=None, stop_event=None ):

    results = []
    peak_range_infos = []
    row = 0
    if exe_queue is not None:
        exe_queue.put( [0, row ] )
        if stop_event.is_set():
            return

    for pno, paired_range in enumerate(solver.cnv_ranges):
        fromto_list = paired_range.get_fromto_list()
        for rno, range_ in enumerate(fromto_list):
            start   = range_[0]
            stop    = range_[1] + 1
            peakset_info = selector.select_peakset(row)
            try:
                if USE_LRF_SOLVER:
                    A, B, Z, E, lrf_info, C = solver.solve_range(start, stop,
                                                peakset_info, lrf_rank=lrf_rank,
                                                stop_event=stop_event,
                                                paired_range=paired_range,
                                                )
                else:
                    A, B, Z, E, lrf_info, C = solver.extrapolate_wiser( start, stop, peakset_info,
                                        stop_event=stop_event,
                                        lrf_rank=lrf_rank,
                                        # lrf_rank=None,
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
