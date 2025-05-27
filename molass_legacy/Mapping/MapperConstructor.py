"""

    MapperConstructor.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF

"""
import copy
import numpy as np
import logging
from molass_legacy.KekLib.OurTkinter             import Tk
import OurMessageBox        as MessageBox
from .MappingParams import get_mapper_opt_params, get_mapper_simplest_params, set_mapper_opt_params
from molass_legacy.KekLib.ExceptionTracebacker   import ExceptionTracebacker, log_exception
from molass_legacy.KekLib.BasicUtils             import Struct
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting, INTEGRAL_BASELINE
from molass_legacy.Test.TesterLogger import write_to_tester_log
from .ElutionMapper import ElutionMapper
from molass_legacy.SerialAnalyzer.ElutionCurve import SEDIMENTATION_LIMIT
from molass_legacy.Baseline.LinearAdjustment import LaJudge

DEBUG = True
ACCEPTABLE_ADJ_DEVIATION    = 0.2   # > 0.1 for Gactin
ADJ_DEVIATION_SCALE         = 10
RE_OPTIMIZE_MIN_SCI_LIMIT   = 50

class MockAnalyzerDialog(Tk.Frame):
    def __init__( self, parent ):
        self.serial_data    = Struct( helper_info=None )
        Tk.Frame.__init__( self, parent )
        self.logger         = logging.getLogger( __name__ )

def create_mapper(parent,
                    serial_data,        # analysis_copy or trimmed sd
                    input_sd,           # sd_orig
                    pre_recog,          # created from sd_orig
                    callbacks=None,
                    analyzer_dialog=None, mapper_hook=None,
                    opt_params=None,
                    return_dialog=False, logger=None ):
    """
    moved this process out of ElutionMapperCanvas.__init__
    because the construction of mapper must be complete
    before ElutionMapperCanvas.__init__
    """

    if analyzer_dialog is None:
        analyzer_dialog = MockAnalyzerDialog( parent )

    mapper = ElutionMapper(serial_data, input_sd, pre_recog, callbacks=callbacks)

    # set them simlpy when used_mapping_params exist
    used_mapping_params = get_setting('used_mapping_params')
    if used_mapping_params is not None:
        if logger is not None:
            logger.info("restoring %s from used_mapping_params.", str(used_mapping_params))
        mapper.set_opt_params(used_mapping_params)
        return mapper

    if serial_data.mtd_elution is None:
        pass
    else:
        # 
        absorbance  = mapper.absorbance
        serial_data.mtd_elution.set_baseplane_params_as_a_temp_fix(absorbance)
        mapper.optimize()
        return mapper

    if mapper_hook is None:
        helper_info = serial_data.helper_info
        apply_patch = False
    else:
        helper_info = mapper_hook.helper_info
        apply_patch = mapper_hook.apply_patch

    if opt_params is None:
        opt_params = get_mapper_opt_params()

    ret = optimize_mapper(mapper, opt_params, parent, serial_data, analyzer_dialog, helper_info, apply_patch, logger)
    if ret is None:
        return None

    nrmsd = ret[1]
    adj_dev = mapper.get_adjustment_deviation()
    alternative = False
    if adj_dev > ACCEPTABLE_ADJ_DEVIATION:
        if logger is not None:
            logger.warning('retrying mapper construction with the simplest params becauce of adj_dev=%g' % adj_dev)
        simplest_params = get_mapper_simplest_params()
        alt_mapper  = ElutionMapper(serial_data, input_sd, pre_recog, callbacks=callbacks)
        alt_ret = optimize_mapper(alt_mapper, simplest_params, parent, serial_data, analyzer_dialog, helper_info, apply_patch, logger)
        if alt_ret is None:
            return None

        alt_adj_dev = alt_mapper.get_adjustment_deviation()
        if alt_adj_dev < adj_dev:
            alt_nrmsd = alt_ret[1]
            if alt_nrmsd < nrmsd:
                alternative = True
                mapper = alt_mapper
                adj_dev = alt_adj_dev
                nrmsd = alt_nrmsd
                if logger is not None:
                    logger.warning('adopted mapper with the simplest params')

    judge = LaJudge(mapper)
    judge.modify_opt_params()

    set_mapper_opt_params(mapper.opt_params)

    write_to_tester_log( 'mapped info: adj_dev=%g, nrmsd=%g, alternative=%s\n' % (adj_dev, nrmsd, str(alternative)) )

    serial_data.helper_info = helper_info
    if return_dialog:
        return mapper, analyzer_dialog
    else:
        return mapper

def optimize_mapper(mapper, opt_params, parent, serial_data, analyzer_dialog, helper_info, apply_patch, logger, debug=False):
    # TODO: remove analyzer_dialog

    pre_recog   = serial_data.pre_recog
    absorbance  = mapper.absorbance
    mapping_ok  = False
    auto_make   = True
    if logger is None:
        logger = analyzer_dialog.logger

    xray_baseline_type = get_setting('xray_baseline_type')
    if xray_baseline_type == INTEGRAL_BASELINE:
        opt_baseline_types = [INTEGRAL_BASELINE]
    else:
        uv_baseline_type = get_setting("uv_baseline_type")
        if serial_data.absorbance.shifted_baseline_ok() or uv_baseline_type == 4:
            opt_baseline_types = [0, 1, 4]
        else:
            # e.g., 20200630_6
            opt_baseline_types = [0, 1]

    temp_params = copy.deepcopy(opt_params)

    while not mapping_ok:
        try:
            min_dev_score = None
            selected_type  = None

            # temporarily allow bad_std_diff to cope with bad data such as Factin
            bad_std_diff_ok_save = get_setting( 'bad_std_diff_ok' )
            set_setting( 'bad_std_diff_ok', 1 )

            if debug:
                dev_score_list = []

            for baseline_type in opt_baseline_types:
                absorbance.compute_base_curve(pre_recog, baseline_type)
                temp_params['uv_baseline_opt'] = 0 if baseline_type == 0 else 1
                temp_params['uv_baseline_type'] = baseline_type
                try:
                    mapper.optimize( helper_info=helper_info, apply_patch=apply_patch, opt_params=temp_params )
                    adj_dev = mapper.get_adjustment_deviation()
                    if logger is not None:
                        method_label = 'LPM' if baseline_type == 0 else 'LB'
                        logger.info('optimize_mapper: nrmsd=%.4g, adj_dev=%.4g with baseline_type=%s for absorbance.compute_base_curve' % (mapper.std_diff, adj_dev, method_label) )
                    dev_score = mapper.std_diff + adj_dev * ADJ_DEVIATION_SCALE
                    if debug:
                        dev_score_list.append(dev_score)
                    if min_dev_score is None or dev_score < min_dev_score:
                        min_dev_score = dev_score
                        selected_type = baseline_type
                except Exception as exc:
                    # baseline_type LB fails for 3/7
                    if DEBUG:
                        log_exception(logger, "failed in mapper.optimize: ", n=8)

            if debug:
                import molass_legacy.KekLib.DebugPlot as plt
                with plt.Dp():
                    fig, ax = plt.subplots()
                    ax.set_title("optimize_mapper debug")
                    ax.plot(dev_score_list, "-o")
                    plt.show()

            if min_dev_score is None:
                raise RuntimeError( 'Failed to optimize for both baseline methods for the absorbance data', [ 5 ] )

            if selected_type < opt_baseline_types[-1]:
                # TODO: improve by avoiding this redundant repitition
                logger.warning( 'undesirable repitition of UV-basecurve computation' )
                absorbance.compute_base_curve(pre_recog, selected_type)
                temp_params['uv_baseline_opt'] = 0 if selected_type == 0 else 1
                temp_params['uv_baseline_type'] = selected_type
                mapper.optimize( helper_info=helper_info, apply_patch=apply_patch, opt_params=temp_params )

            set_setting( 'uv_baseline_type', selected_type )
            mapper.opt_params['uv_baseline_type'] = selected_type       # TODO: investigate why this is required

            set_setting( 'bad_std_diff_ok', bad_std_diff_ok_save )

            if logger is not None:
                logger.info("optimize_mapper: %s has been selected for uv_baseline_method with min_dev_score=%.4g" % ( 'LPM' if selected_type == 0 else 'LB', min_dev_score ))

            mapping_ok  = True
        except RuntimeError as exc:
            etb= ExceptionTracebacker()
            print( etb )
            logger.warning('exception in optimize_mapper etb=%s', str(etb))
            auto_make = False       # don't use MappingAutoHelper further to avoid infinite loop
            if helper_info is None:
                MessageBox.showwarning( "Suggestion",
                        "Restricting elution ranges might help.",
                        parent=parent )
                return None
        except:
            etb = ExceptionTracebacker()
            print(etb)
            logger.error( str(etb) )
            MessageBox.showerror( "Error",
                    "Failed to conctruct a mapper.\n"
                    + "Please report this error to the developer.\n"
                    + "Restricting elution ranges mihgt help.\n\n"
                    + str(etb),
                    parent=parent )
            return None

    if mapper.feature_mapped:
        min_sci = np.min(mapper.get_sci_list())
        if min_sci < RE_OPTIMIZE_MIN_SCI_LIMIT:
            sync_options = 0
            mapper.optimize(opt_params=mapper.opt_params, sync_options=sync_options)
            set_setting('mapper_sync_options', sync_options)
            logger.info('re-optimized mapper with "rmsd finally" due to a bad minimun sci: %.2g.' % min_sci)

    return selected_type, min_dev_score
