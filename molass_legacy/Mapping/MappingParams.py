# coding: utf-8
"""
    MappingParams.py

    Copyright (c) 2018-2021, SAXS Team, KEK-PF
"""
from collections            import OrderedDict
from molass_legacy._MOLASS.SerialSettings         import get_setting, set_setting

MAPPER_OPT_PARAMS   = [ 'uv_baseline_opt', 'uv_baseline_type', 'uv_baseline_adjust',
                        'uv_baseline_with_bpa',
                        'xray_baseline_opt', 'xray_baseline_type', 'xray_baseline_adjust',
                        'xray_baseline_with_bpa', 'dev_allow_ratio' ]

MAPPER_SIMPLEST_VALUES = [  0, 0, 0,
                            0,
                            0, 0, 0,
                            0, 0.5 ]

XRAY_LPM_DEGREE_TEXTS   = [ None, 'linear', 'quatratic', 'spline', None, 'integral' ]
# UV_CORRECT_TYPE_TEXTS   = [ 'standard', 'shift' ]
UV_CORRECT_TYPE_TEXTS   = [ None, 'linear', '', '', 'shifted', 'integral']
UV_METHOD_NAMES = ['LPM', 'LB', '', '', '', 'integral']

def get_mapper_opt_params():
    params = []
    for param in MAPPER_OPT_PARAMS:
        params += [ (param, get_setting( param ) ) ]
    return MappingParams(params)

def set_mapper_opt_params(opt_params):
    for param in MAPPER_OPT_PARAMS:
        set_setting(param, opt_params[param])

def get_mapper_simplest_params():
    params = []
    for k, param in enumerate(MAPPER_OPT_PARAMS):
        params += [ (param, MAPPER_SIMPLEST_VALUES[k] ) ]
    return MappingParams(params)

"""
    TODO: duplicated info in mapped_info and MAPPER_OPT_PARAMS in SerialSettings
"""

class MappingParams(OrderedDict):
    def __init__(self, *args):
        OrderedDict.__init__(self, *args)

    def get_xray_correction_str(self):
        xray_baseline_opt = self.get('xray_baseline_opt')
        if xray_baseline_opt == 0:
            ret = 'no correction'
        else:
            xray_adjust_with_bpa = self.get('xray_adjust_with_bpa')
            xray_baseline_type = self.get('xray_baseline_type')
            if xray_baseline_type == 6:
                ret = 'BPA'
            else:
                ret = 'LPM(%s)' % XRAY_LPM_DEGREE_TEXTS[xray_baseline_type]
                if xray_adjust_with_bpa:
                    ret += '+BPA'
            xray_baseline_adjust = self.get('xray_baseline_adjust')
            if xray_baseline_adjust == 1:
                dev_allow_ratio = self.get('dev_allow_ratio')
                ret  += ' with adjustment(%.1g)' % dev_allow_ratio
        return ret

    def get_uv_correction_str(self):
        use_xray_conc  = get_setting('use_xray_conc')
        if use_xray_conc:
            return ''

        uv_baseline_opt = self.get('uv_baseline_opt')
        if uv_baseline_opt == 0:
            ret = 'no correction'
        else:
            uv_baseline_type = self.get('uv_baseline_type')
            correct_type = UV_CORRECT_TYPE_TEXTS[uv_baseline_type]
            ret = correct_type
            uv_baseline_adjust = self.get('uv_baseline_adjust')
            if uv_baseline_adjust == 1:
                dev_allow_ratio = self.get('dev_allow_ratio')
                ret  += ' with adjustment(%.1g)' % (1 - dev_allow_ratio)
        return ret

class MappedInfo:
    def __init__( self, **entries ): 
        self.__dict__.update(entries)

    def needs_sd_corrcetion( self ):
        return self.opt_params['xray_baseline_opt'] == 1
