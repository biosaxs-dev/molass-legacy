"""
    Batch.LiteBatch.py

    Copyright (c) 2023-2025, SAXS Team, KEK-PF
"""
import logging
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.Optimizer.FuncImporter import get_objective_function_info
from .OptDataSetsProxy import OptDataSetsProxy as OptDataSets
from .FullBatch import FullBatch

class LiteBatch(FullBatch):
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.unified_baseline_type = get_setting("unified_baseline_type")

    def get_init_estimate(self):
        self.func_info = get_objective_function_info(self.logger, default_func_code='G0346')
        self.func_dict = self.func_info.func_dict
        self.key_list = self.func_info.key_list
        self.dsets = OptDataSets(self.sd, self.corrected_sd)
        self.construct_optimizer()
        init_params = self.compute_init_params()
        return init_params

def create_corrected_sd(sd):
    lb = LiteBatch()
    lb.prepare(sd)
    return lb.corrected_sd
