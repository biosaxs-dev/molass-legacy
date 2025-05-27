"""
    Batch.BatchMappingUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from molass_legacy._MOLASS.SerialSettings import clear_temporary_settings, set_setting
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Batch.LiteBatch import LiteBatch     

class MappingBatch(LiteBatch):
    def __init__(self, mapping_src):
        super().__init__()
        self.mapping_src = mapping_src

    def modelled_peaks_method(self, uv_x, uv_y_, xr_x, xr_y_, debug=True):
        def show_mapped_result(debug=True):
            from importlib import reload
            import Batch.BatchMappingImpl
            reload(Batch.BatchMappingImpl)
            from molass_legacy.Batch.BatchMappingImpl import show_mapped_result_impl
            show_mapped_result_impl(self, uv_x, uv_y_, xr_x, xr_y_, debug=debug)

        if debug:
            extra_button_specs = [("Show Mapped Result", show_mapped_result)]
            with plt.Dp(extra_button_specs=extra_button_specs):
                fig, ax = plt.subplots()
                fig.suptitle("modelled_peaks_method")
                plt.show()
        else:
            show_mapped_result(debug=False)

def load_lrfsrc_with_mapping(in_folder, mapping_src, clear=True):
    if clear:
        clear_temporary_settings()
    set_setting('in_folder', in_folder)
    mbatch = MappingBatch(mapping_src)
    lrf_src = mbatch.get_lrf_source(in_folder=in_folder, use_mapping=True)
    return lrf_src