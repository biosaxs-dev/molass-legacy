"""

    QuickAnalysis.AnalyzerDialogProxy.py

    Copyright (c) 2020-2025, SAXS Team, KEK-PF

"""
from molass_legacy._MOLASS.SerialSettings import get_setting, set_setting

class AnalyzerDialogProxy:
    def __init__(self, parent, sd, mapper, judge_holder, correction_necessity):
        from molass_legacy.Extrapolation.PreviewButtonFrame import PreviewButtonFrame
        self.parent = parent
        self.doing_sec  = parent.use_mtd_conc == 0
        self.sd = sd
        self.mapper = mapper
        self.correction_necessity = correction_necessity
        use_elution_models=get_setting('use_elution_models')
        self.preview_frame = PreviewButtonFrame(parent, dialog=self, use_elution_models=use_elution_models, judge_holder=judge_holder)

    def make_range_info(self, *args):
        """
            concentration_datatype will be conveyed to later processes
            through the SerialSettings item set by preview.update_settings()
            independently from these args.
        """
        analysis_range_info = get_setting('analysis_range_info')
        return analysis_range_info.get_ranges()

    def get_decomp_info(self):
        # for compatibility with MctDecompDialog concerning Preview
        pdata, popts = get_setting('preview_params')
        return pdata.decomp_info

    def apply(self):
        self.parent.cfs_entry.update_conc_factors(self.sd, self.mapper)
        set_setting('averaged_data_folder', get_averaged_data_folder())
        set_setting('scattering_correction', int(self.correction_necessity))

        """
        TODO: this update is or can be a duplication. remove such inconvenience.
        """
        if self.doing_sec:
            self.preview_frame.update_settings()
        else:
            # 
            pass

def get_averaged_data_folder():
    # folder = get_setting( 'averaged_data_folder' )
    # if is_empty_val( folder ):
    folder = get_setting( 'analysis_folder' ) + '/averaged'
    return folder
