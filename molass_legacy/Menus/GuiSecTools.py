"""
    GuiSecTools.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF
"""
from molass_legacy.KekLib.OurTkinter import Tk
from molass_legacy._MOLASS.SerialSettings import get_setting

class GuiSecToolsMenu(Tk.Menu):
    def __init__(self, parent, menubar ):
        self.parent = parent

        Tk.Menu.__init__(self, menubar, tearoff=0 )
        from molass_legacy.Env.EnvInfo import get_global_env_info
        self.env_info = get_global_env_info()
        atsas_state = Tk.NORMAL if self.env_info.atsas_is_available else Tk.DISABLED

        menubar.add_cascade( label="SEC Tools", menu=self )
        # self.add_command( label="Average Subtractor", command=self.show_average_subtractor_dialog )
        self.add_command( label="Elution Curve Picker", command=self.show_elution_curve_picker, state=Tk.DISABLED )
        self.add_command( label="Scattering Curve Plotter", command=self.show_scattering_curve_plotter )
        self.add_command( label="SVD Viewer", command=self.show_svd_viewer, state=Tk.DISABLED )
        self.add_command( label="Average Maker", command=self.show_average_maker)
        # self.add_command( label="Correlation Map Maker", command=self.show_cormap_maker)
        # self.add_command( label="Scaled Data CorMap", command=self.show_scaled_cormap, state=atsas_state)
        # self.add_command( label="Abnormal Data Handling", command=self.show_abnomality_check_dialog, state=Tk.DISABLED )
        # self.add_command( label="Manual Adjuster", command=self.show_manual_adjuster, state=Tk.DISABLED )
        self.add_command( label="Same Sample in Diffrent Conditions", command=self.show_ssdc_analysis)

    def update_states(self):
        use_mtd_conc = get_setting('use_mtd_conc')
        state = Tk.NORMAL if self.parent.dataset_is_ready else Tk.DISABLED

        for k in [0, 2]:
            # if k == 2 and use_mtd_conc:
                # currently disabled due to the error:  AttributeError: 'Absorbance' object has no attribute 'tail_index'
            if k == 2:
                # AttributeError: 'Absorbance' object has no attribute 'baseplane_params'
                state_ = Tk.DISABLED
            else:
                state_ = state
            self.entryconfig(k, state=state_)

    def show_average_subtractor_dialog(self):
        print('show_average_subtractor_dialog')
        from molass_legacy.Microfluidics.AverageSubtractorDialog import AverageSubtractorDialog
        dialog = AverageSubtractorDialog(self.parent, use_mtd=False)
        dialog.show()

    def show_elution_curve_picker(self):
        print('show_data_subtractor_dialog')
        from molass_legacy.Tools.ElutionCurvePicker import ElutionCurvePicker
        parent = self.parent
        dialog = ElutionCurvePicker(parent, parent.pre_recog)
        dialog.show()
        if dialog.applied:
            parent.fig_frame.update_elution_curve()

    def show_scattering_curve_plotter(self):
        print('show_scattering_curve_plotter')
        from molass_legacy.DataStructure.ScatteringCurvePlotter import ScatteringCurvePlotter
        dialog = ScatteringCurvePlotter(self.parent)
        dialog.show()

    def show_svd_viewer(self):
        import Decomposer
        from molass_legacy.DataStructure.SvdViewer import SvdViewer
        from molass_legacy.Mapping.MapperConstructor import create_mapper
        from molass_legacy._MOLASS.SerialSettings import set_setting

        parent = self.parent
        parent.config(cursor='wait')    # does not work. why?
        parent.update()

        analysis_name, analysis_folder = parent.make_analysis_folder()
        set_setting('analysis_folder', analysis_folder)

        pre_recog = parent.pre_recog
        # sd_copy = pre_recog.get_analysis_copy()
        input_sd = parent.serial_data
        sd_copy = input_sd._get_analysis_copy_impl(pre_recog)
        in_folder = self.parent.in_folder.get()
        title = "SVD Viewer - " + in_folder
        mapper = create_mapper( parent, sd_copy, input_sd, pre_recog, analyzer_dialog=parent, logger=parent.tmp_logger )
        viewer = SvdViewer( parent, title, sd_copy, mapper )

        parent.config(cursor='')

        viewer.show()

    def show_average_maker(self):
        from SecTools.AverageMakerDialog import AverageMakerDialog
        dialog = AverageMakerDialog(self.parent)
        dialog.show()

    def show_cormap_maker(self):
        from SecTools.CorMap.CormapMakerDialog import CormapMakerDialog
        atsas = self.env_info.atsas_is_available
        dialog = CormapMakerDialog(self.parent, atsas=atsas)
        dialog.show()

    def show_scaled_cormap(self):
        from SecTools.CorMap.ScaledDataDialog import ScaledDataDialog
        atsas = self.env_info.atsas_is_available
        dialog = ScaledDataDialog(self.parent, atsas=atsas)
        dialog.show()

    def show_abnomality_check_dialog(self):
        print('show_abnomality_check_dialog')

    def show_manual_adjuster(self):
        from molass_legacy.Mapping.ManualAdjuster import ManualAdjuster
        parent = self.parent
        sd = parent.pre_recog.pre_recog_copy
        ms = ManualAdjuster( parent, sd )
        ms.show()
        if ms.applied:
            # TODO:
            pass

    def show_ssdc_analysis(self, devel=True):
        from importlib import reload
        if devel:
            import SSDC.FolderEntryDialog
            reload(SSDC.FolderEntryDialog)
        from SSDC.FolderEntryDialog import FolderEntryDialog
        dialog = FolderEntryDialog(self.parent)
        dialog.show()
        if dialog.applied:
            if devel:
                import SSDC.SsdcAnalysis
                reload(SSDC.SsdcAnalysis)
            from SSDC.SsdcAnalysis import SsdcAnalysis
            analysis = SsdcAnalysis(self.parent, dialog.get_info_list())
            analysis.show()
