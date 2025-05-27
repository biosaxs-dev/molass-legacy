"""

    GuiMctTools.py

    Copyright (c) 2018-2022, SAXS Team, KEK-PF

"""
from molass_legacy.KekLib.OurTkinter import Tk

class GuiMctToolsMenu(Tk.Menu):
    def __init__(self, parent, dialog, menubar ):
        self.parent = parent
        self.dialog = dialog

        Tk.Menu.__init__(self, menubar, tearoff=0 )
        menubar.add_cascade( label="MCT Tools", menu=self )
        self.add_command( label="Average Subtractor", command=self.show_average_subtractor_dialog )
        self.add_command( label="Individual Subtractor", command=self.show_individual_subtractor_dialog )
        self.add_command( label="Conc Normalizer", command=self.show_conc_normalizer_dialog )
        self.add_command( label="Elution Curve Picker", command=self.show_elution_curve_picker, state=Tk.DISABLED )
        self.add_command( label="Average Maker", command=self.show_average_maker)

    def update_states(self):
        state = Tk.NORMAL if self.dialog.dataset_is_ready else Tk.DISABLED
        for k in [3]:
            self.entryconfig(k, state=state)

    def show_average_subtractor_dialog(self):
        print('show_average_subtractor_dialog')
        from Microfluidics.AverageSubtractorDialog import AverageSubtractorDialog
        dialog = AverageSubtractorDialog(self.parent)
        dialog.show()

    def show_individual_subtractor_dialog(self):
        print('show_individual_subtractor_dialog')
        from Microfluidics.IndividualSubtractorDialog import IndividualSubtractorDialog
        dialog = IndividualSubtractorDialog(self.parent)
        dialog.show()

    def show_conc_normalizer_dialog(self):
        print('show_conc_normalizer_dialog')
        from Microfluidics.ConcNormalizerDialog import ConcNormalizerDialog
        dialog = ConcNormalizerDialog(self.parent)
        dialog.show()

    def show_elution_curve_picker(self):
        print('show_individual_subtractor_dialog')
        from molass_legacy.Tools.ElutionCurvePicker import ElutionCurvePicker
        parent = self.parent
        dialog = ElutionCurvePicker(parent, parent.pre_recog)
        dialog.show()
        if dialog.applied:
            parent.fig_frame.update_elution_curve()

    def show_average_maker(self):
        from SecTools.AverageMakerDialog import AverageMakerDialog
        dialog = AverageMakerDialog(self.parent)
        dialog.show()
