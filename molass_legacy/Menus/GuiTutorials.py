"""

    GuiTutorials.py

    Copyright (c) 2016-2023, SAXS Team, KEK-PF

"""
from molass_legacy.KekLib.OurTkinter     import Tk, is_empty_val
from molass_legacy._MOLASS.SerialSettings import get_setting
try:
    import molass_legacy.KekLib.CustomMessageBox         as MessageBox
except:
    import OurMessageBox            as MessageBox

CONJUGATE_GRADIENT = 5

class GuiTutorialsMenu(Tk.Menu):
    def __init__(self, parent, menubar ):
        self.parent = parent

        Tk.Menu.__init__(self, menubar, tearoff=0 )
        menubar.add_cascade( label="Tutorials", menu=self )
        self.add_command( label="Linear Transformation", command=self.show_lintran_tutorial )
        self.add_command( label="Matrix Inverse", command=self.show_matrix_inverse )
        self.add_command( label="Moore-Penrose Inverse", command=self.show_moore_penrose_inverse )
        self.add_command( label="SVD Animation", command=self.show_svd_animation )
        self.add_command( label="SVD (2D)", command=self.show_svd_tutorial_2d )
        self.add_command( label="SVD (3D)", command=self.show_svd_tutorial_3d )
        self.add_command( label="Matrix Factorization", command=self.show_mf_tutorial )
        # update CONJUGATE_GRADIENT above if the order No. of this entry changes.
        self.add_command( label="Conjugate Gradient Algorithm", command=self.show_conjugate_gradient, state=Tk.DISABLED )
        self.add_command( label="Affine Transformation of Baseline Adjustment", command=self.show_affine_transformation )
        self.add_command( label="Extrapolation with Spherical Particles", command=self.show_extrapolation_simulation )
        self.add_command( label="Noise and Sigma-width Dependency of Base Percentile Offset", command=self.show_bpo_dependency_demo )

    def update_states(self):
        state = Tk.NORMAL if self.parent.dataset_is_ready else Tk.DISABLED
        self.entryconfig(CONJUGATE_GRADIENT, state=state)

    def show_extrapolation_simulation( self ):
        from SimulatedExtrapolation import SimulatedExtrapolationDialog
        dialog = SimulatedExtrapolationDialog( 'Simulation', parent=self.parent )
        dialog.show()

    def show_bpo_dependency_demo(self):
        from BpoDependencyDemo import BpoDependencyDemo
        self.parent.config(cursor='wait')
        self.parent.update()
        dialog = BpoDependencyDemo(self.parent)
        self.parent.config(cursor='')
        dialog.show()

    def show_lintran_tutorial(self):
        latex_ok = self.parent.env_info.get_latex_availability()
        if False and latex_ok:
            ret = MessageBox.askokcancel("Warning",
                    "The application currently terminates\n"
                    + "after closing this tutorial\n"
                    + "due to a bug or misuse of some codes.\n"
                    + "Do you proceed anyway?",
                    parent=self.parent)
            if not ret:
                return

            from Tutorials.LinAlg import LinTran
            from OurManim import use_default_style
            tut = LinTran()
            tut.show()
            use_default_style()
        else:
            self._show_video('linear-transfomation.mp4')

    def _show_video(self, video_file):
        from Menus.GuiReferences import get_default_browser, get_preferable_browser, get_doc_folder_url
        browser = get_preferable_browser()
        video_url = get_doc_folder_url() + '/' + video_file
        browser.open_new(video_url)

    def show_matrix_inverse(self):
        self._show_video('matrix-inverse.mp4')

    def show_moore_penrose_inverse(self):
        self._show_video('moore-penrose-inverse.mp4')

    def show_affine_transformation(self):
        from molass_legacy.SerialAnalyzer.AffineDemo import AffineDemoDialog
        dialog = AffineDemoDialog( 'Affine transformation demo', parent=self.parent )
        dialog.show()

    def show_svd_animation(self):
        from Menus.GuiReferences import get_default_browser, get_preferable_browser, get_doc_folder_url
        browser = get_preferable_browser()
        movie_url = get_doc_folder_url() + '/SVD-animation.mp4'
        browser.open_new(movie_url)

    def show_svd_tutorial_2d(self):
        from Tutorials.SvdTutorial2D import SvdTutorial2D
        tut = SvdTutorial2D(self.parent)
        tut.show()

    def show_svd_tutorial_3d(self):
        from Tutorials.SvdTutorial3D import SvdTutorial3D
        tut = SvdTutorial3D(self.parent)
        tut.show()

    def show_mf_tutorial(self):
        from Tutorials.MatrixFactorization import MatrixFactorization
        tut = MatrixFactorization(self.parent)
        tut.show()

    def show_conjugate_gradient(self):
        import Extrapolation
        from ConjugateGradientDemo import ConjugateGradientDemo
        from molass_legacy.Mapping.MapperConstructor import create_mapper
        from molass_legacy._MOLASS.SerialSettings import dump_settings

        # dump_settings()
        parent = self.parent
        parent.config( cursor='wait' )
        parent.update()
        analysis_copy = parent.pre_recog.get_analysis_copy()

        mapper = create_mapper( parent, analysis_copy, analyzer_dialog=parent, logger=parent.tmp_logger )
        conc_factor = 5
        analysis_copy.set_mc_vector(mapper, conc_factor)        # TODO: set_mc_vector

        in_folder = parent.in_folder.get()
        try:
            demo = ConjugateGradientDemo(parent, in_folder, analysis_copy, mapper)
            parent.config( cursor='' )
            demo.show()
        except Exception as exc:
            from molass_legacy.KekLib.ExceptionTracebacker import ExceptionTracebacker
            etb = ExceptionTracebacker()
            MessageBox.showerror("ERROR", "Demo failed with the following error. Try another set of data.\n\n\n----" + str(etb), parent=parent)
