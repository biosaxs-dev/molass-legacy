"""
    Optimizer.BoundaryView.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
from molass_legacy.SerialAnalyzer.DataUtils import get_in_folder

class BoundaryView(Dialog):
    def __init__(self, parent, parent_dialog, state_canvas):
        self.parent = parent
        self.parent_dialog = parent_dialog
        self.elution_model = parent_dialog.elution_model
        self.optimizer = parent_dialog.fullopt
        self.state_canvas = state_canvas
        self.x_array = state_canvas.demo_info[1]
        self.best_index = state_canvas.get_best_index()
        self.fullopt = state_canvas.fullopt
        Dialog.__init__(self, parent, "Boundary View", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        tframe = Tk.Frame(body_frame)
        tframe.pack(fill=Tk.X, padx=20)
        tframe_left = Tk.Frame(tframe)
        tframe_left.pack(side=Tk.LEFT)

        self.fig = fig = plt.figure(figsize=(12, 6))
        self.draw_boundary_view(fig)

        fig.tight_layout()
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)

        self.toolbar = NavigationToolbar(self.mpl_canvas, tframe)
        self.toolbar.update()

    def buttonbox( self ):
        box = Tk.Frame(self)
        box.pack()
        w = Tk.Button(box, text="Close", width=10, command=self.cancel)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)

    def draw_boundary_view(self, fig):
        job_info = self.parent_dialog.get_job_info()
        job_name = job_info[0]
        in_folder = get_in_folder()

        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="polar")
        # fig.tight_layout()
        fig.subplots_adjust(top=0.8)

        fig.suptitle("Parameter Variation Sequence Analysis of Job %s on %s" % (job_name, in_folder), fontsize=20)

        ax1.set_title("Boundary Cylinder 3D View", fontsize=16)
        ax2.set_title("Boundary Cylinder Slice View", fontsize=16)

        """
        Add cylinder to plot
        https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
        """
        # Create the mesh in polar coordinates and compute corresponding Z.
        def data_for_cylinder_along_z(center_x, center_y, radius, sequence_length, nx_array=None):
            num_params = 50 if nx_array is None else nx_array.shape[1]
            z = np.arange(sequence_length)
            theta = np.linspace(0, 2*np.pi, num_params)
            theta_grid, z_grid=np.meshgrid(theta, z)
            if nx_array is None:
                x_grid = radius*np.cos(theta_grid) + center_x
                y_grid = radius*np.sin(theta_grid) + center_y
            else:
                radius_ = nx_array[z,:]
                x_grid = radius_*np.cos(theta_grid) + center_x
                y_grid = radius_*np.sin(theta_grid) + center_y
            return x_grid, y_grid, z_grid

        n_params_list = []
        bounds_mask = self.fullopt.bounds_mask
        lower_bounds = self.fullopt.lower_bounds
        upper_bounds = self.fullopt.upper_bounds
        for params in self.x_array:
            n_params = (params[bounds_mask] - lower_bounds)/(upper_bounds - lower_bounds)
            n_params_list.append(n_params)
        nx_array = np.array(n_params_list)
        sequence_length = nx_array.shape[0]
        print("sequence_length=", sequence_length)
        Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, 1, sequence_length)
        ax1.plot_surface(Zc, Xc, Yc, alpha=0.1)
        Xp, Yp, Zp = data_for_cylinder_along_z(0, 0, 0.5, sequence_length, nx_array)
        ax1.plot_surface(Zp, Xp, Yp, color="green", alpha=0.5)
        ax1.set_box_aspect(aspect = (2,1,1))

        n = 50
        x = np.ones(n) * self.best_index
        theta = np.linspace(0, 2*np.pi, n)
        y = np.sin(theta)
        z = np.cos(theta)
        ax1.plot(x, y, z, color="red", alpha=0.5, lw=3)

        if self.elution_model == 1:
            # stochastic models seem to be inclined to have outliers
            # or, should always do this?
            ax1.set_ylim(-1, 1)
            ax1.set_zlim(-1, 1)

        n_params = nx_array[self.best_index,:]
        n = len(n_params)
        ax2.set_rlim(0, 1)
        ax2.set_xticks(np.pi/180. * np.linspace(0,  360, n, endpoint=False))
        num_all_params = self.x_array.shape[1]
        # all_param_ids = np.arange(num_all_params)
        all_names = self.optimizer.get_parameter_names()
        # print("all_names=", all_names)
        ax2.set_xticklabels(all_names[bounds_mask])
        ax2.set_rlabel_position(180/n)
        theta = np.linspace(0, 2*np.pi, n+1)
        n_params_ = np.concatenate([n_params, n_params[0:1]])
        ax2.plot(theta, n_params_, color="green")
