# coding: utf-8
"""
    LinTranAnim.py

    Copyright (c) 2019,2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Circle
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import get_color
from OurManim import manim_init, use_default_style, Animation, Collection, TextGroup, Parallelogram, Arrow
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

class LinTranAnim(Dialog):
    def __init__(self, parent=None):
        if parent is None:
            from SerialTestUtils import get_tk_root
            parent = get_tk_root()
        self.parent = parent
        self.num_frames = 70
        manim_init()
        Dialog.__init__(self, parent, "Linear Transformation Tutorial", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(12,6))
        self.axes[1].set_axis_off()
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, cframe)
        self.toolbar.update()
        self.draw()
        self.fig.tight_layout()
        self.mpl_canvas.draw()

    def buttonbox(self):
        box = Tk.Frame(self)
        box.pack()

        w = Tk.Button(box, text="OK", width=10, command=self.ok, default=Tk.ACTIVE)
        w.pack(side=Tk.LEFT, padx=5, pady=5)
        w = Tk.Button(box, text="Save", width=10, command=self.save)
        w.pack(side=Tk.LEFT, padx=5, pady=5)

    def draw(self):
        ax1, ax2 = self.axes

        cube_labels = ['(0,0)', '(1,0)', '(1,1)', '(0,1)']
        cube_points = [(0,0), (1,0), (1,1), (0,1)]
        data_labels = ['(a,b)', '(a+c, b+d)', '(c,d)']
        data_points = [(2,0.5), (2.5, 2.0), (0.5, 1.5)]

        labels = cube_labels + data_labels
        points = cube_points + data_points
        d = 0.05

        for t, p in zip(labels, points):
            ax1.plot(*p,'o')
            ax1.text(p[0]+d, p[1]+d, t, fontsize=16)

        ax1.set_xlim(-0.5, 3.0)
        ax1.set_ylim(-0.5, 3.0)
        ax1.set_aspect('equal')

        cube =  Polygon(cube_points, alpha=0.3, color='pink')
        ax1.add_patch(cube)

        matrix0 = r"$ y = \begin{bmatrix} a & c \\ b & d \end{bmatrix} x $"
        ax2.text(0.5, 0.8, matrix0, fontsize=30, ha='center', va='center', color='w')

        matrix1_1 = r"$ \begin{bmatrix} \_ \\ \_ \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} $"
        matrix1_2 = r"$ \begin{bmatrix} a \\ \_ \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} $"
        matrix1_3 = r"$ \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} $"

        mt1 = ax2.text(0.5, 0.5, matrix1_1, fontsize=30, ha='center', va='center', color='w')

        matrix2_1 = r"$ \begin{bmatrix} \_ \\ \_ \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} $"
        matrix2_2 = r"$ \begin{bmatrix} c \\ \_ \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} $"
        matrix2_3 = r"$ \begin{bmatrix} c \\ d \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} $"

        mt2 = ax2.text(0.5, 0.2, matrix2_1, fontsize=30, ha='center', va='center', color='w')

        def on_reset():
            print('on_reset')
            mt1.set_text(matrix1_1)
            mt2.set_text(matrix2_1)

        collection = Collection(self.num_frames, ret_artists=[mt1, mt2], on_reset=on_reset)

        anim1_4 = Animation(0, 20)
        arrow1 = Arrow(ax1, cube_points[1], (0,0), width=0.07, head_width=0.2, head_length=0.2)
        arrow1.set_target_position(cube_points[1], np.array(data_points[0]) - np.array(cube_points[0]))
        anim1_4.append(arrow1)
        collection.append(anim1_4)

        # a <= (1,0)
        v10 = (['$1$', '$0$'], [(0.67, 0.53), (0.67, 0.44)])
        tg1 = TextGroup(ax2, *v10, fontsize=30)
        tg1.set_target_positions([(0.47, 0.53), (0.54, 0.53)])

        anim1 = Animation( 0,  5)
        anim1.append(tg1)
        collection.append(anim1)

        # a <= a
        tg2 = TextGroup(ax2, ['$a$'], [(0.47, 0.53)], fontsize=30)
        tg2.set_target_positions([(0.26, 0.53)])
        anim2 = Animation( 5, 10, on_stop=lambda: mt1.set_text(matrix1_2))
        anim2.append(tg2)
        collection.append(anim2)

        # b <= (1,0)
        tg3 = TextGroup(ax2, *v10, fontsize=30)
        tg3.set_target_positions([(0.47, 0.44), (0.54, 0.44)])

        anim3 = Animation(10, 15)
        anim3.append(tg3)
        collection.append(anim3)

        # b <= b
        tg4 = TextGroup(ax2, ['$b$'], [(0.47, 0.43)], fontsize=30)
        tg4.set_target_positions([(0.26, 0.43)])

        anim4 = Animation(15, 20, on_stop=lambda: mt1.set_text(matrix1_3))
        anim4.append(tg4)
        collection.append(anim4)

        # c <= (0,1)
        v01 = ['$0$', '$1$'], [(0.67, 0.23), (0.67, 0.13)]
        tg5 = TextGroup(ax2, *v01, fontsize=30)
        tg5.set_target_positions([(0.47, 0.23), (0.54, 0.23)])
        anim5 = Animation(20, 25)
        anim5.append(tg5)
        collection.append(anim5)

        # c <= c
        tg6 = TextGroup(ax2, ['$c$'], [(0.54, 0.23)], fontsize=30)
        tg6.set_target_positions([(0.26, 0.23)])
        anim6 = Animation(25, 30, on_stop=lambda: mt2.set_text(matrix2_2))
        anim6.append(tg6)
        collection.append(anim6)

        # d <= (0,1)
        tg7 = TextGroup(ax2, *v01, fontsize=30)
        tg7.set_target_positions([(0.47, 0.13), (0.54, 0.13)])
        anim7 = Animation(30, 35)
        anim7.append(tg7)
        collection.append(anim7)

        # d <= d
        tg8 = TextGroup(ax2, ['$d$'], [(0.54, 0.13)], fontsize=30)
        tg8.set_target_positions([(0.26, 0.13)])
        anim8 = Animation(35, 40, on_stop=lambda: mt2.set_text(matrix2_3))
        anim8.append(tg8)
        collection.append(anim8)

        anim9 = Animation(40, 45)
        collection.append(anim9)

        # print('index=', collection.index)
        para1 = Parallelogram(ax1, cube_points, alpha=0.3, color='cyan')
        para1.set_target_vertices([(0,0)] + data_points)
        anim_10 = Animation(45, 60)
        anim_10.append(para1)
        collection.append(anim_10)

        self.anim = collection.make_animation(self.fig)

    def save(self):
        import os
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        path = os.path.join(os.getcwd(), "anim.mp4")
        self.anim.save(path, writer="ffmpeg")
        MessageBox.showinfo("Save Notification", 'The movie has been saved to "%s"' % path, parent=self)
