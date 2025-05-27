# coding: utf-8
"""
    MpInverseAnim.py

    Copyright (c) 2020,2024, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Circle
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import get_color
from OurManim import manim_init, use_default_style, Animation, Collection, TextGroup, Parallelogram, Circles
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

class MpInverseAnim(Dialog):
    def __init__(self, parent=None):
        if parent is None:
            from molass_legacy.KekLib.TkUtils import get_tk_root
            parent = get_tk_root()
        self.parent = parent
        self.num_frames = 85
        manim_init()
        Dialog.__init__(self, parent, "Matrix Inverse Tutorial", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()
        gs = GridSpec(7,3)
        self.fig = fig = plt.figure(figsize=(18,7))
        self.ax__ = fig.add_subplot(gs[0,:])
        self.ax__.set_axis_off()
        self.ax__.text(0.5, 0.5, "Matrix Inverse in Linear Transformation Perspective", ha='center', va='center', fontsize=40)

        self.axes = []
        for k in range(3):
            ax = fig.add_subplot(gs[1:,k])
            self.axes.append(ax)
        # self.fig, self.axes = plt.subplots(nrows=1, ncols=3, figsize=(18,6))
        for ax in  self.axes[1:3]:
            ax.set_axis_off()
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
        ax1, ax2, ax3 = self.axes

        cube_labels = ['(0,0)', '(1,0)', '(1,1)', '(0,1)']
        cube_points = [(0,0), (1,0), (1,1), (0,1)]
        data_labels = ['(a,b)', '(a+c, b+d)', '(c,d)']
        data_points = [(1,1), (2, 0), (1, -1)]
        data_labels2 = ['(2, 0)', '(1,-1)']
        data_points2 = [(2, 0), (1, -1)]

        labels = cube_labels + data_labels + data_labels2
        points = cube_points + data_points + data_points2
        d = 0.05

        for t, p in zip(labels, points):
            ax1.plot(*p,'o')
            if t > '(a':
                va = 'top'
                yd = -0.05
            else:
                va = 'bottom'
                yd = 0
            ax1.text(p[0]+d, p[1]+d+yd, t, fontsize=16, va=va)

        ax1.set_xlim(-1.0, 2.5)
        ax1.set_ylim(-1.5, 2.0)
        ax1.set_aspect('equal')

        cube = Polygon(cube_points, alpha=0.3, color='pink')
        ax1.add_patch(cube)

        matrix0_1 = r"$ v = \begin{bmatrix} a & c \\ b & d \end{bmatrix} u $"
        ax2.text(0.5, 0.8, matrix0_1, fontsize=30, ha='center', va='center', color='w')
        matrix0_2 = r"$ v = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} u $"
        ax3.text(0.5, 0.8, matrix0_2, fontsize=30, ha='center', va='center', color='w')

        matrix1_11 = r"$ \begin{bmatrix} \_ \\ \_ \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} $"
        matrix1_12 = r"$ \begin{bmatrix} \_ \\ \_ \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} $"
        matrix1_21 = r"$ \begin{bmatrix} a \\ \_ \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} $"
        matrix1_22 = r"$ \begin{bmatrix} 1 \\ \_ \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} $"
        matrix1_31 = r"$ \begin{bmatrix} a \\ b \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} $"
        matrix1_32 = r"$ \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} $"

        mt11 = ax2.text(0.5, 0.5, matrix1_11, fontsize=30, ha='center', va='center', color='w')
        mt12 = ax3.text(0.5, 0.5, matrix1_12, fontsize=30, ha='center', va='center', color='w')

        matrix2_11 = r"$ \begin{bmatrix} \_ \\ \_ \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} $"
        matrix2_12 = r"$ \begin{bmatrix} \_ \\ \_ \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} $"
        matrix2_21 = r"$ \begin{bmatrix} c \\ \_ \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} $"
        matrix2_22 = r"$ \begin{bmatrix} 1 \\ \_ \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} $"
        matrix2_31 = r"$ \begin{bmatrix} c \\ d \end{bmatrix} = \begin{bmatrix} a & c \\ b & d \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} $"
        matrix2_32 = r"$ \begin{bmatrix} 1 \\ -1 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} $"

        mt21 = ax2.text(0.5, 0.2, matrix2_11, fontsize=30, ha='center', va='center', color='w')
        mt22 = ax3.text(0.5, 0.2, matrix2_12, fontsize=30, ha='center', va='center', color='w')

        def on_reset():
            print('on_reset')
            mt11.set_text(matrix1_11)
            mt12.set_text(matrix1_12)

        collection = Collection(self.num_frames, ret_artists=[mt11, mt12, mt21, mt22], on_reset=on_reset)

        anim1_4 = Animation(0, 20)
        circles1 = Circles(ax1, [cube_points[1]], radius=0.03, color='red')
        circles1.set_target_positions([data_points[0]])
        anim1_4.append(circles1)
        collection.append(anim1_4)

        # a <= (1,0)
        v10 = (['$1$', '$0$'], [(0.67, 0.53), (0.67, 0.44)])
        tg1 = TextGroup(ax2, *v10, fontsize=30)
        tg1.set_target_positions([(0.47, 0.53), (0.54, 0.53)])
        tg1_ = TextGroup(ax3, *v10, fontsize=30)
        tg1_.set_target_positions([(0.47, 0.53), (0.54, 0.53)])

        anim1 = Animation( 0,  5)
        anim1.append(tg1)
        anim1.append(tg1_)
        collection.append(anim1)

        # a <= a
        tg2 = TextGroup(ax2, ['$a$'], [(0.47, 0.53)], fontsize=30)
        tg2.set_target_positions([(0.26, 0.53)])
        tg2_ = TextGroup(ax3, ['$1$'], [(0.47, 0.53)], fontsize=30)
        tg2_.set_target_positions([(0.26, 0.53)])

        def on_stop_anim2():
            mt11.set_text(matrix1_21)
            mt12.set_text(matrix1_22)

        anim2 = Animation( 5, 10, on_stop=on_stop_anim2 )
        anim2.append(tg2)
        anim2.append(tg2_)
        collection.append(anim2)

        # b <= (1,0)
        tg3 = TextGroup(ax2, *v10, fontsize=30)
        tg3.set_target_positions([(0.47, 0.44), (0.54, 0.44)])
        tg3_ = TextGroup(ax3, *v10, fontsize=30)
        tg3_.set_target_positions([(0.47, 0.44), (0.54, 0.44)])

        anim3 = Animation(10, 15)
        anim3.append(tg3)
        anim3.append(tg3_)
        collection.append(anim3)

        # b <= b
        tg4 = TextGroup(ax2, ['$b$'], [(0.47, 0.43)], fontsize=30)
        tg4.set_target_positions([(0.26, 0.43)])
        tg4_ = TextGroup(ax3, ['$1$'], [(0.47, 0.43)], fontsize=30)
        tg4_.set_target_positions([(0.26, 0.43)])

        def on_stop_anim4():
            mt11.set_text(matrix1_31)
            mt12.set_text(matrix1_32)

        anim4 = Animation(15, 20, on_stop=on_stop_anim4 )
        anim4.append(tg4)
        anim4.append(tg4_)
        collection.append(anim4)

        anim5_8 = Animation(20, 40)
        circles2 = Circles(ax1, [cube_points[3]], radius=0.03, color='red')
        circles2.set_target_positions([data_points[2]])
        anim5_8.append(circles2)
        collection.append(anim5_8)

        # c <= (0,1)
        v01 = ['$0$', '$1$'], [(0.67, 0.23), (0.67, 0.13)]
        tg5 = TextGroup(ax2, *v01, fontsize=30)
        tg5.set_target_positions([(0.47, 0.23), (0.54, 0.23)])
        tg5_ = TextGroup(ax3, *v01, fontsize=30)
        tg5_.set_target_positions([(0.47, 0.23), (0.54, 0.23)])
        anim5 = Animation(20, 25)
        anim5.append(tg5)
        anim5.append(tg5_)
        collection.append(anim5)

        # c <= c
        tg6 = TextGroup(ax2, ['$c$'], [(0.54, 0.23)], fontsize=30)
        tg6.set_target_positions([(0.26, 0.23)])
        tg6_ = TextGroup(ax3, ['$1$'], [(0.54, 0.23)], fontsize=30)
        tg6_.set_target_positions([(0.26, 0.23)])

        def on_stop_anim6():
            mt21.set_text(matrix2_21)
            mt22.set_text(matrix2_22)

        anim6 = Animation(25, 30, on_stop=on_stop_anim6 )
        anim6.append(tg6)
        anim6.append(tg6_)
        collection.append(anim6)

        # d <= (0,1)
        tg7 = TextGroup(ax2, *v01, fontsize=30)
        tg7.set_target_positions([(0.47, 0.13), (0.54, 0.13)])
        tg7_ = TextGroup(ax3, *v01, fontsize=30)
        tg7_.set_target_positions([(0.47, 0.13), (0.54, 0.13)])
        anim7 = Animation(30, 35)
        anim7.append(tg7)
        anim7.append(tg7_)
        collection.append(anim7)

        # d <= d
        tg8 = TextGroup(ax2, ['$d$'], [(0.54, 0.13)], fontsize=30)
        tg8.set_target_positions([(0.26, 0.13)])
        tg8_ = TextGroup(ax3, ['$-1$'], [(0.54, 0.13)], fontsize=30)
        tg8_.set_target_positions([(0.26, 0.13)])

        def on_stop_anim8():
            mt21.set_text(matrix2_31)
            mt22.set_text(matrix2_32)

        anim8 = Animation(35, 40, on_stop=on_stop_anim8 )
        anim8.append(tg8)
        anim8.append(tg8_)
        collection.append(anim8)

        anim9 = Animation(40, 45)
        collection.append(anim9)

        # print('index=', collection.index)
        para1 = Parallelogram(ax1, cube_points, alpha=0.3, color='cyan')
        para1.set_target_vertices([(0,0)] + data_points)
        anim_10 = Animation(45, 60)
        anim_10.append(para1)
        collection.append(anim_10)

        anim_11 = Animation(60, 75)
        para1_r = para1.make_reversed(ax1, alpha=0.3, color='cyan')
        para2 = Parallelogram(ax1, cube_points, alpha=0.3, color='yellow')
        inv = np.linalg.inv(np.array([[1,1], [1,-1]]))
        para2.set_target_vertices([(0,0), inv[:,0], inv[:,0]+inv[:,1], inv[:,1]])

        anim_11.append(para1_r)
        anim_11.append(para2)

        collection.append(anim_11)

        self.anim = collection.make_animation(self.fig)

    def save(self):
        import os
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        path = os.path.join(os.getcwd(), "anim.mp4")
        self.anim.save(path, writer="ffmpeg")
        MessageBox.showinfo("Save Notification", 'The movie has been saved to "%s"' % path, parent=self)
