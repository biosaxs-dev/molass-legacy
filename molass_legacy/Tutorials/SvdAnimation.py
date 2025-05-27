# coding: utf-8
"""
    SvdAnimation.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Polygon, Circle
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from molass_legacy.KekLib.OurMatplotlib import get_color
from OurManim import manim_init, use_default_style, Animation, Collection, TextGroup, Parallelogram, rotation, angle
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar

class SvdAnimation(Dialog):
    def __init__(self, parent=None):
        if parent is None:
            from SerialTestUtils import get_tk_root
            parent = get_tk_root()
        self.parent = parent
        self.num_frames = 100
        manim_init()
        Dialog.__init__(self, parent, "SVD Tutorial", visible=False)

    def show(self):
        self._show()

    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.fig, self.axes = plt.subplots(nrows=1, ncols=2, figsize=(16,8))
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

        square_labels = ['(0,0)', '(1,0)', '(1,1)', '(0,1)']
        square_points = [(0,0), (1,0), (1,1), (0,1)]
        data_labels = ['(a,b)', '(a+c, b+d)', '(c,d)']

        R1 = rotation(-np.pi/4)
        SG = np.diag([4, 1.5])
        R2 = rotation(np.pi/6)
        for x in [R1, SG, R2]:
            print(x)
        F = R1@SG@R2
        print('F=', F)
        f1 = F[:,0]
        f2 = F[:,1]
        data_points = [f1, f1+f2, f2]

        labels = square_labels + data_labels
        points = square_points + data_points
        d = 0.05

        for ax, t in zip(self.axes, [" (cumulative)", " (separative)"]):
            ax.set_title("SVD animation" + t, fontsize=30)
            ax.set_xlim(-4.0, 4.0)
            ax.set_ylim(-4.0, 4.0)
            ax.set_aspect('equal')
            if ax == self.axes[0]:
                labels_ = labels
                points_ = points
            else:
                labels_ = square_labels
                points_ = square_points
            for t, p in zip(labels_, points_):
                ax.plot(*p,'o')
                ax.text(p[0]+d, p[1]+d, t, fontsize=16)
            square =  Polygon(square_points, alpha=0.3, color='pink')
            ax.add_patch(square)

        collection = Collection(self.num_frames)

        final_target = [(0,0)] + data_points
        print('final_target=', final_target)

        para1 = Parallelogram(ax1, square_points, alpha=0.3, color='cyan')
        para1.set_target_vertices(final_target)
        anim_1 = Animation(0, 20)
        anim_1.append(para1)
        collection.append(anim_1)

        M = np.array([ final_target[1], final_target[3] ]).T 
        print('M=', M)
        svd = np.linalg.svd(M)
        for x in svd:
            print(x)

        # U, s, VT = svd
        U, s, VT = R1, np.diag(SG), R2

        v1 = VT[:,0]
        v2 = VT[:,1]
        VT_points = [(0,0), v1, v1+v2, v2]
        para2 = Parallelogram(ax1, square_points, motion='rotation', alpha=0.3, color='cyan')
        para2.set_target_vertices(VT_points)
        para2_ = Parallelogram(ax2, square_points, motion='rotation', alpha=0.3, color='cyan')
        para2_.set_target_vertices(VT_points)
        anim_2 = Animation(20, 40)
        anim_2.append(para2)
        anim_2.append(para2_)
        collection.append(anim_2)

        sVT = np.dot(np.diag(s), VT)
        sv1 = sVT[:,0]
        sv2 = sVT[:,1]
        sVT_points = [(0,0), sv1, sv1+sv2, sv2]
        para3 = Parallelogram(ax1, VT_points, alpha=0.3, color='cyan')
        para3.set_target_vertices(sVT_points)
        sv1_ = SG[:,0]
        sv2_ = SG[:,1]
        SG_points = [(0,0), sv1_, sv1_+sv2_, sv2_]
        para3_ = Parallelogram(ax2, square_points, alpha=0.3, color='cyan')
        para3_.set_target_vertices(SG_points)
        anim_3 = Animation(40, 60)
        anim_3.append(para3)
        anim_3.append(para3_)
        collection.append(anim_3)

        para4 = Parallelogram(ax1, sVT_points, motion='rotation', alpha=0.3, color='cyan')
        para4.set_target_vertices(final_target)
        u1 = U[:,0]
        u2 = U[:,1]
        U_points = [(0,0), u1, u1+u2, u2]
        para4_ = Parallelogram(ax2, square_points, motion='rotation', alpha=0.3, color='cyan')
        para4_.set_target_vertices(U_points)
        anim_4 = Animation(60, 80)
        anim_4.append(para4)
        anim_4.append(para4_)
        collection.append(anim_4)

        self.anim = collection.make_animation(self.fig)

    def save(self):
        import os
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        path = os.path.join(os.getcwd(), "anim.mp4")
        self.anim.save(path, writer="ffmpeg")
        MessageBox.showinfo("Save Notification", 'The movie has been saved to "%s"' % path, parent=self)
