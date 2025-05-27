# coding: utf-8
"""
    MpInverseAnim3D.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib import animation
from molass_legacy.KekLib.OurTkinter import Tk, Dialog
from molass_legacy.KekLib.TkSupplements import tk_set_icon_portable
from OurManim import manim_init
from OurMatplotlib3D import pathpatch_2d_to_3d, pathpatch_translate, plot_parallelogram3d, plot_parallelepiped
from molass_legacy.KekLib.OurMatplotlib import NavigationToolbar
import molass_legacy.KekLib.DebugPlot as plt

def spike():
    manim_init()

    fig = plt.figure(figsize=(8,8))
    ax10 = fig.add_subplot(111, projection='3d')

    cube_points = [(0,0), (1,0), (1,1), (0,1)]
    cube = Polygon(cube_points, alpha=0.3, color='pink')
    ax10.add_patch(cube)
    pathpatch_2d_to_3d(cube, z=0, normal='z')
    # pathpatch_translate(cube, 0.5)

    ax10.plot([0], [0], [0], 'o', color='white')
    ax10.plot([0], [1], [0], 'o')
    ax10.plot([1], [0], [0], 'o')

    ax10.plot([1], [1], [1], 'o')
    ax10.plot([1], [-1], [2], 'o')

    M = np.array([[1, 1, 1], [1, -1, 2]]).T
    Minv = np.linalg.pinv(M)
    print('Minv=', Minv)

    p0 = np.array((0,0,0))
    p1 = np.array((1,1,1))
    p2 = np.array((1,-1,2))
    plot_parallelogram3d(ax10, p0, p1, p2, alpha=0.3, color='cyan')

    ax10.set_xlim(-2,3)
    ax10.set_ylim(-2,3)
    ax10.set_zlim(-2,3)
    fig.tight_layout()
    plt.show()

class MpInverseAnim3D(Dialog):
    def __init__(self, parent=None):
        if parent is None:
            from molass_legacy.KekLib.TkUtils import get_tk_root
            parent = get_tk_root()
        self.parent = parent
        self.manipulating = False
        self.mouse_axis = None
        self.in_sync = True
        self.num_frames = 85
        manim_init()
        Dialog.__init__(self, parent, "Moore-Penrose Inverse Tutorial (3D)", visible=False)

    def show(self):
        self._show()


    def body(self, body_frame):
        tk_set_icon_portable(self)

        cframe = Tk.Frame(body_frame)
        cframe.pack()

        self.fig = plt.figure(figsize=(18,11))
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, cframe)
        self.mpl_canvas_widget = self.mpl_canvas.get_tk_widget()
        self.mpl_canvas_widget.pack(fill=Tk.BOTH, expand=1)
        self.toolbar = NavigationToolbar(self.mpl_canvas, cframe)
        self.toolbar.update()
        gs = GridSpec(11,2)
        self.ax__ = self.fig.add_subplot(gs[0,:])
        self.ax00 = self.fig.add_subplot(gs[1:3,0])
        self.ax01 = self.fig.add_subplot(gs[1:3,1])
        self.ax10 = self.fig.add_subplot(gs[3:11,0], projection='3d')
        self.ax11 = self.fig.add_subplot(gs[3:11,1], projection='3d')
        # self.ax10.disable_mouse_rotation()        # to avoid a rotation bug?
        self.fig.canvas.mpl_connect( 'draw_event', self.on_draw )
        self.fig.canvas.mpl_connect( 'button_press_event', self.on_button_press )
        self.fig.canvas.mpl_connect( 'button_release_event', self.on_button_release )

        self.draw_texts()
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

    def draw_texts(self):
        for ax in [self.ax__, self.ax00, self.ax01]:
            ax.set_axis_off()

        self.ax__.text(0.5, 0.5, "Moore-Penrose Inverse in Linear Transformation Perspective", ha='center', va='center', fontsize=40)

        self.ax00.text(0.2, 0.5, r"$ v=Mu $", ha='center', va='center', fontsize=50)
        self.ax01.text(0.18, 0.5, r"$ u=M^+v $", ha='center', va='center', fontsize=50)

        matrix00 = r"$ \begin{bmatrix} v_1 \\ v_2 \\v_3 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ 1 & -1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} $"
        self.ax00.text(0.7, 0.5, matrix00, ha='center', va='center', fontsize=28)

        matrix01 = r"$ \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} = \begin{bmatrix}  0.29 & 0.57 & 0.14 \\ 0.07 & -0.35 & 0.29 \end{bmatrix} \begin{bmatrix} v_1 \\ v_2 \\ v_3 \end{bmatrix} $"
        self.ax01.text(0.8, 0.5, matrix01, ha='center', va='center', fontsize=28)

    def draw(self):
        fig = self.fig

        ax10 = self.ax10
        ax11 = self.ax11

        u0 = np.array((0,0,0))
        u1 = np.array((1,0,0))
        u2 = np.array((0,1,0))
        p1 = np.array((1,1,1))
        p2 = np.array((1,-1,2))

        M = np.array([[1, 1, 1], [1, -1, 2]]).T
        Minv = np.linalg.pinv(M)
        print('Minv=', Minv)
        # print('Minv*p1=', np.dot(Minv, p1))
        # print('Minv*p2=', np.dot(Minv, p2))

        def set_xyzlims():
            for ax in [ax10, ax11]:
                ax.set_xlim(-2,3)
                ax.set_ylim(-2,3)
                ax.set_zlim(-2,3)

        def draw_anim(u0, u1, u2, p1, p2, v1, v2):

            square = plot_parallelogram3d(ax10, u0, u1, u2, alpha=0.3, color='pink')

            ax10.plot([0], [0], [0], 'o', color='white')
            ax10.plot([0], [1], [0], 'o')
            ax10.plot([1], [0], [0], 'o')

            ax10.plot([1], [1], [1], 'o')
            ax10.plot([1], [-1], [2], 'o')

            ax10.plot([1,1], [0,1], [0,1], ':', color='red')
            ax10.plot([0,1], [1,-1], [0,2], ':', color='blue')

            # point, = ax10.plot([u1[0]], [u1[1]], [u1[2]], 'o')
            parallel = plot_parallelogram3d(ax10, u0, v1, v2, alpha=0.3, color='cyan')
            ax10.plot(*np.array([u0,v1]).T, color='red')
            ax10.plot(*np.array([u0,v2]).T, color='blue')
            ax10.plot(*np.array([v1,v2]).T, color='orange')
            return parallel

        set_xyzlims()
        draw_anim(u0, u1, u2, p1, p2, u1, u2)

        for ax in [ax10, ax11]:
            # ax.view_init(10, 20)    # better angle
            ax.view_init(13, 25)

        u3 = np.array((0,0,1))
        q1 = np.dot(Minv, u1)
        q2 = np.dot(Minv, u2)
        q3 = np.dot(Minv, u3)
        q1_ = np.zeros(3)
        q1_[0:2] = q1
        q2_ = np.zeros(3)
        q2_[0:2] = q2
        q3_ = np.zeros(3)
        q3_[0:2] = q3

        def draw_reverse_anim(r1, r2, r3, **kwargs):
            plot_parallelepiped(ax11, (0,0,0), [r1, r2, r3], **kwargs)

        num_frames = 50
        num_moves = int(num_frames*0.8)
        w = np.linspace(0, 1, num_moves)

        def animate(i):
            ax10.cla()
            ax11.cla()
            set_xyzlims()
            if i < num_frames:
                i_ = min(num_moves-1, i)        # pause in the end of the first half
                w_ = w[i_]
            else:
                j = i - num_frames
                j_ = max(0, num_moves - 1 - j)  # pause in the end of the second half
                w_ = w[j_]
            v1 = u1*(1-w_) + p1*w_
            v2 = u2*(1-w_) + p2*w_
            draw_anim(u0, u1, u2, p1, p2, v1, v2)
            if i >= num_frames:
                draw_reverse_anim(u1, u2, u3, color='pink', edgecolor='red', alpha=0.3)
                r1 = q1_*(1-w_) + u1*w_
                r2 = q2_*(1-w_) + u2*w_
                r3 = q3_*(1-w_) + u3*w_
                draw_reverse_anim(r1, r2, r3, color='yellow', edgecolor='orange', alpha=0.3)

        self.anim = animation.FuncAnimation(fig, animate, frames=num_frames*2)

    def on_draw(self, event):
        # print('on_draw')
        if self.manipulating or self.in_sync:
            return

        this_axis = self.mouse_axis
        if this_axis == self.ax10:
            other_axis = self.ax11
        elif this_axis == self.ax11:
            other_axis = self.ax10
        else:
            other_axis = None

        if other_axis is not None:
            other_axis.view_init( this_axis.elev, this_axis.azim )

    def on_button_press(self, event):
        # print('on_button_press')
        self.manipulating = True
        self.mouse_axis = event.inaxes

    def on_button_release(self, event):
        # print('on_button_release')
        self.manipulating = False
        self.in_sync = False
        self.mouse_axis = event.inaxes

    def save(self):
        import os
        import molass_legacy.KekLib.CustomMessageBox as MessageBox
        path = os.path.join(os.getcwd(), "anim.mp4")
        self.anim.save(path, writer="ffmpeg")
        MessageBox.showinfo("Save Notification", 'The movie has been saved to "%s"' % path, parent=self)
