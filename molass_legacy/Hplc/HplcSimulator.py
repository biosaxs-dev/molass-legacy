# coding: utf-8
"""
    HplcSimulator.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np
import matplotlib.pyplot as mplt
from matplotlib.gridspec import GridSpec
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d import proj3d
import matplotlib.animation as animation
from shapely.geometry import LineString, Point
from descartes import PolygonPatch
import molass_legacy.KekLib.DebugPlot as plt

"""
From:
Add cylinder to plot
https://stackoverflow.com/questions/26989131/add-cylinder-to-plot
"""

RADIUS = 0.3
MOVE_DELTA = 0.1
TITLE_FONTSIZE = 32

COLUMN_CENTER_Y = 0.5
COLUMN_CENTER_Z = 0.5
COLUMN_LENGTH = 3
COUNTER_LENGTH = 0.1
START_LENGTH = 0.1

def data_for_cylinder_along_z(center_x, center_y, radius, min_height, max_height, res_h=50, res_c=50):
    z = np.linspace(min_height, max_height, res_h)
    theta = np.linspace(0, 2*np.pi, res_c)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

"""
Generate a random point within a circle (uniformly)
https://stackoverflow.com/questions/5837572/generate-a-random-point-within-a-circle-uniformly
"""
def uniform_random_disc(n, radius):
    r = radius * np.sqrt(np.random.rand(n))
    theta = np.random.rand(n) * 2 * np.pi
    return r * np.cos(theta), r * np.sin(theta)

class Cylinder:
    def __init__(self, xmin, xmax, radius):
        self.xmin = xmin
        self.xmax = xmax
        self.radius = radius
        self.ycenter = COLUMN_CENTER_Y
        self.zcenter = COLUMN_CENTER_Z

class PaticleGroup:
    def __init__(self, n, cylinder, color=None, speed=0):
        self.n = n
        self.r = cylinder.radius
        self.cy = cylinder.ycenter
        self.cz = cylinder.zcenter
        self.xmin = cylinder.xmin
        self.xmax = cylinder.xmax
        self.color = color
        self.speed = speed
        th = -np.pi/2
        c = np.cos(th)
        s = np.sin(th)
        self.rotation = np.array([[c, -s], [s, c] ])
        self.set_init_positions()

    def set_init_positions(self):
        r = self.r
        cy = self.cy
        cz = self.cz
        self.circle = Point(cy, cz).buffer(r)
        n = self.n
        self.x = self.xmin + (self.xmax - self.xmin) * np.random.rand(n)
        y_, z_ = uniform_random_disc(n, r)
        self.y = cy + y_
        self.z = cz + z_

    def plot(self, ax):
        self.artists, = ax.plot(self.x, self.y, self.z, 'o', color=self.color, markersize=1, alpha=0.5)

    def get_artists(self):
        return [self.artists]

    def reset(self):
        self.set_init_positions()
        self.artists.set_data(self.x, self.y)
        self.artists.set_3d_properties(self.z)

    def compute_bounced_coord(self, ty, tz, ey, ez):
        p = np.array([(ey - ty), (ez - tz)]).T
        v = np.array([(self.cy - ty), (self.cz - tz)]).T/self.r
        u = np.dot(self.rotation, v)
        W = np.array([u, v]).T
        W_ = np.linalg.inv(W)
        q_ = np.dot(W_, p)
        q = np.dot(W, np.array([q_[0], -q_[1]]))
        return ty+q[0], tz+q[1]

    def random_move(self, delta):
        n = self.n
        self.x += (np.random.rand(n) - 0.5) * delta + self.speed * delta
        y = self.y + (np.random.rand(n) - 0.5) * delta
        z = self.z + (np.random.rand(n) - 0.5) * delta

        r = self.r
        cy = self.cy
        cz = self.cz
        circle = self.circle
        for k in np.where((y - cy)**2 + (z - cz)**2 > r**2)[0]:
            segment = LineString([(self.y[k], self.z[k]), (y[k], z[k])])
            coords = circle.intersection(segment).coords
            if len(coords) == 0:
                # this case does occur
                segment = LineString([(cy, cz), (y[k], z[k])])
                coords = circle.intersection(segment).coords
                if len(coords) == 0:
                    continue

            q = self.compute_bounced_coord(*coords[-1], y[k], z[k])

            if False:
                plt.push()
                fig = plt.figure()
                ax = fig.gca()
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal')
                patch = PolygonPatch(circle, alpha=0.1)
                ax.add_patch(patch)
                for c in coords:
                    ax.plot(*c, 'o')
                ax.plot(y[k], z[k], 'o', color='red')
                ax.plot([cy, c[0]], [cz, c[1]], color='yellow')
                ax.plot(*q, 'o', color='green')
                plt.show()
                plt.pop()
                exit()

            y[k] = q[0]
            z[k] = q[1]

        self.y = y
        self.z = z
        self.artists.set_data(self.x, self.y)
        self.artists.set_3d_properties(self.z)

    def get_num_detected(self):
        return np.sum(np.logical_and(self.x > 2.9, self.x < 3.0))

class HplcSimulator:
    def __init__(self, num_particles=5000, num_frames=250, start_length=START_LENGTH, plt=mplt):
        self.num_particles = num_particles
        self.num_frames = num_frames
        self.start_length = start_length
        self.plt = plt
        self.fig = fig = plt.figure(figsize=(18,10))
        gs = GridSpec(2, 3)
        ax0 = fig.add_subplot(gs[0,:], projection='3d')
        # print('dir(ax1)=', dir(ax1))
        ax1 = fig.add_subplot(gs[1,0])
        ax2 = fig.add_subplot(gs[1,1])
        ax3 = fig.add_subplot(gs[1,2])
        self.axes = ax0, ax1, ax2, ax3

    def play(self, save_only=False):
        ax0, ax1, ax2, ax3 = self.axes

        Xc, Yc, Zc = data_for_cylinder_along_z(COLUMN_CENTER_Y, COLUMN_CENTER_Z, RADIUS, 0, COLUMN_LENGTH)
        entire_column = ax0.plot_surface(Zc, Xc, Yc, alpha=0.1)
        Xc, Yc, Zc = data_for_cylinder_along_z(COLUMN_CENTER_Y, COLUMN_CENTER_Z, RADIUS, COLUMN_LENGTH-COUNTER_LENGTH, COLUMN_LENGTH, res_h=5)
        counter_column = ax0.plot_surface(Zc, Xc, Yc, alpha=0.5, color='yellow')

        N = self.num_particles
        ax0.set_title("Conceptully Simulated HPLC Column (%d red, %d blue Particles)" % (N, N), fontsize=TITLE_FONTSIZE, y=1.1)
        ax1.set_title("Number of Particles", fontsize=TITLE_FONTSIZE)
        ax2.set_title("UV Absorbance", fontsize=TITLE_FONTSIZE)
        ax3.set_title("Xray Scattering", fontsize=TITLE_FONTSIZE)

        ax0.set_xlim3d(0.6, 2.5)
        ax0.set_ylim3d(0, 1)
        shrink = 0.25
        ax0.set_zlim3d(shrink, 1-shrink)
        ax0.set_axis_off()
        ax0.view_init(0, -90)

        tx, ty, _ = proj3d.proj_transform(2.95, 0.5, 0.2, ax0.get_proj())
        ax0.annotate("Counter Ring", xy=(tx, ty), xytext=(tx, ty-0.02), ha='center', arrowprops=dict( arrowstyle="->", color='k' ) )

        start_column = Cylinder(0, self.start_length, RADIUS)
        red_particles = PaticleGroup(self.num_particles, start_column, color='red', speed=0.2)
        red_particles.plot(ax0)
        blue_particles = PaticleGroup(self.num_particles, start_column, color='blue', speed=0.4)
        blue_particles.plot(ax0)

        hx = np.arange(self.num_frames)
        zeros = np.zeros(self.num_frames)
        hy1 = zeros.copy()
        hy2 = zeros.copy()
        hy = zeros.copy()

        hist1, = ax1.plot(hx, hy1, ':', color='red', label='red particles')
        hist2, = ax1.plot(hx, hy2, ':', color='blue', label='blue particles')
        hist, = ax1.plot(hx, hy, color='green', alpha=0.2, label='total')
        ax1.legend()

        artists = []
        hg_artists = [hist1, hist2, hist]
        hg_y_s = [hy1, hy2, hy]
        artists += hg_artists

        ax1.set_xlim(0, self.num_frames)
        ax1.set_ylim(0, 0.2*self.num_particles)

        def scaled_plot(ax, scale1, scale2):
            y1 = hy1*uv_scale1
            y2 = hy2*uv_scale2
            y = y1 + y2
            ret1, = ax.plot(hx, y1, ':', color='red', label='red particles')
            ret2, = ax.plot(hx, y2, ':', color='blue', label='blue particles')
            ret, = ax.plot(hx, y, color='green', alpha=0.2, label='total')
            ax.legend()
            return [ret1, ret2, ret], [y1, y2, y]

        uv_scale1 = 0.1/self.num_frames
        uv_scale2 = 0.2/self.num_frames

        ax2.set_xlim(0, self.num_frames)
        ax2.set_ylim(0, 0.2*self.num_particles*max(uv_scale1, uv_scale2))

        uv_artists, uv_y_s = scaled_plot(ax2, uv_scale1, uv_scale2)
        artists += uv_artists

        xr_scale1 = 0.08/self.num_frames
        xr_scale2 = 0.05/self.num_frames

        ax3.set_xlim(0, self.num_frames)
        ax3.set_ylim(0, 0.2*self.num_particles*max(xr_scale1, xr_scale2))

        xr_artists, xr_y_s = scaled_plot(ax3, xr_scale1, xr_scale2)
        artists += xr_artists

        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.92, hspace=0.2)

        def reset():
            red_particles.reset()
            blue_particles.reset()
            for k, a in enumerate(hg_artists):
                hg_y_s[k] = y = zeros.copy()
                a.set_data(hx, y)
            for k, a in enumerate(uv_artists):
                uv_y_s[k] = y = zeros.copy()
                a.set_data(hx, y)
            for k, a in enumerate(xr_artists):
                xr_y_s[k] = y = zeros.copy()
                a.set_data(hx, y)
            return red_particles.get_artists() + blue_particles.get_artists() + artists

        def animate(i):
            red_particles.random_move(MOVE_DELTA)
            blue_particles.random_move(MOVE_DELTA)
            c1 = red_particles.get_num_detected()
            c2 = blue_particles.get_num_detected()

            def set_data_to_artists(artists_, y_s, scale1, scale2):
                y1, y2, y = y_s
                y1[i] = c1*scale1
                y2[i] = c2*scale2
                y[i] = y1[i] + y2[i]
                for a, ay in zip(artists_, y_s):
                    a.set_data(hx, ay)

            set_data_to_artists(hg_artists, hg_y_s, 1, 1)
            set_data_to_artists(uv_artists, uv_y_s, uv_scale1, uv_scale2)
            set_data_to_artists(xr_artists, xr_y_s, xr_scale1, xr_scale2)

            return red_particles.get_artists() + blue_particles.get_artists() + artists

        self.anim = animation.FuncAnimation(self.fig, animate, frames=self.num_frames, blit=True, init_func=reset)

        if save_only:
            pass
        else:
            self.plt.show()

    def save(self, file, **kwargs):
        self.anim.save(file, **kwargs)
