# coding: utf-8
"""
    LowPercentileSpline.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from BasePlane import LowPercentilePlane
import molass_legacy.KekLib.DebugPlot as plt
# import matplotlib.pyplot as plt
from molass_legacy.KekLib.OurMatplotlib import get_color

USE_OVERLAPPING_PLANES  = False
USE_AVERAGED_JOINTS     = False
USE_GEOMDL_FITTING      = False

def bspline_surface(control_points, num_planes, fig, axis_recs):
    # import matplotlib.pyplot as mplt
    if USE_GEOMDL_FITTING:
        from geomdl import fitting
    else:
        from geomdl import BSpline , utilities
    from geomdl.visualization.VisMPL import VisConfig
    from OurGeomdl import VisSurface

    if USE_OVERLAPPING_PLANES:
        n = num_planes + 2
    elif USE_AVERAGED_JOINTS:
        n = num_planes*2 + 1
    else:
        n = num_planes + 1

    if USE_GEOMDL_FITTING:
        size_u = n
        size_v = 3
        degree_u = 3
        degree_v = 2

        # Do global surface approximation
        surf = fitting.approximate_surface(control_points, size_u, size_v, degree_u, degree_v)
    else:
        # Create a surface instance
        surf = BSpline.Surface()

        # Set degrees
        surf.degree_u = 3
        surf.degree_v = 2

        # Set control points
        surf.set_ctrlpts(control_points, n, 3)

        # Auto-generate knot vectors
        surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, surf.ctrlpts_size_u)
        surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, surf.ctrlpts_size_v)

        # Set sample size
        # surf.sample_size = 25

        # Evaluate surface
        surf.evaluate()

    # Set the visualization component
    alpha = 0.5
    config = VisConfig(alpha=alpha)

    for ax, ax_lim in axis_recs:
        vis_component = VisSurface(config=config, fig=fig, ax=ax, alpha=alpha)
        surf.vis = vis_component

        # Plot the surface
        surf.render(plot=False)
        # axes = fig.get_axes()
        # print(len(axes))
        # ax = fig.gca()
        ax.set_xlim(ax_lim.get_xlim())
        ax.set_ylim(ax_lim.get_ylim())
        ax.set_zlim(ax_lim.get_zlim())
        # plt.show()

class LowPercentileSpline:
    def __init__(self, sd):
        array = sd.intensity_array[:,:,1].T

        full_width = 40
        half_width = full_width//2
        num_planes = 15

        q = sd.intensity_array[0,:,0]
        i = np.arange(array.shape[0])
        j = np.arange(array.shape[1])
        ii, jj = np.meshgrid(i, j)
        xx = q[ii]
        zz = array[ii,jj]

        smp_i = (sd.xray_slice.start + sd.xray_slice.stop)//2
        iy = slice(0, array.shape[1])

        control_points = []
        lpm_list = []
        abc_list = []
        for k in range(num_planes):
            if USE_OVERLAPPING_PLANES:
                start = k*half_width
                stop = (k+1)*full_width
            else:
                start = k*full_width
                stop = (k+1)*full_width

            data = array[start:stop,:]

            zpp = np.percentile(data, 30)
            wpp = np.where( data < zpp )
            x_ = q[start+wpp[0]]
            y_ = wpp[1]
            z_ = data[wpp[0], wpp[1]]
            lpm_list.append([x_, y_, z_])
            ix = slice(start, stop)
            lpp = LowPercentilePlane(array, q, sd.ivector, ix, iy, smp_i, debug=False)
            abc = lpp.get_params()
            abc_list.append([ix, abc])

            a, b, c = abc
            middle = (start + stop)//2

            if USE_OVERLAPPING_PLANES:
                if k == 0:
                    index = [start, middle]
                elif k == num_planes - 1:
                    index = [middle, stop]
                else:
                    index = [middle]

                for m, x in enumerate(q[index]):
                    for n, y in enumerate([iy.start, (iy.start+iy.stop)/2, iy.stop]):
                        z = a*x + b*y + c
                        point = [x, y, z]
                        control_points.append(point)
            elif USE_AVERAGED_JOINTS:
                index = [start, middle, stop]
                keep_points = []
                for m, x in enumerate(q[index]):
                    for n, y in enumerate([iy.start, (iy.start+iy.stop)/2, iy.stop]):
                        z = a*x + b*y + c
                        point = [x, y, z]
                        if m == 0:
                            if k == 0:
                                control_points.append(point)
                            else:
                                last_point = kept_points[n]
                                average_point = list(np.average([last_point, point], axis=0))
                                control_points.append(average_point)
                        elif m == 1:
                            control_points.append(point)
                        elif m == 2:
                            if k < num_planes - 1:
                                keep_points.append(point)
                            else:
                                control_points.append(point)
                kept_points = keep_points
            else:
                index = [start] if k < num_planes - 1 else [start, stop]
                for m, x in enumerate(q[index]):
                    for n, y in enumerate([iy.start, (iy.start+iy.stop)/2, iy.stop]):
                        z = a*x + b*y + c
                        point = [x, y, z]
                        control_points.append(point)

        debug = True
        if debug:
            fig = plt.figure(figsize=(24,14))
            ax1 = fig.add_subplot(231, projection='3d')
            ax2 = fig.add_subplot(232, projection='3d')
            ax3 = fig.add_subplot(233, projection='3d')
            ax4 = fig.add_subplot(234, projection='3d')
            ax5 = fig.add_subplot(235, projection='3d')
            ax6 = fig.add_subplot(236, projection='3d')

            ax1.plot_surface(xx, jj, zz)

            for k, rec in enumerate(lpm_list):
                x_, y_, z_ = rec

                ix, abc = abc_list[k]
                xs = q[ix]
                ys = np.arange( iy.start, iy.stop )
                xx, yy = np.meshgrid( np.linspace(xs[0], xs[-1], 10), np.linspace(ys[0], ys[-1], 10) )
                A, B, C = abc
                zz = xx * A + yy * B + C

                color = get_color(k)
                for ax in [ax1, ax4]:
                    ax.plot(x_, y_, z_, 'o', markersize=1, color=color)

                for ax in [ax2, ax3, ax5, ax6]:
                    ax.plot_surface(xx, yy, zz, color='red', alpha=0.1 )

            x_, y_, z_ = zip(*control_points)
            for ax in [ax2, ax5]:
                ax.plot(x_, y_, z_, 'o')

            ax2.set_xlim(ax1.get_xlim())
            ax2.set_ylim(ax1.get_ylim())
            ax2.set_zlim(ax1.get_zlim())
            ax4.set_xlim(ax5.get_xlim())
            ax4.set_ylim(ax5.get_ylim())
            ax4.set_zlim(ax5.get_zlim())

            axis_recs = [[ax3, ax1], [ax6, ax5]]
            bspline_surface(control_points, num_planes, fig, axis_recs)

            fig.tight_layout()
            plt.show()
            plt.close()
