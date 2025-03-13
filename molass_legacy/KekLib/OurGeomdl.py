"""
    OurGeomdl.py

    modified from geomdl/visualization/VisMPL.VisSurface

"""

from geomdl import vis
from geomdl.visualization.VisMPL import VisConfig
import numpy as np
import logging
import matplotlib as mpl
import matplotlib.tri as mpltri
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import animation

class VisSurface(vis.VisAbstract):
    """ Matplotlib visualization module for surfaces.

    Wireframe plot for the control points and triangulated plot (using ``plot_trisurf``) for the surface points.
    The surface is triangulated externally using :py:func:`.utilities.make_triangle_mesh()` function.
    """
    def __init__(self, config=VisConfig(), **kwargs):
        super(VisSurface, self).__init__(config, **kwargs)
        self._module_config['ctrlpts'] = "quads"
        self._module_config['evalpts'] = "triangles"
        self.fig = kwargs.pop('fig', None)
        self.ax = kwargs.pop('ax', None)
        alpha = kwargs.pop('alpha', None)
        if alpha is not None:
            self.vconf.alpha = alpha
        self.logger = logging.getLogger( __name__ )
        self.logger.info( "using modified version of geomdl.visualization.VisMPL.VisSurface. alpha=%g" % self.vconf.alpha )

    def animate(self, **kwargs):
        """ Animates the surface.

        This function only animates the triangulated surface. There will be no other elements, such as control points
        grid or bounding box.

        Keyword arguments:
            * ``colormap``: applies colormap to the surface

        Colormaps are a visualization feature of Matplotlib. They can be used for several types of surface plots via
        the following import statement: ``from matplotlib import cm``

        The following link displays the list of Matplolib colormaps and some examples on colormaps:
        https://matplotlib.org/tutorials/colors/colormaps.html
        """
        # Calling parent render function
        super(VisSurface, self).render(**kwargs)

        # Colormaps
        surf_cmaps = kwargs.get('colormap', None)

        # Initialize variables
        tri_idxs = []
        vert_coords = []
        trisurf_params = []
        frames = []
        frames_tris = []
        num_vertices = 0

        # Start plotting of the surface and the control points grid
        fig = plt.figure(figsize=self.vconf.figure_size, dpi=self.vconf.figure_dpi)
        ax = Axes3D(fig)

        # Start plotting
        surf_count = 0
        for plot in self._plots:
            # Plot evaluated points
            if plot['type'] == 'evalpts' and self.vconf.display_evalpts:
                # Use internal triangulation algorithm instead of Qhull (MPL default)
                verts = plot['ptsarr'][0]
                tris = plot['ptsarr'][1]
                # Extract zero-indexed vertex number list
                tri_idxs += [[ti + num_vertices for ti in tri.data] for tri in tris]
                # Extract vertex coordinates
                vert_coords += [vert.data for vert in verts]
                # Update number of vertices
                num_vertices = len(vert_coords)

                # Determine the color or the colormap of the triangulated plot
                params = {}
                if surf_cmaps:
                    try:
                        params['cmap'] = surf_cmaps[surf_count]
                        surf_count += 1
                    except IndexError:
                        params['color'] = plot['color']
                else:
                    params['color'] = plot['color']
                trisurf_params += [params for _ in range(len(tris))]

        # Pre-processing for the animation
        pts = np.array(vert_coords, dtype=self.vconf.dtype)

        # Create the frames (Artists)
        for tidx, pidx in zip(tri_idxs, trisurf_params):
            frames_tris.append(tidx)
            # Create MPL Triangulation object
            triangulation = mpltri.Triangulation(pts[:, 0], pts[:, 1], triangles=frames_tris)
            # Use custom Triangulation object and the choice of color/colormap to plot the surface
            p3df = ax.plot_trisurf(triangulation, pts[:, 2], alpha=self.vconf.alpha, **pidx)
            # Add to frames list
            frames.append([p3df])

        # Create MPL ArtistAnimation
        ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=1000)

        # Remove axes
        if not self.vconf.display_axes:
            plt.axis('off')

        # Set axes equal
        if self.vconf.axes_equal:
            self.vconf.set_axes_equal(ax)

        # Axis labels
        if self.vconf.display_labels:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        # Process keyword arguments
        fig_filename = kwargs.get('fig_save_as', None)
        fig_display = kwargs.get('display_plot', True)

        # Display the plot
        if fig_display:
            plt.show()
        else:
            fig_filename = self.vconf.figure_image_filename if fig_filename is None else fig_filename

        # Save the figure
        self.vconf.save_figure_as(fig, fig_filename)

        # Return the figure object
        return fig

    def render(self, **kwargs):
        """ Plots the surface and the control points grid.

        Keyword arguments:
            * ``colormap``: applies colormap to the surface

        Colormaps are a visualization feature of Matplotlib. They can be used for several types of surface plots via
        the following import statement: ``from matplotlib import cm``

        The following link displays the list of Matplolib colormaps and some examples on colormaps:
        https://matplotlib.org/tutorials/colors/colormaps.html
        """
        # Calling parent function
        super(VisSurface, self).render(**kwargs)

        # Colormaps
        surf_cmaps = kwargs.get('colormap', None)

        # Initialize variables
        legend_proxy = []
        legend_names = []

        # Start plotting of the surface and the control points grid
        if self.fig is None:
            fig = plt.figure(figsize=self.vconf.figure_size, dpi=self.vconf.figure_dpi)
            ax = Axes3D(fig)
        else:
            fig = self.fig
            ax = self.ax
            assert ax is not None

        surf_count = 0
        # Start plotting
        for plot in self._plots:
            # Plot control points
            if plot['type'] == 'ctrlpts' and self.vconf.display_ctrlpts:
                vertices = [v.data for v in plot['ptsarr'][0]]
                faces = [q.data for q in plot['ptsarr'][1]]
                for q in faces:
                    el = np.array([vertices[i] for i in q], dtype=self.vconf.dtype)
                    el[:, 2] += self._ctrlpts_offset
                    pc3d = Poly3DCollection([el], alpha=0.0, edgecolors=plot['color'], linewidths=1.0, linestyles='-.')
                    pc3d.set_facecolor(None)
                    ax.add_collection3d(pc3d)
                pts = np.array(vertices, dtype=self.vconf.dtype)
                pts[:, 2] += self._ctrlpts_offset
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=plot['color'], linestyle='-.', marker='o')
                plot_proxy = mpl.lines.Line2D([0], [0], linestyle='-.', color=plot['color'], marker='o')
                legend_proxy.append(plot_proxy)
                legend_names.append(plot['name'])

            # Plot evaluated points
            if plot['type'] == 'evalpts' and self.vconf.display_evalpts:
                # Use internal triangulation algorithm instead of Qhull (MPL default)
                verts = plot['ptsarr'][0]
                tris = plot['ptsarr'][1]
                # Extract zero-indexed vertex number list
                tri_idxs = [tri.data for tri in tris]
                # Extract vertex coordinates
                vert_coords = [vert.data for vert in verts]
                pts = np.array(vert_coords, dtype=self.vconf.dtype)

                # Determine the color or the colormap of the triangulated plot
                trisurf_params = {}
                if surf_cmaps:
                    try:
                        trisurf_params['cmap'] = surf_cmaps[surf_count]
                        surf_count += 1
                    except IndexError:
                        trisurf_params['color'] = plot['color']
                else:
                    trisurf_params['color'] = plot['color']

                # Create MPL Triangulation object
                if pts.size != 0:
                    triangulation = mpltri.Triangulation(pts[:, 0], pts[:, 1], triangles=tri_idxs)
                    # Use custom Triangulation object and the choice of color/colormap to plot the surface
                    ax.plot_trisurf(triangulation, pts[:, 2], alpha=self.vconf.alpha, **trisurf_params)
                    # Add to legend
                    plot_proxy = mpl.lines.Line2D([0], [0], linestyle='none', color=plot['color'], marker='^')
                    legend_proxy.append(plot_proxy)
                    legend_names.append(plot['name'])

            # Plot bounding box
            if plot['type'] == 'bbox' and self.vconf.display_bbox:
                pts = np.array(plot['ptsarr'], dtype=self.vconf.dtype)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=plot['color'], linestyle='--')
                plot_proxy = mpl.lines.Line2D([0], [0], linestyle='--', color=plot['color'])
                legend_proxy.append(plot_proxy)
                legend_names.append(plot['name'])

            # Plot trim curves
            if self.vconf.display_trims:
                if plot['type'] == 'trimcurve':
                    pts = np.array(plot['ptsarr'], dtype=self.vconf.dtype)
                    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=plot['color'], marker='o',
                               s=self.vconf.trim_size, depthshade=False)
                    plot_proxy = mpl.lines.Line2D([0], [0], linestyle='none', color=plot['color'], marker='o')
                    legend_proxy.append(plot_proxy)
                    legend_names.append(plot['name'])

            # Plot extras
            if plot['type'] == 'extras':
                pts = np.array(plot['ptsarr'], dtype=self.vconf.dtype)
                ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
                        color=plot['color'][0], linestyle='-', linewidth=plot['color'][1])
                plot_proxy = mpl.lines.Line2D([0], [0], linestyle='-', color=plot['color'][0])
                legend_proxy.append(plot_proxy)
                legend_names.append(plot['name'])

        # Add legend to 3D plot, @ref: https://stackoverflow.com/a/20505720
        if self.vconf.display_legend:
            ax.legend(legend_proxy, legend_names, numpoints=1)

        # Remove axes
        if not self.vconf.display_axes:
            plt.axis('off')

        # Set axes equal
        if self.vconf.axes_equal:
            self.vconf.set_axes_equal(ax)

        # Axis labels
        if self.vconf.display_labels:
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')

        # Process keyword arguments
        fig_filename = kwargs.get('fig_save_as', None)
        fig_display = kwargs.get('display_plot', True)

        # Display the plot
        if fig_display:
            plt.show()
        else:
            fig_filename = self.vconf.figure_image_filename if fig_filename is None else fig_filename

        # Save the figure
        self.vconf.save_figure_as(fig, fig_filename)

        # Return the figure object
        return fig
