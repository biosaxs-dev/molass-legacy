"""
    OurPlotUtils.py

    Copyright (c) 2019-2020, Masatsuyo Takahashi, KEK-PF
"""

def draw_as_image(ax, from_fig, from_ax, exp_elements=False):
    """
    https://stackoverflow.com/questions/4325733/save-a-subplot-in-matplotlib
    https://stackoverflow.com/questions/35286540/display-an-image-with-python
    """
    import io
    import matplotlib.image as mpimg
    buffer = io.BytesIO()
    if exp_elements:
        """
        see also:
        https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.axes.Axes.get_window_extent.html
        https://matplotlib.org/3.1.0/api/_as_gen/matplotlib.axes.Axes.get_tightbbox.html#matplotlib.axes.Axes.get_tightbbox
        """
        renderer = from_fig.canvas.get_renderer()
        extent = from_ax.get_tightbbox(renderer).transformed(from_fig.dpi_scale_trans.inverted())
    else:
        extent = from_ax.get_window_extent().transformed(from_fig.dpi_scale_trans.inverted())
    from_fig.savefig(buffer, bbox_inches=extent)
    buffer.seek(0)
    img=mpimg.imread(buffer)
    ax.imshow(img)
