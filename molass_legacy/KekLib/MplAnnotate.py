"""
    MplAnnotate.py

    * [How do I find the largest empty space in such images?]
        (https://stackoverflow.com/questions/67421025/how-do-i-find-the-largest-empty-space-in-such-images)
    * [How to read image from in memory buffer (StringIO) or from url with opencv python library]
        (https://stackoverflow.com/questions/13329445/how-to-read-image-from-in-memory-buffer-stringio-or-from-url-with-opencv-pytho)

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import numpy as np
import cv2 as cv

def get_opencv_img_from_buffer(buffer):
    bytes_as_np_array = np.frombuffer(buffer.read(), dtype=np.uint8)
    return cv.imdecode(bytes_as_np_array, flags=cv.IMREAD_UNCHANGED)   # 

def get_image_from_axis(fig, ax):
    import io
    import matplotlib.image as mpimg
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    buffer = io.BytesIO()
    fig.savefig(buffer, bbox_inches=extent)
    buffer.seek(0)
    # img = mpimg.imread(buffer)
    # img = cv.imread(buffer)
    img = get_opencv_img_from_buffer(buffer)
    return img

MARGIN_ADJUST_RATIO = 0.1

def get_annotate_position(ax, debug=False):
    fig = ax.figure
    color_img = get_image_from_axis(fig, ax)
    gray_img = cv.cvtColor(color_img, cv.COLOR_BGR2GRAY)
    kernel = np.ones((5,5),np.uint8)
    erosion = cv.erode(gray_img, kernel, iterations=1)
    thresh = cv.threshold(erosion, 128, 255, cv.THRESH_BINARY)[1]

    # add black border around threshold image to avoid corner being largest distance
    thresh2 = cv.copyMakeBorder(thresh, 1,1,1,1, cv.BORDER_CONSTANT, (0))
    distimg = cv.distanceTransform(thresh2, cv.DIST_L2, 5)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(distimg)
    centx = max_loc[0] - 1  # -1 : adjust by border width
    centy = max_loc[1] - 1  # -1 : adjust by border width
    radius = int(max_val)
    isize, jsize = gray_img.shape   # note that image shape (i, j) corresponds to (y, x)
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    if debug:
        import molass_legacy.KekLib.DebugPlot as dplt

        print("centx, centy, radius=", centx, centy, radius)
        print("xmin, xmax=", xmin, xmax)
        print("ymin, ymax=", ymin, ymax)

        with dplt.Dp():
            fig_, ax_ = dplt.subplots()
            ax.imshow(distimg)
            dplt.show()

    ret_values = []
    for vmin, vmax, size, v in [
        (xmin, xmax, jsize, centx),
        (ymin, ymax, isize, isize - centy),
        ]:
        w = v/size
        ret_value = vmin*(1 - w) + vmax*w
        ret_values.append(ret_value)

    src_scale = np.sqrt(isize**2 + jsize**2)
    tgt_scale = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
    ret_radius = radius/src_scale*tgt_scale

    return ret_values + [ret_radius]
