"""

    OutlineFigureSaver.py

    Copyright (c) 2022, SAXS Team, KEK-PF

"""
import numpy as np
import imageio.v3 as iio
import molass_legacy.KekLib.DebugPlot as plt

def save_the_figure_trial_impl( self, folder, analysis_name ):
    # print( 'save_the_figure: ', folder, analysis_name )
    filename = analysis_name.replace( 'analysis', 'figure' )
    path = os.path.join( folder, filename )

    """
    Matplotlib figure to image as a numpy array
    https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
    """
    s, (width, height) = self.mpl_canvas.print_to_buffer()

    # Option 2a: Convert to a NumPy array.
    image = np.fromstring(s, np.uint8).reshape((height, width, 4))
    print('image.shape=', image.shape)

    """
    How to Resize, Pad Image to Square Shape and Keep Its Aspect Ratio with Python
    https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    """
    import cv2
    white = [255, 255, 255]
    top, bottom, left, right = 50, 10, 10, 10
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=white)

    if True:
        import molass_legacy.KekLib.DebugPlot as plt
        plt.show()
        with plt.Dp():
            fig, ax = plt.subplots()
            ax.imshow(new_im)
            plt.show()
    else:
        cv2.imshow("image", new_im)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    cv2.imwrite(path + '.png', new_im)

def save_the_upper_figure_impl(self):
    from tkinter import filedialog
    from DataUtils import get_in_folder
    """
    Matplotlib figure to image as a numpy array
    https://stackoverflow.com/questions/35355930/matplotlib-figure-to-image-as-a-numpy-array
    """
    buf = self.mpl_canvas.buffer_rgba()

    # convert to a NumPy array
    outline_image = np.asarray(buf)
    h, w = outline_image.shape[0:2]

    in_folder = get_in_folder()

    # using DebugPlot as a tool for internal drawing
    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title(in_folder, fontsize=16)
        ax.set_axis_off()
        ax.imshow(outline_image)
        fig.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        X = np.asarray(buf)
        H, W = X.shape[0:2]
        hc = H//2 
        wc = W//2
        hslice = slice(0, hc+30)
        sw = w//2 + 30
        wslice = slice(wc-sw, wc+sw)
        upper_image = X[hslice,wslice,:]
        upper_image[-20:,:,:] = 255
        # plt.show()

    files = [('PNG Files', '*.png'), ('JPEG Files', '*.jpg')]
    filename = in_folder.split("/")[-1] + ".png"
    path = filedialog.asksaveasfilename(parent=self.parent, initialfile =filename, filetypes=files, defaultextension=files)
    iio.imwrite(path, upper_image)

def save_the_elution_curve_impl(self, type):
    from tkinter import filedialog
    from DataUtils import get_in_folder

    sd = self.serial_data
    if type == 0:
        ecurve = sd.get_xray_curve()
    else:
        ecurve = sd.get_uv_curve()

    x = ecurve.x
    y = ecurve.y

    in_folder = get_in_folder()

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title(in_folder, fontsize=16)
        ax.plot(x, y)
        fig.tight_layout()
        plt.show()
