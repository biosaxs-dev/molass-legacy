"""
    PhaseRetrieval.py

    Copyright (c) 2020-2023, SAXS Team, KEK-PF
"""
import copy
import numpy as np
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_image(ax, image, cmap=cm.plasma, min_value=None):
    if min_value is None:
        image_ = image
    else:
        image_ = copy.deepcopy(image)
        image_[ncp.logical_not(ncp.isfinite(image_))] = 0
        image_[image_ < min_value] = 0

    ax.voxels(image_, cmap=cmap)

def plot_3d_scatter(ax, image, cmap=cm.plasma, min_value=0, shape_limits=False):
        w = np.where(image > min_value)
        wi = np.array(w, dtype=int).T
        n = image.shape[0]
        xyz = wi - n/2
        print(xyz.shape)
        v = image[w]
        sc = ax.scatter3D(xyz[:,0], xyz[:,1], xyz[:,2], c=v, cmap=cmap, alpha=1, s=30)
        ax.get_figure().colorbar(sc, ax=ax)
        if shape_limits:
            h = n/2
            ax.set_xlim(-h, h)
            ax.set_ylim(-h, h)
            ax.set_zlim(-h, h)

"""
    originally got from
        https://github.com/tuelwer/phase-retrieval
        Phase retrieval: Fienup's algorithms

    ----------------------------------------------------------------------
    Implementation of Fienup's phase-retrieval methods. This function
    implements the input-output, the output-output and the hybrid method.
    
    Note: Mode 'output-output' and beta=1 results in 
    the Gerchberg-Saxton algorithm.
    
    Parameters:
        mag: Measured magnitudes of Fourier transform
        mask: Binary array indicating where the image should be
              if padding is known
        beta: Positive step size
        steps: Number of iterations
        mode: Which algorithm to use
              (can be 'input-output', 'output-output' or 'hybrid')
        verbose: If True, progress is shown
    
    Returns:
        x: Reconstructed image
    
    Author: Tobias Uelwer
    Date: 30.12.2018
    
    References:
    [1] E. Osherovich, Numerical methods for phase retrieval, 2012,
        https://arxiv.org/abs/1203.4756
    [2] J. R. Fienup, Phase retrieval algorithms: a comparison, 1982,
        https://www.osapublishing.org/ao/abstract.cfm?uri=ao-21-15-2758
    [3] https://github.com/cwg45/Image-Reconstruction
    ----------------------------------------------------------------------

    significantly mofified by SAXS Team, KEK-PF
    where the only skelton with mode = 'hybrid' is left.
"""
def fienup_phase_retrieval(mag, mask=None, beta=0.8, gpu=False, roll=True,
                            min_value=None, lim_value=0, saxs_curve=None,
                           steps=200, verbose=True):
    assert beta > 0, 'step size must be a positive number'
    assert steps > 0, 'steps must be a positive number'

    gpu_ok = False
    if gpu:
        try:
            import cupy as ncp
            gpu_ok = True
            if mask is not None:
                mask = ncp.asarray(mask)
        except:
            pass

    if gpu_ok:
        print('Using GPU')
        mag = ncp.asarray(mag)
    else:
        import numpy as ncp

    if mask is None:
        mask = ncp.ones(mag.shape)

    assert mag.shape == mask.shape, 'mask and mag must have same shape'

    # sample random phase and initialize image x 
    y_hat = mag*ncp.exp(1j*2*ncp.pi*ncp.random.rand(*mag.shape))
    # y_hat === rho in DENSS
    x = ncp.zeros(mag.shape)

    # previous iterate
    x_p = None
    x_hat_p = None

    # main loop
    for i in range(1, steps+1):
        # show progress
        if i % 100 == 0 and verbose: 
            print("step", i, "of", steps)

        # inverse fourier transform
        y = ncp.real(ncp.fft.ifftn(y_hat))

        # previous iterate
        if x_p is None:
            x_p = y
        else:
            x_p = x

        # updates for elements that satisfy object domain constraints
        x = y

        # find elements that violate object domain constraints 
        # or are not masked
        indices = ncp.logical_or(ncp.logical_and(y<lim_value, mask), 
                                ncp.logical_not(mask))

        # updates for elements that violate object domain constraints
        x[indices] = x_p[indices]-beta*y[indices]

        # fourier transform
        x_hat = ncp.fft.fftn(x)

        # satisfy fourier domain constraints
        # (replace magnitude with input magnitude)
        if saxs_curve is None:
            y_hat = mag*ncp.exp(1j*ncp.angle(x_hat))
        else:
            assert False

    if gpu_ok:
        x = ncp.asnumpy(x)

    if roll:
        try:
            from scipy import ndimage
            if min_value is None:
                x_ = x
            else:
                x_ = copy.deepcopy(x)
                x_[x_ < min_value] = 0
            com = ndimage.measurements.center_of_mass(x_)   # not supported in cupy
            m = np.array(com, dtype=int)
            print('com, m=', com, m)
            for k in range(x.ndim):
                m_ = x.shape[k]//2 - m[k]
                print([k], 'roll', m_)
                x = np.roll(x, m_, axis=k)
        except:
            print('roll failed.')

    return x
