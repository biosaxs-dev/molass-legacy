# coding: utf-8
"""
    ReciprocalData.py

    Copyright (c) 2019, SAXS Team, KEK-PF
"""
import os
from bisect import bisect_right
import numpy as np
from scipy import ndimage, interpolate
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from molass_legacy.KekLib.NumpyUtils import np_loadtxt_robust

class ReciprocalData:
    def __init__(self, shape):
        dmax = 100.0
        voxel = 5
        oversampling = 3

        D = dmax
        dn = shape[0]

        ############### from denss begin ###################

        #Initialize variables

        side = oversampling*D
        halfside = side/2

        if dn is None:
            dn = int(side/voxel)
            #want dn to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
            if dn%2==1:
                dn += 1

        #store dn for later use if needed
        nbox = dn

        dx = side/dn
        dV = dx**3
        V = side**3
        x_ = np.linspace(-halfside,halfside,dn)
        x,y,z = np.meshgrid(x_,x_,x_,indexing='ij')
        r = np.sqrt(x**2 + y**2 + z**2)

        df = 1/side
        qx_ = np.fft.fftfreq(x_.size)*dn*df*2*np.pi

        qx, qy, qz = np.meshgrid(qx_,qx_,qx_,indexing='ij')
        qr = np.sqrt(qx**2+qy**2+qz**2)
        qmax = np.max(qr)
        qstep = np.min(qr[qr>0])
        nbins = int(qmax/qstep)
        qbins = np.linspace(0,nbins*qstep,nbins+1)

        #create modified qbins and put qbins in center of bin rather than at left edge of bin.
        self.qbinsc = np.copy(qbins)
        self.qbinsc[1:] += qstep/2.

        #create an array labeling each voxel according to which qbin it belongs
        self.qbin_labels = np.searchsorted(qbins,qr,"right")
        self.qbin_labels -= 1

    def get_reciprocal(self, data):
        return np.fft.fftn(data)

    def get_scattering_curve(self, q, F, absolute=False):
        qbinsc = self.qbinsc
        qbin_labels = self.qbin_labels

        #APPLY RECIPROCAL SPACE RESTRAINTS
        #calculate spherical average of intensities from 3D Fs
        if absolute:
            I3D = F**2
        else:
            I3D = np.abs(F)**2
        print('I3D.shape=', I3D.shape, 'qbin_labels.shape=', qbin_labels.shape)
        index = np.arange(0,qbin_labels.max()+1)
        # print('index=', index)
        Imean = ndimage.mean(I3D, labels=qbin_labels, index=index)
        self.Imean = Imean

        #scale Fs to match data
        self.interp = interpolate.interp1d(qbinsc, Imean, kind='cubic', fill_value="extrapolate")
        I4chi = self.interp(q)

        ############### from denss end ###################

        curve_y = I4chi

        if False:
            import molass_legacy.KekLib.DebugPlot as dplt
            dplt.push()
            fig = dplt.figure()
            ax = fig.gca()
            ax.plot(curve_y)
            dplt.show()
            dplt.pop()

        return curve_y

    def get_3d_magnitudes(self, qbin_values, convolve=False):
        sqv = np.sqrt(qbin_values)
        ret = sqv[self.qbin_labels]
        if convolve:
            dn = 256
            nm = dn - 8
            weights = np.array([
                        [[1/dn, 1/dn, 1/dn], [1/dn,  1/dn, 1/dn], [1/dn, 1/dn, 1/dn]],
                        [[1/dn, 1/dn, 1/dn], [1/dn, nm/dn, 1/dn], [1/dn, 1/dn, 1/dn]],
                        [[1/dn, 1/dn, 1/dn], [1/dn,  1/dn, 1/dn], [1/dn, 1/dn, 1/dn]],
                        ])
            ret = ndimage.convolve(ret, weights, mode='constant', cval=1.0)
        return ret

    def get_3d_magnitudes_from_curve(self, q, y):
        interp = interpolate.interp1d(q, y, kind='cubic', fill_value="extrapolate")
        qbin_values = interp(self.qbinsc)
        return self.get_3d_magnitudes(qbin_values)

    def get_3d_magnitudes_from_file(self, path):
        array, _ = np_loadtxt_robust(path)
        q = array[:,0]
        y = array[:,1]
        return self.get_3d_magnitudes_from_curve(q, y)
