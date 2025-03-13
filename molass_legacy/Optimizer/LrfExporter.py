"""
    Optimizer.LrfExporter.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import os
import re
import numpy as np
from molass_legacy.KekLib.NumpyUtils import np_savetxt
from molass_legacy._MOLASS.SerialSettings import get_setting
from molass_legacy.KekLib.BasicUtils import Struct
from molass_legacy.SerialAnalyzer.AnalyzerUtil import compute_conc_factor_util

class LrfExporter:
    def __init__(self, optimizer, params, dsets):
        xrD_ = optimizer.xrD_
        uvD_ = optimizer.uvD_
        self.optimizer = optimizer
        self.separate_eoii = optimizer.separate_eoii
        self.xr_curve = optimizer.xr_curve
        self.uv_curve = optimizer.uv_curve
        self.xrD = optimizer.xrD
        self.xrD_ = optimizer.xrD_
        self.xrE = optimizer.xrE
        self.uvD = optimizer.uvD
        self.uvD_ = optimizer.uvD_
        self.rg_curve = optimizer.rg_curve
        self.dsets = dsets
        self.params = params
        self.separate_params = optimizer.split_params_simple(params)
        self.conc_factor = compute_conc_factor_util()
        lrf_info = optimizer.objective_func(params, return_lrf_info=True)

        Pxr, Cxr, Puv, Cuv, mapped_UvD = lrf_info.matrices
        mapping = self.separate_params[3]
        uv_points = lrf_info.estimate_uv_peak_points()

        self.uv_result = Struct(Cuv=Cuv, Puv=Puv, uv_curve=self.uv_curve, mapping=mapping, peaktops=uv_points, uv_x=lrf_info.uv_x, uv_y=lrf_info.uv_y)

        xr_points = lrf_info.estimate_xr_peak_points()
        valid_rgs = lrf_info.get_valid_rgs(self.separate_params[2])
        self.xr_result = XrLrfResult(self.xrD, self.xrD_, self.xrE, Pxr, Cxr, optimizer.qvector,
                                    self.xr_curve, self.conc_factor, valid_rgs, xr_points)
                                    # self.xrD ok?

    def export(self, folder=None):
        from molass_legacy.KekLib.OurTkinter import Tk
        from molass_legacy.KekLib.PathUtils import NonExistingPath

        if folder is None:
            from .TheUtils import get_optimizer_folder
            optimizer_folder = get_optimizer_folder()
            folder = os.path.join(optimizer_folder, "exported")

        folder = str(NonExistingPath(folder))
        os.makedirs(folder)

        self.xr_result.export(folder)
        dummy_parent = Tk.Tk()  # this dummy_parent should be removed
        dummy_parent.withdraw()
        sheet = self.optimizer.params_type.get_params_sheet(dummy_parent, self.params, self.dsets, self.optimizer)
        sheet.save_as(os.path.join(folder, "parameters.csv"))
        dummy_parent.destroy()

        gr_result = self.get_graphic_result()
        gr_folder = os.path.join(folder, "graphic")
        os.makedirs(gr_folder)
        gr_result.export(gr_folder)

        return folder

    def get_graphic_result(self):
        return GraphicResult(self.uv_result, self.xr_result, self.rg_curve)

class XrLrfResult:
    def __init__(self, xrD, xrD_, xrE, P, C, qv, xr_curve, conc_factor, rg_params, peaktops):
        self.info = (xrD, xrD_, xrE, P, C, qv)
        self.xr_curve = xr_curve
        self.conc_factor = conc_factor
        self.rg_params = rg_params
        self.peaktops = peaktops

    def export(self, folder):
        
        """
        P  = M @ X, X = Minv @ P
        Pe <== E @ X
        Pe = sqrt((E**2) @ (X**2))
        """

        M, M_, E, P, C, qv = self.info

        X = np.linalg.pinv(M) @ P           # M or M_
        Ep = np.sqrt((E**2) @ (X**2))
        num_components = C.shape[0] - 1
        for j, p in enumerate(P.T):
            filename = 'component-%d.dat' % (j+1) if j < num_components else 'baseline-component.dat'
            file = os.path.join(folder, filename)
            np_savetxt(file, np.array([qv, p, Ep[:,j]]).T, fmt="%.5e")

        eno = np.arange(C.shape[1])
        for i, c in enumerate(C):
            filename = 'concentration-%d.dat' % (i+1) if i < num_components else 'baseline-concentration.dat'
            file = os.path.join(folder, filename)
            np_savetxt(file, np.array([eno, c]).T, fmt="%.5e")

NP_SAVETXT_FMT = "%g"

class GraphicResult:
    def __init__(self, uv_result, xr_result, rg_curve):
        self.uv_result = uv_result
        self.xr_result = xr_result
        self.rg_curve = rg_curve

    def export(self, folder):
        from GuinierTools.RgCurveUtils import compute_rg_curves

        xr_curve = self.xr_result.xr_curve
        xr_x = xr_curve.x
        xr_y = xr_curve.y
        C = self.xr_result.info[4]  # C
        CY = C/self.xr_result.conc_factor

        # xr
        file = os.path.join(folder, "xr-elution-data.txt")
        np.savetxt(file, np.array([xr_x, xr_y]).T, fmt=NP_SAVETXT_FMT)
        ty = np.zeros(len(xr_x))
        xr_cy_list = []
        for k, cy in enumerate(CY, start=1):
            if k < len(CY):
                name = "xr-elution-c-%d.txt" % k
            else:
                name = "xr-elution-baseline.txt"
            file = os.path.join(folder, name)
            np.savetxt(file, np.array([xr_x, cy]).T, fmt=NP_SAVETXT_FMT)
            ty += cy
            xr_cy_list.append(cy)

        file = os.path.join(folder, "xr-elution-c-total.txt")
        np.savetxt(file, np.array([xr_x, ty]).T, fmt=NP_SAVETXT_FMT)
        peaktops = self.xr_result.peaktops
        file = os.path.join(folder, "xr-peaktop-dots.txt")
        np.savetxt(file, peaktops, fmt=NP_SAVETXT_FMT)

        # rg
        ty_ = ty - xr_cy_list[-1]       # remove baseline part
        weights = self.xr_result.peaktops[:,1]
        rg_params = self.xr_result.rg_params
        rg_curves1, rg_curves2 = compute_rg_curves(xr_x, weights/np.max(weights), rg_params, xr_cy_list[:-1], ty_, self.rg_curve)
        for k, ((x1, rg1), (x2, rg2)) in enumerate(zip(rg_curves1, rg_curves2), start=1):
            file = os.path.join(folder, "rg-observed-%d.txt" % k)
            np.savetxt(file, np.array([x1, rg1]).T, fmt=NP_SAVETXT_FMT)
            file = os.path.join(folder, "rg-reconstructed-%d.txt" % k)
            np.savetxt(file, np.array([x2, rg2]).T, fmt=NP_SAVETXT_FMT)

        # uv
        uv_x = self.uv_result.uv_x
        uv_y = self.uv_result.uv_y
        file = os.path.join(folder, "uv-elution-data.txt")
        np.savetxt(file, np.array([uv_x, uv_y]).T, fmt=NP_SAVETXT_FMT)

        Cuv = self.uv_result.Cuv
        ty = np.zeros(len(uv_x))
        for k, cy in enumerate(Cuv, start=1):
            if k < len(Cuv):
                name = "uv-elution-c-%d.txt" % k
            else:
                name = "uv-elution-baseline.txt"
            file = os.path.join(folder, name)
            np.savetxt(file, np.array([uv_x, cy]).T, fmt=NP_SAVETXT_FMT)
            ty += cy

        file = os.path.join(folder, "uv-elution-c-total.txt")
        np.savetxt(file, np.array([uv_x, ty]).T, fmt=NP_SAVETXT_FMT)
        peaktops = self.uv_result.peaktops
        file = os.path.join(folder, "uv-peaktop-dots.txt")
        np.savetxt(file, peaktops, fmt=NP_SAVETXT_FMT)
