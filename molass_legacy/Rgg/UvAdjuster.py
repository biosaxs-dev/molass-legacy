# coding: utf-8
"""
    Rgg.UvAdjuster.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Peaks.ElutionModels import egh_pdf

TAU_LIMIT_RATIO = 0.5   # relative to sigma
VERY_SMALL_VALUE = 1e-10
XRAY_SCALE = 1

class UvAdjuster:
    def __init__(self, xr_x, xr_y, xr_params):
        self.xr_x = xr_x
        self.xr_y = xr_y
        self.xr_params = xr_params

    def guess_initial_mapping_info(self, uv_x, uv_y, uv_spline):
        xr_x = self.xr_x
        ty = np.zeros(len(xr_x))
        cy_list = []
        for w, m, s, t in self.xr_params:
            cy = w * egh_pdf(xr_x, m, s, t)
            cy_list.append(cy)
            ty += cy

        max_ty = np.max(ty)
        xr_scale = np.max(self.xr_y)/max_ty
        max_uv_y = np.max(uv_y)
        uv_scale = max_uv_y/max_ty
        print("max_uv_y=", max_uv_y)
        print("uv_scale=", uv_scale)
        print("xr_scale=", xr_scale)

        if False:
            plt.push()
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,6))
            ax1.plot(uv_x, uv_y)
            ax1.plot(uv_x, uv_spline(uv_x))
            ax2.plot(xr_x, self.xr_y)
            for cy in cy_list:
                ax2.plot(xr_x, xr_scale*cy, ':')
            fig.tight_layout()
            debug_plot = plt.show()
            plt.pop()

        debug_plot = False

        def obj_func(p, return_scale=False):
            nonlocal debug_plot
            slope, intercept = p
            uv_ty = np.zeros(len(xr_x))
            uv_x = slope * xr_x + intercept
            y_ = uv_spline(uv_x)
            k = 0
            uv_cy_list = []
            for w, m, s, t in self.xr_params:
                uv_m = slope * m + intercept
                uv_s = slope * s
                uv_t = slope * t
                uv_cy = uv_scale * w * egh_pdf(uv_x, uv_m, uv_s, uv_t)
                uv_cy_list.append(uv_cy)
                uv_ty += uv_cy

            scale = max_uv_y/np.max(uv_ty)
            if return_scale:
                return scale

            fv = np.sum((scale*uv_ty - y_)**2)
            if debug_plot:
                print("p=", p)
                print("fv=", fv)
                # scale = xr_scale/uv_scale
                plt.push()
                fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,6))
                ax1.plot(uv_x, y_)
                ax2.plot(xr_x, self.xr_y)
                for cy in uv_cy_list:
                    ax1.plot(uv_x, scale*cy, ':')
                    ax2.plot(xr_x, xr_scale*cy, ':')
                ax1.plot(uv_x, scale*cy, ':')
                ax2.plot(xr_x, xr_scale*cy, ':')
                fig.tight_layout()
                debug_plot = plt.show()
                plt.pop()

            return fv
        init_params = uv_x[-1]/xr_x[-1], uv_x[0]
        bounds = [ (v*0.8, v*1.2) for v in init_params]
        res = minimize(obj_func, init_params, bounds=bounds)

        ret_scale = obj_func(res.x, return_scale=True)
        return ret_scale, res.x

    def obj_func(self, p):
        if self.fixed_mapping:
            w_ = p
            slope = self.slope
            intercept = self.intercept
        else:
            w_ = p[0:-2]
            slope, intercept = p[-2:]
        uv_ty = np.zeros(len(self.xr_x))
        uv_x = slope * self.xr_x + intercept
        k = 0
        uv_cy_list = []
        for _, m, s, t in self.xr_params:
            w = w_[k]
            uv_m = slope * m + intercept
            uv_s = slope * s
            uv_t = slope * t
            uv_cy = w * egh_pdf(uv_x, uv_m, uv_s, uv_t)
            uv_cy_list.append(uv_cy)
            uv_ty += uv_cy
            k += 1
        uv_y = self.uv_spline(uv_x)
        scale = np.max(uv_y)/np.max(uv_ty)
        fv = np.sum((scale*uv_ty - uv_y)**2)

        if self.debug_plot:
            print("fv=", fv)
            plt.push()
            fig, ax = plt.subplots()
            ax.plot(uv_x, uv_y, color='blue')
            for k, uv_cy in enumerate(uv_cy_list):
                ax.plot(uv_x, scale*uv_cy, ':')
            ax.plot(uv_x, scale*uv_ty, ':', color='red')
            fig.tight_layout()
            self.debug_plot = plt.show()
            plt.pop()

        return fv

    def fit(self, uv_x, uv_y, uv_spline, mapping_info=None):
        self.uv_x = uv_x
        self.uv_spline = uv_spline
        if mapping_info is None:
            scale, (slope, intercept) = self.guess_initial_mapping_params(uv_x, uv_y, uv_spline)
        else:
            scale, (slope, intercept) = mapping_info

        xr_params = self.xr_params.copy()
        self.fixed_mapping = False
        if self.fixed_mapping:
            self.slope = slope
            self.intercept = intercept
            init_params = xr_params[:,0]
        else:
            init_params = np.concatenate([xr_params[:,0], [slope], [intercept]])

        self.debug_plot = False

        if True:
            # method='Nelder-Mead'
            method = None
            self.result = res = minimize(self.obj_func, init_params, method=method)
        else:
            seed = np.random.randint(1000, 9999)
            self.result = res = basinhopping(self.obj_func, init_params, seed=seed)

        if self.fixed_mapping:
            self.mapping_params = np.array([slope, intercept])
            xr_params[:,0] = res.x
        else:
            self.mapping_params = res.x[-2:]
            xr_params[:,0] = res.x[0:xr_params.shape[0]]
        self.decomp_params = xr_params

    def get_fit_score(self):
        return np.log(self.obj_func(self.result.x)/len(self.xr_x))

    def plot_result(self, ax):
        slope, intercept = self.mapping_params
        uv_x = self.uv_x
        uv_ty = np.zeros(len(uv_x))
        k = 0
        uv_cy_list = []
        for w, m, s, t in self.decomp_params:
            uv_m = slope * m + intercept
            uv_s = slope * s
            uv_t = slope * t
            uv_cy = w * egh_pdf(uv_x, uv_m, uv_s, uv_t)
            uv_cy_list.append(uv_cy)
            uv_ty += uv_cy
            k += 1
        uv_y = self.uv_spline(uv_x)
        scale = np.max(uv_y)/np.max(uv_ty)

        for k, uv_cy in enumerate(uv_cy_list):
            ax.plot(uv_x, scale*uv_cy, ':', label='component-%d' % k)
        ax.plot(uv_x, scale*uv_ty, ':', color='red', label='total')

    def plot_mapped_uv(self, ax):
        slope, intercept = self.mapping_params
        x = self.xr_x
        uv_ty = np.zeros(len(x))
        k = 0
        uv_cy_list = []
        for w, m, s, t in self.decomp_params:
            uv_cy = w * egh_pdf(x, m, s, t)
            uv_cy_list.append(uv_cy)
            uv_ty += uv_cy
            k += 1

        uv_y = self.uv_spline(slope*x + intercept)
        y = np.zeros(len(x))
        for uv_cy, w_uv, w_xr in zip(uv_cy_list, self.decomp_params[:,0], self.xr_params[:,0]):
            y += w_xr/w_uv * uv_cy/uv_ty * uv_y

        scale = np.max(self.xr_y)/np.max(y)

        # for k, uv_cy in enumerate(uv_cy_list):
        #     ax.plot(x, scale*uv_cy, ':', label='component-%d' % k)
        ax.plot(x, scale*y, ':', color='blue', label='mapped uv')

def spike_demo(in_folder, sd=None, xr_params=None, seeds=None, mm_no=None):
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Elution.CurveUtils import simple_plot

    if sd is None:
        sp = StandardProcedure()
        sp.load(in_folder, debug=False)
        sd = sp.get_corrected_sd()

    D, E, qv, xr_curve = sd.get_xr_data_separate_ly()
    U, wlvec, uv_curve = sd.conc_array, sd.lvector, sd.get_uv_curve()

    if xr_params is None:
        """
        # 6999, 7028
        xr_params = np.array([
            [  2.57017904, 151.13250995,  27.53165132,   7.36131756,  99.42063111],
            [ 14.46655509, 222.19960345,  22.67418396, -11.30840347,  41.30237997],
            [  2.27135188, 261.20370834,  14.33768941,   7.15074626,  38.81913523],
            [  1.13742087, 320.3231673,   31.64481979,  -1.19046219,  42.69479584],
            ])

        # 9475, 7228
        xr_params = np.array([
            [  2.14588686, 145.16722691,  17.95032162,  -5.48518373,  94.23021699],
            [  3.56763662, 183.06996861,  13.01316199,  -6.46241911,  57.03501218],
            [ 16.49286555, 220.10341326,  22.00469214,  10.9826917,   40.68061606],
            [  1.68483415, 312.14292219,  41.90521428, -10.98016292,  41.69664135],
            ])
        # 9749, 9287
        xr_params = np.array([
            [  2.55842938, 143.78127637,  22.34910787,  10.24668622,  99.2569971 ],
            [ 10.56334344, 204.95578495,  20.13302181,  -9.14981967,  47.5842871 ],
            [ 10.55376469, 232.60451753,  18.68320577,   9.33390851,  38.54086037],
            [  2.11340872, 303.27840531,  43.90177422,   3.14959293,  42.00333368],
            ])
        """
        mm_seed, rr_seed = 7063, 1769
        xr_params = np.array([
            [  2.72274354, 145.29605286,  17.47330807,  -7.60850013,  94.15078257],
            [  5.05797034, 183.27475203,  13.37163632,  -6.61815137,  57.59996813],
            [ 21.1954766,  220.11493197,  21.50917803,  10.7468006,   40.70055306],
            [  2.83886823, 297.77671412,  51.77808462,  24.13764017,  41.89812747],
            ])
    else:
        mm_seed, rr_seed = seeds

    def debug_plot(uv_curve, xr_curve, ua=None):
        dp = plt.push()
        if ua is None:
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(14,6))
        else:
            from DataUtils import get_in_folder
            fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(21,7))
            in_folder_ = get_in_folder(in_folder)
            fig.suptitle("Simultaneuos UV-scaling and Mapping for %s with seeds=(%d, %d)" % (in_folder_, mm_seed, rr_seed), fontsize=24)
            ax1.set_title("UV Scaled Decomposition", fontsize=20)
            ax2.set_title("Xray Decomposition", fontsize=20)
            ax3.set_title("Mapped UV Elution overlaid on Xray Elution", fontsize=20)

        simple_plot(ax1, uv_curve,  color='blue', legend=False)
        if ua is not None:
            ua.plot_result(ax1)
        ax1.legend(fontsize=16)

        simple_plot(ax2, xr_curve, color='orange', legend=False)

        x = xr_curve.x
        y = xr_curve.y

        ty = np.zeros(len(x))
        cy_list = []
        for w, m, s, t in xr_params[:,0:4]:
            cy = w * egh_pdf(x, m, s, t)
            cy_list.append(cy)
            ty += cy
        scale = xr_curve.max_y/np.max(ty)

        for k, cy in enumerate(cy_list):
            ax2.plot(x, scale*cy, ':', label='component-%d' % k)

        ax2.plot(x, scale*ty, ':', color='red', label='total')
        ax2.legend(fontsize=16)

        if ua is not None:
            simple_plot(ax3, xr_curve,  color='orange', legend=False)
            ua.plot_mapped_uv(ax3)
            ax3.legend(fontsize=16)

        fig.tight_layout()
        if ua is not None:
            fig.subplots_adjust(top=0.85)

        if mm_no is None:
            plt.show()
        else:
            path = get_figure_path(mm_no)
            dp.fig.canvas.draw()
            dp.fig.savefig(path)

        plt.pop()

    if False:
        debug_plot(uv_curve, xr_curve)

    x = xr_curve.x
    y = xr_curve.y
    uv_x = uv_curve.x
    uv_y = uv_curve.y
    uv_spline = uv_curve.spline
    ua = UvAdjuster(x, y, xr_params[:,0:4])
    mapping_info = ua.guess_initial_mapping_info(uv_x, uv_y, uv_spline)
    print("mapping_info=", mapping_info)
    ua.fit(uv_x, uv_y, uv_spline, mapping_info)
    fit_score = ua.get_fit_score()
    print("fit_score=", fit_score)

    if True:
        debug_plot(uv_curve, xr_curve, ua=ua)

    return fit_score

def get_figure_path(mm_no):
    import os
    from molass_legacy._MOLASS.SerialSettings import get_setting
    sub_no = 0
    while True:
        sub_no += 1
        path  = os.path.join(get_setting("temp_folder"), "fig-%03d-%d.jpg" % (mm_no, sub_no))
        if os.path.exists(path):
            continue
        else:
            break
    return path

def compute_all(in_folder, log_file=None):
    import logging
    import re
    from molass_legacy.Batch.StandardProcedure import StandardProcedure
    from molass_legacy.Elution.CurveUtils import simple_plot
    from NumpyArrayUtils import from_space_separated_list_string

    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    sd = sp.get_corrected_sd()

    logger = logging.getLogger()
    parsing_params = False
    parsing_start = False
    params_key = "params="
    mixture_re = re.compile(r"Mixture\((\d+)\)")
    refiner_seeds_re = re.compile(r"Refiner\((\d+), (\d+)\)")
    trucate_re = re.compile(r"^(.+\]\]), func_value=(.+)")

    seeds_list = []
    params_list = []
    fv_list = []
    score_list = []
    classify_key_list = []
    mm_no = -1
    if log_file is None:
        # log_file = r"C:\Users\takahashi\Dropbox\_MOLASS\meeting-docs\20210426\rgg-figs\20190529_4\test.log"
        # log_file = r"D:\PyTools\pytools-2_2_4_develop\test\temp\test.log"
        log_file = r"D:\TODO\20210426\rgg-figs\test.log"
    fh = open(log_file)
    params_count = 0
    for k, line in enumerate(fh.readlines()):
        if params_count >= 9:
            # break
            pass

        if line.find("Mixture") > 0:
            m = mixture_re.search(line)
            if m:
                mm_no += 1

        if line.find("Refiner") > 0:
            parsing_params = True
            parsing_start = True
            m = refiner_seeds_re.search(line)
            if m:
                seeds = [int(m.group(n)) for n in [1,2]]
            else:
                assert False
        if parsing_params:
            # print([k], line)
            if parsing_start:
                parsing_start = False
                pos = line.find(params_key) + len(params_key)
                params_str = line[pos:]
                params_count += 1
            else:
                if line.find("]]") > 0:
                    m = trucate_re.match(line)
                    if m:
                        params_str += m.group(1)
                        fv = float(m.group(2))
                    else:
                        assert False

                    if np.isfinite(fv):
                        fv_list.append(fv)
                        parsing_params = False
                        params = from_space_separated_list_string(params_str)
                        logger.info("seeds=%s, params={%s}", str(seeds), str(params))
                        fit_score = spike_demo(in_folder, sd=sd, xr_params=params, seeds=seeds, mm_no=mm_no)
                        logger.info("fit_score=%g", fit_score)
                        seeds_list.append(seeds)
                        params_list.append(params)
                        score_list.append(fit_score)
                        # classify_key_list.append(np.concatenate([params[:,1], [fv]]))
                        classify_key_list.append(np.concatenate([[np.std(params[:,1])/10], [fv + fit_score]]))
                else:
                    params_str += line
    fh.close()

    classify_key_array = np.array(classify_key_list)

    print("classify_key_array.shape=", classify_key_array.shape)
    np.savetxt("classify_key_array.txt", classify_key_array)

    # X = classify_key_array[:,0:2]
    X = classify_key_array
    if True:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=5, random_state=0).fit(X)
        labels = kmeans.labels_
    else:
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=5).fit(X)
        labels = gmm.predict(X)
    if True:
        plt.push()
        fig, ax = plt.subplots()
        ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
        plt.show()
        plt.pop()

    fh = open("classification.csv", "w")
    for seeds, label, fv, score in zip(seeds_list, labels, fv_list, score_list):
        print((seeds, label, fv, score))
        fh.write(','.join([str(v) for v in [*seeds, label, fv, score]])+'\n')

    fh.close()
