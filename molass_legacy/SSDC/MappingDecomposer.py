"""
    SSDC.MappingDecomposer.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import linregress
from scipy.interpolate import UnivariateSpline
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Models.ModelUtils import compute_cy_list, compute_area_props
from molass_legacy.Models.Stochastic.MonoporeMoments import study_monopore_moments_impl
from SecTheory.BasicModels import robust_single_pore_pdf

def guess_init_mapping(sorted_recs_list, moments_list):
    x = []
    y = []
    for (i, p), (j, q) in zip(sorted_recs_list[0][0:2], sorted_recs_list[1][0:2]):
        x.append(moments_list[0][i][0])
        y.append(moments_list[1][j][0])

    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept

def guess_mapping(source, target, props_list, moments_list):
    sorted_recs_list = []
    for props in props_list:
        sorted_recs = sorted([(i, p) for i, p in enumerate(props)], key=lambda x: -x[1])
        sorted_recs_list.append(sorted_recs)
    model = source.model
    sx = source.xr_x
    sy = source.xr_y
    tx = target.xr_x
    ty = target.xr_y

    sorted_recs = sorted_recs_list[0]
    if len(sorted_recs) < 2:        
        """
        tx = a*sx + b
        b = tx - a*sx
        """
        a = 1
        i = sorted_recs[0][0]
        b = moments_list[1][i][0] - a*moments_list[0][i][0]     # computing b using the most significant peak
        print("b=", b)
        mapping = a, b
    else:
        mapping = guess_init_mapping(sorted_recs_list, moments_list)
        return mapping

    speaks = source.get_peaks()
    tpeaks = target.get_peaks()
    src_params_list = []
    ret_params_list = []
    ret_indeces = []
    sry_ = sy.copy()
    try_ = ty.copy()
    for k, (i, p) in enumerate(sorted_recs, start=1):
        sparams = speaks[i]
        print([i], "sparams=", sparams)
        print([i], "tparams",  tpeaks[i])
        scale = np.max(try_)/np.max(sry_)
        ret = get_a_mapped_component_params(mapping, scale, model, sparams, sx, sry_, tx, try_, k, i, ret_params_list)
        if ret is None:
            return
        src_params_list.append(sparams)
        ret_params_list.append(ret)
        ret_indeces.append(i)
        mapping = update_mapping(mapping, src_params_list, ret_params_list)
        if k >= 3:
            break
        cy = model(sx, sparams)
        sry_ -= cy
        cy = model(tx, ret)
        try_ -= cy

    return mapping

def update_mapping(mapping, src_params_list, ret_params_list):
    a, b = mapping
    points = []
    if len(src_params_list) < 2:
        points.append((1, a+b))
    for sparams, tparams in zip(src_params_list, ret_params_list):
        points.append((sparams[1], tparams[1]))
    points_array = np.array(points)
    slope, intercept, r_value, p_value, std_err = linregress(points_array[:,0], points_array[:, 1])
    return slope, intercept

def compute_monopore_curves(x, mnp_params, rgs):
    N, T, t0, me, mp, poresize = mnp_params[0:6]
    scales = mnp_params[6:]
    rhov = rgs/poresize
    rhov[rhov > 1] = 1
    cy_list = []
    for k, (rho, scale) in enumerate(zip(rhov, scales)):
        ni_ = N * (1 - rho)**me
        ti_ = T * (1 - rho)**mp
        cy = scale * robust_single_pore_pdf(x - t0, ni_, ti_)
        cy_list.append(cy)
    tty = np.sum(cy_list, axis=0)
    return cy_list, tty

def decompose_paired_elution_curves(info_list, moments_list=None, num_components=None):
    print("decompose_paired_elution_curves")
    if moments_list is None:
        moments_list = []
        for in_folder, lrf_src in info_list:
            moments_list.append(lrf_src.get_egh_moments_list())
    
    props_list = []
    for in_folder, lrf_src in info_list:
        peaks = lrf_src.get_peaks()
        cy_list = compute_cy_list(lrf_src.model, lrf_src.xr_x, peaks)
        props = compute_area_props(cy_list)
        props_list.append(props)

    source = info_list[0][1]
    target = info_list[1][1]    
    # props = props_list[0]

    mapping = guess_mapping(source, target, props_list, moments_list)

    # sorted_recs = sorted([(i, p) for i, p in enumerate(props)], key=lambda x: -x[1])
    # print("sorted_recs=", sorted_recs)

    model = source.model
    sx = source.xr_x
    sy = source.xr_y
    tx = target.xr_x
    ty = target.xr_y

    # 
    ret_params, ret_rgs, unreliable_indeces, params_scaler = study_monopore_moments_impl(source, debug=False)
    in_folder = info_list[0][0]
    if in_folder.find("20210727") >= 0:
        unreliable_indeces = np.array([3])      # temporary data dependent fix
    elif in_folder.find("20220716/BSA_") >= 0:
            unreliable_indeces = np.array([0])  # temporary data dependent fix
    elif in_folder.find("20230706") >= 0:
            unreliable_indeces = np.array([0])  # temporary data dependent fix

    reliable_indeces = np.setdiff1d(np.arange(len(ret_rgs)), unreliable_indeces)
    print("reliable_indeces=", reliable_indeces)
    ret_scales = ret_params[6:]
    mnp_params = np.concatenate([ret_params[0:6], ret_scales[reliable_indeces]])
    temp_rgs = ret_rgs[reliable_indeces]

    cy_list, tty = compute_monopore_curves(sx, mnp_params, temp_rgs)

    a, b = mapping
    tspline = UnivariateSpline(tx, ty, s=0, ext=3)
    tx_ = a*sx + b
    ty_ = tspline(tx_)
    init_scale = np.max(ty)/np.max(sy)
    with plt.Dp():
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
        fig.suptitle("Monopore Curve Mapping", fontsize=20)
        ax1.plot(sx, sy)
        ax2.plot(tx_, ty_)
        for k, cy in enumerate(cy_list):
            ax1.plot(sx, cy, ":", label='component-%d' % k)
            ax2.plot(tx_, init_scale*cy, ":", label='component-%d' % k)
        fig.tight_layout()
        ret = plt.show()
    if not ret:
        return

    n = len(mnp_params)
    num_peaks = len(temp_rgs)
    print("temp_rgs=", temp_rgs)
    print("num_peaks=", num_peaks)
    print("init mapping=", mapping)
    init_a, init_b = mapping

    def split_params(p):
        mnp_params_ = p[0:n]
        rgs_ = p[n:n+num_peaks] 
        a, b, scale = p[n+num_peaks:]
        return mnp_params_, rgs_, a, b, scale

    def mapping_objective(p):
        mnp_params_, rgs_, a, b, scale = split_params(p)
        scy_list, sty = compute_monopore_curves(sx, mnp_params_, rgs_)            
        tx_ = a*sx + b
        ty_ = tspline(tx_)
        mapping_penalty = max(0, abs(1 - a/init_a) - 0.05)**2 *1e6
        return np.log(np.sum((sty - sy)**2)) +  np.log(np.sum((scale*sty - ty_)**2)) + mapping_penalty

    init_sacle = np.max(ty)/np.max(sy)
    init_params = np.concatenate((mnp_params, temp_rgs, mapping, [init_sacle]))
    res = minimize(mapping_objective, init_params, method='Nelder-Mead')
    mnp_params_, rgs_, a, b, scale = split_params(res.x)
    print("optimized mapping=", (a, b))

    return mnp_params_, rgs_, (a, b), scale

def get_a_mapped_component_params(mapping, scale, model, sparams, sx, sy, tx, ty, k, i, ret_params_list, debug=False):
    a, b = mapping

    def plot_component_state(sparams, tparams, title=None):
        if title is None:
            title = "Component %d" % k

        with plt.Dp():
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))
            fig.suptitle(title)
            ax1.set_title("Source Decomposition State")
            ax1.plot(sx, sy, color="orange")
            cy = model(sx, sparams)
            ax1.plot(sx, cy, ":", color="C%d" % i)
            ax2.set_title("Target Decomposition State")
            ax2.plot(tx, ty, color="orange")
            cy = model(tx, tparams)
            ax2.plot(tx, cy, ":", color="C%d" % i)
            fig.tight_layout()
            ret = plt.show()
        return ret
    
    tparams = sparams.copy()
    tparams[0] = sparams[0]*scale
    tparams[1] = a*sparams[1] + b

    if debug:
        ret = plot_component_state(sparams, tparams, title="[%d-%d] befor optimize" % (k,i))
        if not ret:
            return

    m, s = sparams[1:3]
    target_range = []
    for px in m - 2*s, m + 2*s:
        target_range.append(a*px + b)
    
    start = max(0, int(target_range[0])) 
    stop = min(len(tx), int(target_range[1]))
    tx_ = tx[start:stop]
    ty_ = ty[start:stop]

    def objective(p):
        tparams[[0, 1]] = p
        return np.sum((ty_ - model(tx_, tparams))**2)

    res = minimize(objective, tparams[[0,1]])
    tparams[[0,1]] = res.x

    if debug:    
        ret = plot_component_state(sparams, tparams, title="[%d-%d] after optimize" % (k,i))
        if not ret:
            return

    return tparams
