"""
    TentativeDemo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""

def demo(in_folder, sd):
    import molass_legacy.KekLib.DebugPlot as plt
    from importlib import reload
    import Models.ElutionCurveModels
    reload(Models.ElutionCurveModels)
    from .ElutionCurveModels import EGHA, EMGA

    print(in_folder)
    ecurve = sd.get_xray_curve()
    x = ecurve.x
    y = ecurve.y

    if len(ecurve.peak_info) > 2:
        pno = 1
    else:
        pno = 0

    f, m, t = ecurve.peak_info[pno]
    x_ = x[f:t]
    y_ = y[f:t]

    model = EMGA()
    params = model.guess(y_, x=x_)
    out = model.fit(y_, params, x=x_)
    fy = model.eval(out.params, x=x)

    print(model.get_name())
    h, mu, sigma, tau, a = out.params
    print(sigma, tau)
    ratio = model.tau_bound_ratio
    print(ratio, sigma*ratio, tau)

    with plt.Dp():
        fig, ax = plt.subplots()
        ax.set_title(in_folder)
        ax.plot(x, y)
        ax.plot(x_, y_)
        ax.plot(x, fy, ":")
        fig.tight_layout()
        plt.show()
