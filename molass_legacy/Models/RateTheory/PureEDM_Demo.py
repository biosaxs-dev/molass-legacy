"""
    Models/RateTheory/PureEDM_Demo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
def demo(in_folder, sd):
    from importlib import reload
    import Models.RateTheory.PureEDM
    reload(Models.RateTheory.PureEDM)
    from molass_legacy.Models.RateTheory.PureEDM import guess_multiple, edm_impl

    ecurve = sd.get_xray_curve()
    x = ecurve.x.copy()
    y = ecurve.y

    num_peaks = len(ecurve.peak_info) + 1

    if False:
        fh = open("moments-params.csv", "w")
    else:
        fh = None

    guess_multiple(x, y, num_peaks, fh=fh, debug=True)

    if fh is not None:
        fh.close()
