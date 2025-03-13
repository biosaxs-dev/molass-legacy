"""
    Models/RateTheory/EDM_Demo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""

def demo(in_folder, sd, num_peaks=None):
    from importlib import reload
    import Models.RateTheory.EDM
    reload(Models.RateTheory.EDM)
    from molass_legacy.Models.RateTheory.EDM import guess_multiple, edm_impl

    uv_ecurve = sd.get_xray_curve()
    xr_ecurve = sd.get_uv_curve()

    for ecurve in uv_ecurve, xr_ecurve:

        x = ecurve.x
        y = ecurve.y

        if num_peaks is None:
            num_peaks_ = len(ecurve.peak_info) + 1
        else:
            num_peaks_ = num_peaks

        guess_multiple(x, y, num_peaks_, debug=True)
