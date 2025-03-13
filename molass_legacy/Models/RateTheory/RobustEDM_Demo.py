"""
    Models/RateTheory/RobustEDM_Demo.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
def demo(in_folder, sd):
    from importlib import reload
    import Models.RateTheory.RobustEDM
    reload(Models.RateTheory.RobustEDM)
    from molass_legacy.Models.RateTheory.RobustEDM import guess_multiple

    ecurve = sd.get_xray_curve()
    x = ecurve.x.copy()
    y = ecurve.y

    guess_multiple(x, y, debug=True)
