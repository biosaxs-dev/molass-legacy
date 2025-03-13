"""

    LambertBeerTester.py

    Copyright (c) 2024, SAXS Team, KEK-PF

"""
from importlib import reload
import molass_legacy.KekLib.DebugPlot as plt

def test_compute_base_plane(data, index, ecurve, denoise=False):
    BP = None
    def test_compute_base_plane_impl():
        import Baseline.LambertBeer
        reload(Baseline.LambertBeer)
        from molass_legacy.Baseline.LambertBeer import BasePlane
        nonlocal BP
        bp = BasePlane(data, index, ecurve, denoise=denoise, debug=True)
        bp.solve(debug=True)
        BP = bp.get_baseplane()
        return BP

    with plt.Dp(extra_button_specs=[("Test", test_compute_base_plane_impl)]):
        fig, ax = plt.subplots()
        ax.set_title("test_compute_base_plane")
        plt.show()

    return BP