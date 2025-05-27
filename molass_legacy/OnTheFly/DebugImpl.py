"""
    DebugImpl.py

    Copyright (c) 2023-2024, SAXS Team, KEK-PF
"""
import molass_legacy.KekLib.DebugPlot as plt
from molass_legacy.Elution.CurveUtils import simple_plot

def debug_impl_ConcDepend(self):
    print("debug_impl")
    rdr_hints, ecurve, judge_holder = self.debug_info
    major_peak_info = ecurve.get_major_peak_info()
    print(len(rdr_hints), len(ecurve.peak_info), len(major_peak_info))

    x = ecurve.x
    y = ecurve.y

    with plt.Dp():
        fig, ax = plt.subplots()
        # simple_plot(ax, ecurve)
        ax.plot(x, y)
        for rec in ecurve.get_major_peak_info():
            px = rec[1]
            ax.plot(px, ecurve.spline(px), "o", color="red")
        fig.tight_layout()
        plt.show()


def debug_impl_SummaryBook(self):
    from openpyxl import Workbook
    from importlib import reload
    import Reports.SummaryBook
    reload(Reports.SummaryBook)
    from molass_legacy.Reports.SummaryBook import SummaryBook

    book_file, controller = self.debug_info
    print("debug_impl: book_file=", book_file)

    wb = Workbook()
    summary = SummaryBook(wb, controller)
    wb.save(book_file)

def debug_impl(self):
    print("debug_impl")
    with plt.Dp():
        fig, axes = plt.subplots(ncols=5, figsize=(15, 3))
        for ax in axes:
            ax.plot(0, 0, "o")
        fig.tight_layout()
        plt.show()