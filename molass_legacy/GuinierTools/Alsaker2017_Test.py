"""
    Alsaker2017_Test.py

    Copyright (c) 2024, SAXS Team, KEK-PF

    How to test

    R version:
        on VS-code R-debugger in the TODO/Rg code
        setwd('D:\\TODO\\20240523\\Rg code')
        data = read.table("oval_01C_S008_0_01.dat", header = FALSE)
        source('file1.R')
        estimate_Rg(data, 1, 5)

    Python version:
        on Command Pronmpt in the test folder
        python .../GuinierTools/Alsaker2017_Test.py
"""
import numpy as np

def load_data(file):
    data = np.loadtxt(file)
    return data

def demo_impl():
    from molass_legacy.SerialAnalyzer.DataUtils import get_local_path
    from importlib import reload
    import GuinierTools.Alsaker2017 as module
    reload(module)
    from GuinierTools.Alsaker2017 import estimate_Rg
    TODO = get_local_path('TODO')
    data = load_data(TODO + r'\20240523\Rg code\oval_01C_S008_0_01.dat')
    print(data.shape)
    estimate_Rg(data, 1, 5)

def demo():
    import molass_legacy.KekLib.DebugPlot as dpl

    extra_button_specs = [("Demo", demo_impl)]
    with dpl.Dp(extra_button_specs=extra_button_specs):
        fig, ax = dpl.subplots()
        dpl.show()

if __name__ == '__main__':
    import sys
    import seaborn as sns
    sns.set_theme()
    sys.path.append("../lib")
    demo()