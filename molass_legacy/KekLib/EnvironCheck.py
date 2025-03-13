"""
    EnvironCheck.py

    Copyright (c) 2018-2024, Masatsuyo Takahashi, KEK-PF
"""

def executables_check(parent):
    try:
        import numba
        numba_is_available = True
    except:
        numba_is_available = False

    if numba_is_available:
        ret = True
    else:
        from molass_legacy._MOLASS.SerialSettings import get_setting
        suppress_numba_warning = get_setting('suppress_numba_warning')
        if suppress_numba_warning:
            ret = True
        else:
            import OurMessageBox as MessageBox
            yn = MessageBox.askyesno("No numba warning",
                '"numba" does not seem to be available.\n'
                "It would be desirable if you could take some time to update.\n"
                "Or, extrapolation with smoothing will be significantly slow.\n"
                "Would you like to proceed anyway?",
                parent=parent)
            ret = yn
    return ret
