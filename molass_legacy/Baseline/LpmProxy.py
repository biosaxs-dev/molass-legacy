# coding: utf-8
"""
    Baseline.LpmProxy.py

    Copyright (c) 2020, SAXS Team, KEK-PF
"""
import numpy as np

class LpmProxy:
    def __init__(self, data, **kwargs):
        thru_mode = kwargs.pop('thru_mode', False)
        self.qvector = kwargs.pop('q')
        self.mapper = kwargs.pop('mapper')
        if thru_mode:
            from LPM import LPM_3d
            lpm = LPM_3d(data, **kwargs)
            self.data = lpm.data
        else:
            self.data = self.correct_old_style(data, **kwargs)

    def correct_old_style(self, data, **kwargs):
        from ScatteringBaseCorrector import ScatteringBaseCorrector
        jsize = data.shape[1]
        qvector = self.qvector
        jvector = np.arange(jsize)
        assert qvector is not None

        evector = np.zeros(len(qvector))
        intensity_array = np.array([np.vstack([qvector, data[:,j], evector]).T for j in range(jsize)])
        print('intensity_array.shape=', intensity_array.shape)

        # affine_info = [jvector, np.zeros(jsize), 0]
        affine_info = self.mapper.get_affine_info()

        inty_curve_y = kwargs.pop('ecurve_y')
        corrector = ScatteringBaseCorrector(jvector, qvector, intensity_array,
                        curve=self.mapper.sd_xray_curve
                        affine_info=affine_info, inty_curve_y=inty_curve_y,
                        )

        corrector.correct_all_q_planes()
        corrected = intensity_array[:,:,1].T
        print('corrected.shape=', corrected.shape)

        return corrected
