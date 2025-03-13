# coding: utf-8
"""
    Rgg.SpikeData.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import numpy as np
from molass_legacy.Peaks.ElutionModels import egh
from .RggUtils import normal_pdf

FULL_PARAMS =  np.array([
        [35, [1.0, 200, 20,  0]],
        [31, [0.2, 250, 20,  0]],
        [23, [0.8, 400, 30,  0]],
        ])

def generate_demo_data(num_components=3):
    x = np.arange(600)
    y = np.zeros(len(x))
    rg_list = []
    cy_list = []

    gen_params = FULL_PARAMS if num_components == 3 else FULL_PARAMS[[0,2]]
    for rg, params in gen_params:
        if False:
            cy = egh(x, *params)
        else:
            h, mu, sigma, tau = params
            cy = h*normal_pdf(x, mu, sigma)
        cy_list.append(cy)
        y += cy
        rg_list.append(rg)

    rg = np.sum(np.array(rg_list)[:,np.newaxis]*np.array(cy_list)/y, axis=0)
    return x, y, rg, len(cy_list)
