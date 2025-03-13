"""
    Models.Stochastic.RgReliability.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
import numpy as np

def determine_unreliables(rgs, qualities, props, trust_max_num=None, debug=True):
    logger = logging.getLogger(__name__)
    logger.info("determin_unreliables: rgs=%s, props=%s, qualities=%s, trust_max_num=%s", rgs, props, qualities, trust_max_num)

    rg_diff = np.diff(rgs)
    desc_indeces = np.where(rg_diff > 0)[0]
    indeces_from_desc_test = []
    for k in desc_indeces:
        # pick one with a lower quality rg as unreliable if they are not in decending order
        indeces_from_desc_test.append(k if qualities[k] < qualities[k+1] else k+1)

    indeces_from_quality_test = np.where(np.logical_or(props < 0.05,
                                                       qualities < 0.3
                                                       ))[0]

    ret_indeces = np.asarray(list(set(indeces_from_desc_test) | set(indeces_from_quality_test)), dtype=int)   # dtype=int is required when it is empty
    if trust_max_num is not None:
        num_peaks = len(rgs)
        if num_peaks - len(ret_indeces) > trust_max_num:
            full_indeces = np.arange(num_peaks)
            trusted_indeces = np.setdiff1d(full_indeces, ret_indeces)
            trustes_qualities = qualities[trusted_indeces]
            num_remove = len(trusted_indeces) - trust_max_num
            kk = np.argpartition(trustes_qualities, num_remove)
            if debug:
                print("qualities=", qualities, "ret_indeces=", ret_indeces)
                print("trusted_indeces=", trusted_indeces)
                print("trustes_qualities=", trustes_qualities, "kk=", kk, "remove=", kk[:num_remove])
            ret_indeces = np.asarray(sorted(np.concatenate([ret_indeces, trusted_indeces[kk[:num_remove]]])))
    if debug:
        print("indeces_from_desc_test=", indeces_from_desc_test)
        print("indeces_from_quality_test=", indeces_from_quality_test)
        print("ret_indeces=", ret_indeces)
    
    return ret_indeces