# coding: utf-8
"""

    PairedRangeLogger.py

    Copyright (c) 2019, SAXS Team, KEK-PF

"""

PAIRED_RANGES_LOG_HEADER  = "--- paired_ranges log begin ---"
PAIRED_RANGES_LOG_TRAILER = "--- paired_ranges log end ---"

def log_paired_ranges(logger, paired_ranges):
    # print('log_paied_ranges: ', paired_ranges)

    used_elution_model = None
    model_name = None

    logger.info(PAIRED_RANGES_LOG_HEADER)

    for k, range_ in enumerate(paired_ranges):
        range_str = range_.get_log_str()

        if used_elution_model is None:
            used_elution_model = range_.elm_recs is not None

        if used_elution_model:
            if model_name is None:
                model_name = range_.elm_recs[0][1].get_model_name()
            logger.info("%d-th %s has been decomposed using %s as follows." % (k, range_str, model_name))
            for rec in range_.elm_recs:
                # print('rec=', rec)
                kno = rec[0]        # rec.kno?
                evaluator = rec[1]  # rec.evaluator?
                logger.info("%d-th element: %s" % (kno, evaluator.get_all_params_string()) )
        else:
            logger.info("%d-th %s has no decomposition info" % (k, range_str))

    logger.info(PAIRED_RANGES_LOG_TRAILER)
