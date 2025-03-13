"""
    Tools/MapperSingleton.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""
from molass_legacy.Trimming.PreliminaryRecognition import PreliminaryRecognition
from molass_legacy.Mapping.MapperConstructor import create_mapper

mapper = None
sd = None

def get_mapper(sd_orig):
    global mapper, sd

    if mapper is None:
        pre_recog = PreliminaryRecognition(sd_orig)
        sd = sd_orig._get_analysis_copy_impl(pre_recog)
        mapper = create_mapper(None, sd, sd_orig, pre_recog)

    return mapper, sd
