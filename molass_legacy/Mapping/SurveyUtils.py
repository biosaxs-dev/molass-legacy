"""
    SurveyUtils.py

    Copyright (c) 2023, SAXS Team, KEK-PF
"""

from molass_legacy._MOLASS.SerialSettings import get_setting

def write_info(survey_fh, mapper):
    in_folder = get_setting('in_folder')
    x = mapper.x_curve.x
    cx = x[len(x)//2]
    cx_uv = []
    A, B = mapper.map_params
    for a, b in [(mapper.A_init, mapper.B_init), (A, B)]:
        cx_uv.append(cx*a + b)
    diff = cx_uv[0] - cx_uv[1]
    survey_params = ["%g" % v for v in [mapper.A_init, mapper.B_init, A, B, diff, diff*len(x)/len(mapper.a_curve.x), *mapper.get_sci_list()]]
    survey_fh.write(",".join([in_folder] + survey_params) + "\n")
    survey_fh.flush()
