# coding: utf-8
"""
    Rgg.LogUtils.py

    Copyright (c) 2021, SAXS Team, KEK-PF
"""
import logging
import numpy as np
import re
from NumpyArrayUtils import from_space_separated_list_string
from molass_legacy.Batch.StandardProcedure import StandardProcedure
from .UvAdjuster import spike_demo

def find_best_score(in_folder, log_file):
    sp = StandardProcedure()
    sp.load(in_folder, debug=False)
    sd = sp.get_corrected_sd()

    logger = logging.getLogger()

    mixture_re = re.compile(r"Mixture\((\d+)\)")
    refiner_seeds_re = re.compile(r"Refiner\((\d+), (\d+)\)")
    trucate_re = re.compile(r"^(.+\]\]), func_value=(.+)")

    parsing_params = False
    parsing_start = False
    params_key = "params="

    seeds_list = []
    params_list = []
    fv_list = []
    score_list = []
    classify_key_list = []

    mm_no = -1
    params_count = 0
    fh = open(log_file)
    for k, line in enumerate(fh.readlines()):

        if line.find("Mixture") > 0:
            m = mixture_re.search(line)
            if m:
                mm_no += 1

        if line.find("Refiner") > 0:
            parsing_params = True
            parsing_start = True
            m = refiner_seeds_re.search(line)
            if m:
                seeds = [int(m.group(n)) for n in [1,2]]
            else:
                assert False

        if parsing_params:
            # print([k], line)
            if parsing_start:
                parsing_start = False
                pos = line.find(params_key) + len(params_key)
                params_str = line[pos:]
                params_count += 1
            else:
                if line.find("]]") > 0:
                    m = trucate_re.match(line)
                    if m:
                        params_str += m.group(1)
                        fv = float(m.group(2))
                        print([mm_no], "seeds=(%d, %d), fv=%.3g" % (*seeds, fv))
                    else:
                        assert False

                    if np.isfinite(fv):
                        fv_list.append(fv)
                        parsing_params = False
                        params = from_space_separated_list_string(params_str)
                        logger.info("seeds=%s, params={%s}", str(seeds), str(params))
                        fit_score = spike_demo(in_folder, sd=sd, xr_params=params, seeds=seeds, mm_no=mm_no)
                        logger.info("fit_score=%g", fit_score)
                        fv_list.append(fv)
                        seeds_list.append(seeds)
                        params_list.append(params)
                        score_list.append(fit_score)
                        # classify_key_list.append(np.concatenate([params[:,1], [fv]]))
                        classify_key_list.append(np.concatenate([[np.std(params[:,1])/10], [fv + fit_score]]))

                else:
                    params_str += line

    fh.close()

    fv_array = np.array(fv_list)
    n = np.argmin(fv_array)
    print([n], seeds_list[n], fv_array[n])

def get_result_from_log(seeds, test_log):
    from molass_legacy.KekLib.BasicUtils import Struct

    seeds_key = "(%d, %d)" % seeds
    re_opts = re.MULTILINE | re.DOTALL
    xr_params_re = re.compile(r"xr_params=(\[\[.*?\]\])", re_opts)
    rg_params_re = re.compile(r"rg_params=(\[.*?\])", re_opts)
    tricky_params_re = re.compile(r"tricky_params=(\[.*?\])", re_opts)
    a_b_params_re = re.compile(r"\(a, b\)=(\(.*?\))", re_opts)
    uv_params_re = re.compile(r"uv_params=(\[.*?\])", re_opts)
    c_d_params_re = re.compile(r"\(c, d\)=(\(.*?\))", re_opts)
    fv_re = re.compile(r"fv=(.+)\.", re_opts)

    held_lines = []
    fh = open(test_log)
    for k, line in enumerate(fh.readlines()):
        if line.find("init_seed") > 0:
            held_lines = []
        held_lines.append(line)
        if line.find(seeds_key) > 0:
            break
    fh.close()

    log_buffer = ''.join(held_lines)
    print(log_buffer)
    m = xr_params_re.search(log_buffer)
    if m:
        xr_params = from_space_separated_list_string(m.group(1))
        print("xr_params=", xr_params)
    m = rg_params_re.search(log_buffer)
    if m:
        rg_params = from_space_separated_list_string(m.group(1))
        print("rg_params=", rg_params)
    m = tricky_params_re.search(log_buffer)
    if m:
        tricky_params = from_space_separated_list_string(m.group(1))
        print("tricky_params=", tricky_params)
        (a, b) = tricky_params
    else:
        m = a_b_params_re.search(log_buffer)
        if m:
            (a, b) = eval(m.group(1))
            print("(a, b)=", (a, b))

    m = uv_params_re.search(log_buffer)
    if m:
        uv_params = from_space_separated_list_string(m.group(1))
        print("uv_params=", uv_params)
    m = c_d_params_re.search(log_buffer)
    if m:
        (c, d) = eval(m.group(1))
        print("(c, d)=", (c, d))

    m = fv_re.search(log_buffer)
    if m:
        fv = eval(m.group(1))
        print("fv=", fv)

    n = xr_params.shape[0]
    x = np.concatenate([xr_params.flatten(), rg_params, (a, b), uv_params, (c, d)])
    return n, Struct(x=x, fun=fv)
