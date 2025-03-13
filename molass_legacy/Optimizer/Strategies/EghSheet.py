"""
    Optimizer.Strategies.EghSheet.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

def get_data_list(num_components):
    data_list = []
    for i in range(num_components):
        data_list.append(("h_%02d"%i, "", "1", ""))
        data_list.append(("mu_%02d"%i, "", "1", ""))
        data_list.append(("sigma_%02d"%i, "", "1", ""))
        data_list.append(("tau_%02d"%i, "", "1", ""))

    for i in range(num_components):
        data_list.append(("rg_%02d"%i, "", "", "1"))
    
    data_list.append(("xr_base_a", "1", "", ""))
    data_list.append(("xr_base_b", "1", "", ""))

    for i in range(num_components):
        data_list.append(("uvh_%02d"%i, "", "1", ""))

    data_list.append(("uv_base_L", "1", "", ""))
    data_list.append(("uv_base_x0", "1", "", ""))
    data_list.append(("uv_base_k", "1", "", ""))
    data_list.append(("uv_base_b", "1", "", ""))
    data_list.append(("uv_base_s1", "1", "", ""))
    data_list.append(("uv_base_s2", "1", "", ""))
    data_list.append(("uv_base_diff", "1", "", ""))

    data_list.append(("mapping_a", "", "1", ""))
    data_list.append(("mapping_b", "", "1", ""))

    data_list.append(("mappable_a", "", "1", ""))
    data_list.append(("mappable_b", "", "1", ""))

    data_list.append(("Npc", "", "", "1"))
    data_list.append(("poresize", "", "", "1"))
    data_list.append(("tI", "", "", "1"))
    data_list.append(("t0", "", "", "1"))
    data_list.append(("P", "", "", "1"))
    data_list.append(("m", "", "", "1"))
    return data_list