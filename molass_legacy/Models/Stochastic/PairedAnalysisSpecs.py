"""
    Models.Stochastic.PairedAnalysisSpecs.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
PORESIZE_BOUNDS = 75, 100
NUM_PLATES = 20000      # 14400*1.36 = 19584, statinary 0.6**2 : mobile 1.0**2
NUM_PLATES_THIN = 5000

def get_exec_spec(n):
    from molass_legacy.SerialAnalyzer.DataUtils import get_local_path

    PyTools = get_local_path('PyTools')

    exec_spec_list = [
        {
        "input_folders" :       [ PyTools + r"\Data\20210727\data01",   PyTools + r"\Data\20210727\data02"],
        "num_components" :      4,
        "unreliable_indeces" :  [1, 2],
        "poresize_bounds" :     PORESIZE_BOUNDS,
        "init_N0" :             [NUM_PLATES, NUM_PLATES_THIN ],
        },

        {
        "input_folders" :       [ PyTools + r"\Data\20220716\BSA_201",   PyTools + r"\Data\20220716\BSA_202"],
        "num_components" :      4,
        "unreliable_indeces" :  [0],
        "poresize_bounds" :     PORESIZE_BOUNDS,
        "init_N0" :             [NUM_PLATES, NUM_PLATES_THIN],
        },

        {
        "input_folders" :       [ PyTools + r"\Data\20220716\FER_OA_301",   PyTools + r"\Data\20220716\FER_OA_302"],
        "num_components" :      4,
        "unreliable_indeces" :  [0, 2],
        "poresize_bounds" :     PORESIZE_BOUNDS,
        "init_N0" :             [NUM_PLATES, NUM_PLATES_THIN],
        },

        {
        "input_folders" :       [ PyTools + r"\Data\20220716\OA_ALD_201",   PyTools + r"\Data\20220716\OA_ALD_202"],
        "num_components" :      4,
        "unreliable_indeces" :  [0, 2],
        "use_mapping" : True,
        "poresize_bounds" :     PORESIZE_BOUNDS,
        "init_N0" :             [NUM_PLATES, NUM_PLATES_THIN],
        },

        {
        "input_folders" :       [ PyTools + r"\Data\20230706\ALD_OA001",   PyTools + r"\Data\20230706\ALD_OA002"],
        "num_components" :      3,
        "unreliable_indeces" :  [],
        "poresize_bounds" :     (75, 300),
        "init_N0" :             [20000, 20000],
        },

        {
        "input_folders" :       [ PyTools + r"\Data\20230706\BSA001",   PyTools + r"\Data\20230706\BSA002"],
        "num_components" :      4,
        "components_to_use" :   [[0, 1, 2, 3], None],
        "unreliable_indeces" :  [0],
        "poresize_bounds" :     (75, 300),
        "init_N0" :             [20000, 20000],
        },
    ]

    return exec_spec_list[n]

def get_components_to_use(exec_spec):
    components_to_use = exec_spec.get('components_to_use')
    if components_to_use is None:
        return [None, None]
    else:
        return components_to_use