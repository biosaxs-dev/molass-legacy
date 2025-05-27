"""
    Simulative.DevelTest.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

def devel_test_impl(editor):
    print("devel_test_impl")
    from importlib import reload
    import Batch.DataBridgeUtils
    reload(Batch.DataBridgeUtils)
    from molass_legacy.Batch.DataBridgeUtils import get_lrf_source_impl

    lrf_src = get_lrf_source_impl(editor)

    def moments_study():
        print("another_monopore_study")
        import Models.Stochastic.MomentsStudy
        reload(Models.Stochastic.MomentsStudy)
        from molass_legacy.Models.Stochastic.MomentsStudy import moments_study_impl
        moments_study_impl(lrf_src, debug=True)

    def show_num_plates_demo():
        import SecTheory.MartinSynge
        reload(SecTheory.MartinSynge)
        from SecTheory.MartinSynge import num_plates_demo
        num_plates_demo()

    def show_current_model_info():
        model = editor.get_current_model()
        modelname = editor.get_current_modelname()
        params_array = editor.get_current_params_array()
        # print("show_current_model_info", modelname)
        # print(params_array)
        return model, modelname, params_array

    def dispersive_pdf():
        import Models.Stochastic.DispersivePdfStudy
        reload(Models.Stochastic.DispersivePdfStudy)
        from molass_legacy.Models.Stochastic.DispersivePdfStudy import study_pdf
        study_pdf()

    def dispersive_time():
        import Models.Stochastic.DispersivePdf
        reload(Models.Stochastic.DispersivePdf)
        from molass_legacy.Models.Stochastic.DispersivePdf import timescale_proof
        timescale_proof()

    def dispersive_study(debug=False):
        import Models.Stochastic.DispersiveStudy
        reload(Models.Stochastic.DispersiveStudy)
        from molass_legacy.Models.Stochastic.DispersiveStudy import study
        rgs, trs, orig_props, peak_rgs, peak_trs, props, indeces, qualities = lrf_src.compute_rgs(want_num_components=4, debug=False)
        info = show_current_model_info()
        study(lrf_src.xr_x, lrf_src.xr_y, lrf_src.baselines[1], lrf_src.model, lrf_src.xr_peaks[indeces,:], peak_rgs, qualities, props, curent_info=info, debug=debug)

    def test_stochastic_adapter(editor):
        print("test_stochastic_adapter")
        from importlib import reload
        import Selective.StochasticAdapter
        reload(Selective.StochasticAdapter)
        from Selective.StochasticAdapter import convert_to_stochastic_decomposition
        convert_to_stochastic_decomposition(editor, debug=True)

    extra_button_specs = [
            ("Guess Lnpore Params", lrf_src.guess_lnpore_params),
            ("Moments Study", moments_study),
            ("Moments Demo", lrf_src.moments_demo),
            ("Monopore Study", lrf_src.monopore_study),
            ("Num Plates Demo", show_num_plates_demo),
            ("Current Model Info", show_current_model_info),
            ("Dispersive PDF", dispersive_pdf),
            ("Dispersive TIme", dispersive_time),
            ("Dispersive Debug", lambda: dispersive_study(debug=True)),
            ("Dispersive Study", lambda: dispersive_study(debug=False)),
            ("Lnpore Study", lrf_src.lnpore_study),
            # ("Stochastic Adapter", lambda : test_stochastic_adapter(editor)),
            ] 

    lrf_src.draw(extra_button_specs=extra_button_specs)