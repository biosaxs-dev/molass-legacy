"""
    GuinierTools.RgCurveDbMaker.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import os
import logging

class RgCurveDbMaker:
    def __init__(self, db_folder):
        assert os.path.exists(db_folder)
        self.db_folder = db_folder
        logfile = os.path.join(self.db_folder, "dbmaker.log")
        logging.basicConfig(filename=logfile,
                            level=logging.INFO,
                            format='%(asctime)s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        self.logger = logging.getLogger(__name__)

    def make_rg_curves(self, restart_count=None):
        from molass_legacy.Batch.DataWalk import test_data_walk
        print("make_rg_curves: ")
        self.restart_count = restart_count

        counters = [0]
        test_data_walk(self.make_call_back, counters=counters)

    def make_call_back(self, in_folder, batch, count, **kwargs):
        from molass_legacy.SerialAnalyzer.DataUtils import cut_upper_folders
        from RgProcess.RgCurve import RgCurve

        if self.restart_count is None:
            pass
        else:
            if count < self.restart_count:
                self.logger.info("make_call_back: skipping %s, %s" % (in_folder, count))
                return True

        print("make_call_back: ", in_folder, count, kwargs)
        sd = batch.load_data(in_folder)
        batch.prepare(sd)
        D_, E_, qv_, xr_curve_ = batch.corrected_sd.get_xr_data_separate_ly()
        rg_curve = RgCurve(qv_, xr_curve_, D_, E_)
        folder = cut_upper_folders(in_folder)
        save_folder = os.path.join(self.db_folder, folder)
        os.makedirs(save_folder, exist_ok=True)
        rg_curve.export(save_folder)
        self.logger.info("make_call_back: %s, %s, ok", in_folder, count)
        return True

if __name__ == '__main__':
    import sys
    this_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.join(this_dir, '..')
    sys.path.append(lib_dir)
    maker = RgCurveDbMaker(r'D:\PyTools\RgCurves')
    maker.make_rg_curves(restart_count=67)