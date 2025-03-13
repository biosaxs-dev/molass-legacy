"""
    FitRecord.py

    Copyright (c) 2018-2024, SAXS Team, KEK-PF
"""
import copy

USE_MODEL_TO_GET_RANGES = True
MIN_SEPARATE_RANGE_LENGTH = 10
MIN_SEPARATE_AREA_RATIO = 0.1
IGNORABLE_AREA_RATIO = 0.03

class FitRecord:
    def __init__(self, kno, evaluator, chisqr_n, peak, major_pno=None, area=None):
        self.kno    = kno       # negative for smaller peaks
        self.evaluator = evaluator
        self.chisqr_n = chisqr_n
        self.peak = peak
        self.major_pno = major_pno
        self.area = area
        self.item_list = [self.kno, self.evaluator, self.chisqr_n, self.peak, self.major_pno, self.area]

    def set_evaluator(self, evaluator):
        self.evaluator = evaluator
        self.item_list[1] = evaluator

    def get_params(self):
        return self.evaluator.get_all_param_values()

    def scale_to(self, other, conc_factor):
        new_evaluator = copy.deepcopy(self.evaluator)   # without this copy, self.evaluator would be changed
        new_rec = FitRecord(self.kno, new_evaluator, self.chisqr_n, self.peak, self.major_pno)
        other_h = other.evaluator.get_param_value(0)
        new_rec.evaluator.update_param(0, other_h*conc_factor)
        if True:
            this_h_old = self.evaluator.get_param_value(0)
            this_h_new = new_rec.evaluator.get_param_value(0)
            print('scale_to:', this_h_old, other_h, this_h_new)
        return new_rec

    def __getitem__(self, index):
        return self.item_list[index]

    def __repr__(self):
        return "<FitRecord %s>" % str(self.item_list)

    def __lt__(self, other):
        # neccesary in case when sorted( temp_pair_list ) fails
        return self.evaluator.get_param_value(1) < other.evaluator.get_param_value(1)

    def get_range_list(self, x, debug=True):
        # print('get_range_list: self=', self)

        rec_id = self.kno
        peak = self.peak

        if debug:
            print('get_range_list: peak.area_prop=', peak.area_prop)

        if USE_MODEL_TO_GET_RANGES:
            model_evaluator = self[1]
            foot_L, top_x, foot_R = model_evaluator.get_range_params(x)
            if (rec_id >= 0 and foot_R - foot_L >= MIN_SEPARATE_RANGE_LENGTH
                and (peak.area_prop is None or peak.area_prop > MIN_SEPARATE_AREA_RATIO)):
                top_x_ = int(round(top_x))
                range_list = [ [ foot_L, top_x_ ], [ top_x_, foot_R ] ]
            else:
                range_list = [ [ foot_L, foot_R ] ]
        else:
            if rec_id >= 0:
                range_list = [ [ peak.sigma_points[0], peak.top_x_m ], [ peak.top_x_m, peak.sigma_points[1] ] ]
            else:
                range_list = [ [ peak.sigma_points[0], peak.sigma_points[1] ] ]

        return range_list

def sort_fit_recs(recs):
    return sorted(recs, key=lambda p:p.peak.top_x)
