# coding: utf-8
"""

    AnimObjects.py

        recognition of peaks

    Copyright (c) 2018, Masatsuyo Takahashi, KEK-PF

"""
import copy

class AnimMapper:
    def __init__( self, mapper ):
        self.mapper     = mapper
        self.opt_params = mapper.opt_params
        self.a_curve    = mapper.a_curve
        self.a_spline   = mapper.a_spline
        self.a_vector   = mapper.a_vector
        self.a_base     = mapper.a_base
        self.a_base_adjustment   = 0
        self.flow_changes   = mapper.flow_changes
        self.x_curve    = mapper.x_curve
        self.x_spline   = mapper.x_spline
        self.x_vector   = mapper.x_vector
        self.x_base     = mapper.x_base
        self.x_base_adjustment   = 0
        self.x_ranges   = None
        self.uv_ranges  = None
        self.mapping_ranges = None
        self.opt_results    = None
        self.mapped_vector  = None
        self.mapped_spline  = None
        self.a_base_adjustment  = 0
        self.x_base_adjustment  = 0
        self.inv_mapped_boundaries  = None
        self.x_curve_y_adjusted = None
        self.in_opt = False
        self.in_adj = False
        self.opt_phase = 0

    def copy( self, deep_ranges=False, with_adj=False ):
        mapper  = self.mapper
        obj = AnimMapper( mapper )
        obj.opt_phase = self.opt_phase      # opt_phase of the copy source
        try:
            if deep_ranges:
                obj.x_ranges        = copy.deepcopy( mapper.x_ranges )
                obj.uv_ranges       = copy.deepcopy( mapper.uv_ranges )
                obj.mapping_ranges  = copy.deepcopy( mapper.mapping_ranges )
            else:
                obj.x_ranges        = mapper.x_ranges
                obj.uv_ranges       = mapper.uv_ranges
                obj.mapping_ranges  = mapper.mapping_ranges
        except:
            pass
        try:
            if with_adj:
                obj.a_base_adjustment   = copy.deepcopy( mapper.a_base_adjustment )
                obj.x_base_adjustment   = copy.deepcopy( mapper.x_base_adjustment )
            obj.opt_results     = mapper.opt_results
        except:
            pass

        try:
            obj.mapped_vector   = mapper.mapped_vector
            obj.mapped_spline   = mapper.mapped_spline
        except:
            pass

        try:
            obj.x_curve_y_adjusted      = mapper.x_curve_y_adjusted
        except:
            pass

        try:
            obj.inv_mapped_boundaries   = mapper.inv_mapped_boundaries
        except:
            pass

        return obj
