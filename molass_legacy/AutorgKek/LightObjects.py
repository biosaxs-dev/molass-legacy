# coding: utf-8
"""
    LightObjects.py

    Objects to facilitate garbage collection

    Copyright (c) 2017, Masatsuyo Takahashi, KEK-PF
"""
class AnyObject:
    def __init__( self, **entries ): 
        self.__dict__.update(entries)

class LightIntensity:
    def __init__( self, intensity ):
        if intensity is None:
            self.orig_array = None
        else:
            self.orig_array = intensity.orig_array
            self.comments   = '' 

class LightFit:
    def __init__( self, fit ):
        self.Rg     = fit.Rg
        self.I0     = fit.I0
        self.degree = fit.degree
        self.result = AnyObject( aic=fit.result.aic, bic=fit.result.bic )

class LightResult:
    def __init__( self, result ):
        self.Rg             = result.Rg
        self.Rg_            = result.Rg_
        self.Rg_stdev       = result.Rg_stdev
        self.I0             = result.I0
        self.I0_            = result.I0_
        self.I0_stdev       = result.I0_stdev
        self.From           = result.From
        self.To             = result.To
        self.min_qRg        = result.min_qRg
        self.max_qRg        = result.max_qRg
        self.fit            = LightFit( result.fit )
        self.bico_mono_ratio    = result.bico_mono_ratio
        self.IpI            = result.IpI
        self.bicomponent    = result.bicomponent
        self.head_trend     = result.head_trend
        self.result_type    = result.result_type
        self.quality_object = result.quality_object
        self.Quality        = result.Quality
