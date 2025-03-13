"""
    ErrorResult.py

    Copyright (c) 2016-2023, SAXS Team, KEK-PF
"""
class ErrorFitResult:
    def __init__( self ):
        self.aic    = None
        self.bic    = None

class ErrorFit:
    def __init__( self ):
        self.Rg     = None
        self.degree = None
        self.result = ErrorFitResult()
        self.I0     = None

class ErrorResult:
    def __init__( self ):
        self.Rg         = None
        self.Rg_        = 0
        self.Rg_stdev   = 0
        self.I0         = None
        self.I0_        = 0
        self.I0_stdev   = 0
        self.From       = None
        self.To         = None
        self.min_q      = None
        self.max_q      = None
        self.min_qRg    = None
        self.max_qRg    = None
        self.fit        = ErrorFit()
        self.bico_mono_ratio    = None
        self.IpI            = None
        self.bicomponent    = None
        self.head_trend     = None
        self.result_type    = None
        self.quality_object = None
        self.basic_quality  = 0
        self.Quality        = None
