# coding: utf-8
"""
    Result.py

    URL: http://stackoverflow.com/questions/1305532/convert-python-dict-to-object

    namedtuple の場合、メンバーの更新は can't set attritube により不可のようなので
    この Struct の方が柔軟。
"""

item_dict = {
    0 : 'Rg',
    1 : 'Rg_stdev',
    2 : 'I0',
    3 : 'I0_stdev',
    4 : 'From',
    5 : 'To',
    6 : 'Quality',
    7 : 'Aggregated',
    8 : 'min_qRg',
    9 : 'max_qRg',
    10 : 'gpfit_I0',
    11 : 'gpfit_Rg',
    12 : 'gpfit_d',
    13 : 'basic_quality',
    14 : 'positive_score',
    15 : 'fit_consistency_pure',
    16 : 'stdev_score',
    17 : 'q_rg_score',
    18 : 'IpI',
    19 : 'bicomponent',
    20 : 'bico_mono_ratio',
    21 : 'bico_G1',
    22 : 'bico_G2',
    23 : 'bico_Rg1',
    24 : 'bico_Rg2',
    25 : 'bico_d1',
    26 : 'bico_d2',
    }

class Result:
    def __init__( self, **entries ): 
        self.__dict__.update(entries)

    def __getitem__( self, i ):
        if i < 0:
            i = len( item_dict ) + i
        assert( i >= 0 and i < len( item_dict )  )
        return self.__dict__.get( item_dict[ i ] )

    def get_quality_signal( self ):
        if self.type == 'K':
            color = self.get_kekpf_signal()
        elif self.type == 'E':
            color = None
        else:
            color = self.get_atsas_signal()
        return color

    def get_atsas_signal( self ):
        quality = self.Quality
        aggregated_ = abs( self.Aggregated ) > 0.5

        if quality is None or quality <= 0.3:
            color = 'red'
        elif quality < 0.7 or aggregated_:
            color = 'yellow'
        else:
            color = 'green'

        return color

    def get_kekpf_signal( self ):
        quality = self.Quality

        if quality is None or quality <= 0.3:
            color = 'red'
        elif quality < 0.7:
            color = 'yellow'
        else:
            color = 'green'

        return color

