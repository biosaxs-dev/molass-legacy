"""
    _MOLASS.DummyClasses.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
class Dummy:
    def __repr__(self):
        return "%s()" % self.__class__.__qualname__

class DummySd(Dummy):
    def __init__(self):
        self.mc_vector = None

class DummyMapper(Dummy):
    def __init__(self):
        self.x_ranges = None
 
class DummyJudgeHolder(Dummy):
    def __init__(self):
        pass

class PeakInfo(Dummy):
    def __init__(self, *args, **kwargs):
        pass

class DecompEditorInfo(Dummy):
    def __init__(self, *args, **kwargs):
        pass

class RangeEditorInfo(Dummy):
    def __init__(self, *args, **kwargs):
        pass

class PairedRange(Dummy):
    def __init__(self, *args, **kwargs):
        pass

class PreviewData(Dummy):
    def __init__(self, *args, **kwargs):
        pass

class PreviewOptions(Dummy):
    def __init__(self, *args, **kwargs):
        pass

class MappingParams(Dummy):
    def __init__(self, *args, **kwargs):
        pass

class TrimmingInfo(Dummy):
    def __init__(self, *args, **kwargs):
        pass

class ModelEvaluator(Dummy):
    def __init__(self, *args, **kwargs):
        pass

class DualEvaluator(Dummy):
    def __init__(self, *args, **kwargs):
        pass

class EGHA(Dummy):
    def __init__(self, *args, **kwargs):
        pass

class EMGA(Dummy):
    def __init__(self, *args, **kwargs):
        pass

if __name__ == '__main__':
    print(DummySd())
