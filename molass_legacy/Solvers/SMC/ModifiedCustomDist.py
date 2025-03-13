"""
    SMC.ModifiedCustomDist.py
    
    Copyright (c) 2024, SAXS Team, KEK-PF   
"""
import numpy as np
import pymc as pm

class ModifiedCustomDist(pm.CustomDist):
    """
    from the original CustomDist class documentation
    https://github.com/pymc-devs/pymc/blob/main/pymc/distributions/custom.py

        random : Optional[Callable]
            A callable that can be used to generate random draws from the distribution

            It must have the following signature: ``random(*dist_params, rng=None, size=None)``.
            The numerical distribution parameters are passed as positional arguments in the
            same order as they are supplied when the ``CustomDist`` is constructed.
            The keyword arguments are ``rng``, which will provide the random variable's
            associated :py:class:`~numpy.random.Generator`, and ``size``, that will represent
            the desired size of the random draw. If ``None``, a ``NotImplemented``
            error will be raised when trying to draw random samples from the distribution's
            prior or posterior predictive.
    """
    def __new__(cls, name, *args, **kwargs):
        def rand_cunstom(params, rng=None, size=None):

            return 
        kwargs['random'] = rand_cunstom
        return pm.CustomDist.__new__(cls, name, *args, **kwargs)