"""
    Solvers.ABC.CustomDistance.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from typing import Callable, Union
from pyabc.distance.base import Distance, NoDistance

class CustomFunctionDistance(Distance):
    """
    This is a wrapper around a simple function which calculates the distance.
    If a function/callable is passed to the ABCSMC class, which is not
    subclassed from pyabc.Distance, then it is converted to an instance of the
    SimpleFunctionDistance class.

    Parameters
    ----------
    fun: Callable[[dict, dict], float]
        A Callable accepting as parameters (a subset of) the arguments of the
        pyabc.Distance.__call__ function. Usually at least the summary
        statistics x and x_0. Returns the distance between both.
    """

    def __init__(self, fun):
        super().__init__()
        self.fun = fun

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None,
    ) -> float:
        return self.fun(x, x_0, t, par)

    def get_config(self):
        conf = super().get_config()
        # try to get the function name
        try:
            conf["name"] = self.fun.__name__
        except AttributeError:
            try:
                conf["name"] = self.fun.__class__.__name__
            except AttributeError:
                pass
        return conf

    @staticmethod
    def to_distance(maybe_distance: Union[Callable, Distance]) -> Distance:
        """
        Parameters
        ----------
        maybe_distance: either a Callable as in FunctionDistance, or a
        pyabc.Distance object.

        Returns
        -------
        A Distance instance.
        """
        if maybe_distance is None:
            return NoDistance()

        if isinstance(maybe_distance, Distance):
            return maybe_distance

        return CustomFunctionDistance(maybe_distance)

