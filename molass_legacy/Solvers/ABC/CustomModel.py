"""
    Solvers.ABC.CustomModel.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from typing import Any, Callable, Union

from pyabc.acceptor import Acceptor
from pyabc.distance import Distance
from pyabc.epsilon import Epsilon
from pyabc.parameters import Parameter

from pyabc.model import ModelResult


class CustomObjectiveModel:
    """
    General model. This is the most flexible model class, but
    also the most complicated one to use.
    This is an abstract class and not functional on its own.
    Derive concrete subclasses for actual usage.

    The individual steps

      * sample
      * summary_statistics
      * distance
      * accept

    can be overwritten.

    To use this class, at least the sample method has to be overriden.

    .. note::

        Most likely you do not want to use this class directly, but the
        :class:`FunctionModel` instead, or even just pass a plain function
        as model.

    Parameters
    ----------
    name: str, optional (default = "model")
        A descriptive name of the model. This name can simplify further
        analysis for the user as it is stored in the database.
    """

    def __init__(self, func, name: str = "Model"):
        self.func = func
        self.name = name

    def __repr__(self):
        return "<{} {}>".format(self.__class__.__name__, self.name)

    def sample(self, pars: Parameter):
        """
        Return a sample from the model evaluated at parameters ``pars``. This
        can be raw data, or already summarized statistics thereof.

        This method has to be implemented by any subclass.

        Parameters
        ----------
        pars: Parameter
            Dictionary of parameters.

        Returns
        -------
        sample: any
            The sampled data.
        """
        raise NotImplementedError()

    def summary_statistics(
        self, t: int, pars: Parameter, sum_stat_calculator: Callable
    ) -> ModelResult:
        """
        Sample, and then calculate the summary statistics.

        Called from within ABCSMC during the initialization process.

        Parameters
        ----------
        t: int
            Current time point.
        pars: Parameter
            Model parameters.
        sum_stat_calculator: Callable
            A function which calculates summary statistics, as passed to
            :class:`pyabc.smc.ABCSMC`.
            The user is free to use or ignore this function.

        Returns
        -------
        model_result: ModelResult
            The result with filled summary statistics.
        """
        if self.func is None:
            raw_data = self.sample(pars)
            sum_stat = sum_stat_calculator(raw_data)
        else:
            sum_stat = None
        return ModelResult(sum_stat=sum_stat)

    def distance(
        self,
        t: int,
        pars: Parameter,
        sum_stat_calculator: Callable,
        distance_calculator: Distance,
        x_0: dict,
    ) -> ModelResult:
        """
        Sample, calculate summary statistics, and then calculate the distance.

        Not required in the current implementation.

        Parameters
        ----------
        t: int
            Current time point.
        pars: Parameter
            Model parameters.
        sum_stat_calculator: Callable
            A function which calculates summary statistics, as passed to
            :class:`pyabc.smc.ABCSMC`.
            The user is free to use or ignore this function.
        distance_calculator: Callable
            A function which calculates the distance, as passed to
            :class:`pyabc.smc.ABCSMC`.
            The user is free to use or ignore this function.
        x_0: dict
            Observed summary statistics.

        Returns
        -------
        model_result: ModelResult
            The result with filled distance.
        """

        sum_stat_result = self.summary_statistics(t, pars, sum_stat_calculator)
        if self.func is None:
            distance = distance_calculator(sum_stat_result.sum_stat, x_0, t, pars)
        else:
            distance = self.func(pars)
        sum_stat_result.distance = distance

        return sum_stat_result

    def accept(
        self,
        t: int,
        pars: Parameter,
        sum_stat_calculator: Callable,
        distance_calculator: Distance,
        eps_calculator: Epsilon,
        acceptor: Acceptor,
        x_0: dict,
    ):
        """
        Sample, calculate summary statistics, calculate distance, and then
        accept or not accept a parameter.

        Called from within ABCSMC in each iteration to evaluate a parameter.


        Parameters
        ----------
        t: int
            Current time point.
        pars: Parameter
            The model parameters.
        sum_stat_calculator: Callable
            A function which calculates summary statistics.
            The user is free to use or ignore this function.
        distance_calculator: pyabc.Distance
            The distance function.
            The user is free to use or ignore this function.
        eps_calculator: pyabc.Epsilon
            The acceptance thresholds.
        acceptor: pyabc.Acceptor
            The acceptor judging whether to accept, based on distance and
            epsilon.
        x_0: dict
            The observed summary statistics.

        Returns
        -------
        model_result: ModelResult
            The result with filled accepted field.

        """
        result = self.summary_statistics(t, pars, sum_stat_calculator)
        acc_res = acceptor(
            distance_function=distance_calculator,
            eps=eps_calculator,
            x=result.sum_stat,
            x_0=x_0,
            t=t,
            par=pars,
        )
        result.distance = acc_res.distance
        result.accepted = acc_res.accept
        result.weight = acc_res.weight

        return result


class CustomObjectiveFunction(CustomObjectiveModel):
    """
    A model which is initialized with a function which generates the samples.
    For most cases this class will be adequate.
    Note that you can also pass a plain function to the ABCSMC class, which
    then gets automatically converted to a FunctionModel.

    Parameters
    ----------
    sample_function: Callable[[Parameter], Any]
        Returns the sample to be passed to the summary statistics method.
        This function as a single argument which is a Parameter.
    name: str. optional
        The name of the model. If not provided, the names if inferred from
        the function name of `sample_function`.
    """

    def __init__(
        self,
        objective_function,
        sample_function: Callable[[Parameter], Any],
        name: str = None
    ):
        self.objective_function = objective_function
        if name is None:
            # try to get the model name
            try:
                name = sample_function.__name__
            except AttributeError:
                name = sample_function.__class__.__name__
        super().__init__(name)
        self.sample_function = sample_function

    def sample(self, pars: Parameter):
        return self.sample_function(pars)

    @staticmethod
    def to_model(func, maybe_model: Union[Callable, CustomObjectiveModel]) -> CustomObjectiveModel:
        """
        Alternative constructor. Accepts either a Model instance or a
        function and returns always a Model instance.

        Parameters
        ----------
        maybe_model:
            Constructs a FunctionModel instance if a function is passed.
            If a Model instance is passed, the Model instance itself is
            returned.

        Returns
        -------
        model: A valid model instance
        """
        if isinstance(maybe_model, CustomObjectiveModel):
            return maybe_model
        else:
            return CustomObjectiveFunction(func, maybe_model)