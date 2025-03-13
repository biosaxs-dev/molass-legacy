"""
    SMC.SolverLogLike.py

    Copyright (c) 2024, SAXS Team, KEK-PF    
"""
import numpy as np
import pytensor.tensor as pt
from pytensor.graph import Apply, Op

"""
    This implementation is based on the PyMC tutorial:
    https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html
    Using a “black box” likelihood function
    PyTensor Op without gradients
"""
class SolverLogLike(Op):
    def __init__(self, objective):
        self.objective = objective

    def make_node(self, params) -> Apply:
        # Convert inputs to tensor variables
        params = pt.as_tensor(params)

        inputs = [params]
        # Define output type, in our case a vector of likelihoods
        # with the same dimensions and same data type as data
        # If data must always be a vector, we could have hard-coded
        # outputs = [pt.vector()]
        outputs = [pt.dscalar()]

        # Apply is an object that combines inputs, outputs and an Op (self)
        return Apply(self, inputs, outputs)

    def perform(self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]) -> None:
        # This is the method that compute numerical output
        # given numerical inputs. Everything here is numpy arrays
        params, = inputs  # this will contain my variables

        # call our numpy log-likelihood function
        loglike_eval = -self.objective(params)
        # self.logger.info("loglike_eval=%s", loglike_eval)

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)

"""
pytensor.compile.function.types.UnusedInputError:
pytensor.function was asked to create a function computing outputs given certain inputs,
but the provided input variable at index 0 is not part of the computational graph
needed to compute the outputs: joined_inputs.
To make this error into a warning,
you can pass the parameter on_unused_input='warn' to pytensor.function.
To disable it completely, use on_unused_input='ignore'.
"""