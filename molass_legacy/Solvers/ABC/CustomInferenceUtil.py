"""
    Solvers.ABC.CustomInferenceUtil.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import logging
from typing import Callable, List
import numpy as np
import pandas as pd

from pyabc.acceptor import Acceptor
from pyabc.distance import Distance
from pyabc.epsilon import Epsilon
from pyabc.model import Model
from pyabc.parameters import Parameter
from pyabc.population import Particle
from pyabc.random_choice import fast_random_choice
from pyabc.random_variables import RV, Distribution
from pyabc.transition import ModelPerturbationKernel, Transition
from pyabc.inference_util.inference_util import (
    create_prior_pdf,
    create_transition_pdf,
    # create_weight_function,   # customized below
    # generate_valid_proposal,
    # evaluate_proposal,        # customized below
    # only_simulate_data_for_proposal
)

logger = logging.getLogger("ABC")

def create_weight_function(
    prior_pdf: Callable,
    transition_pdf: Callable,
) -> Callable:
    """Create a function that calculates a sample's importance weight.
    The weight is the prior divided by the transition density and the
    acceptance step weight.

    Parameters
    ----------
    prior_pdf: The prior density.
    transition_pdf: The transition density.

    Returns
    -------
    weight_function: The importance sample weight function.
    """

    def weight_function(m_ss, theta_ss, acceptance_weight: float):
        """Calculate total weight, from sampling and acceptance weight.

        Parameters
        ----------
        m_ss: The model sample.
        theta_ss: The parameter sample.
        acceptance_weight: The acceptance weight sample. In most cases 1.

        Returns
        -------
        weight: The total weight.
        """
        # prior and transition density (can be equal)
        prior_pd = prior_pdf(m_ss, theta_ss)
        transition_pd = transition_pdf(m_ss, theta_ss)
        # calculate weight
        weight = acceptance_weight * prior_pd / transition_pd
        if False:
            if np.isnan(weight):
                print("m_ss=", m_ss)
                print("theta_ss=", theta_ss)
                print("acceptance_weight=", acceptance_weight)
                print("prior_pd=", prior_pd)
                print("transition_pd=", transition_pd)
                raise ValueError("weight is NaN")
        return weight

    return weight_function

def generate_valid_proposal(
    t: int,
    m: np.ndarray,
    p: np.ndarray,
    model_prior: RV,
    parameter_priors: List[Distribution],
    model_perturbation_kernel: ModelPerturbationKernel,
    transitions: List[Transition],
):
    """Sample a parameter for a model.

    Parameters
    ----------
    t: Population index to generate for.
    m: Indices of alive models.
    p: Probabilities of alive models.
    model_prior: The model prior.
    parameter_priors: The parameter priors.
    model_perturbation_kernel: The model perturbation kernel.
    transitions: The transitions, one per model.

    Returns
    -------
    (m_ss, theta_ss): Model, parameter.
    """
    # first generation
    if t == 0:
        # sample from prior
        m_ss = int(model_prior.rvs())
        theta_ss = parameter_priors[m_ss].rvs()
        return m_ss, theta_ss

    # later generation
    # counter
    n_sample, n_sample_soft_limit = 0, 1000
    # sample until the prior density is positive
    while True:
        if len(m) > 1:
            index = fast_random_choice(p)
            m_s = m[index]
            m_ss = model_perturbation_kernel.rvs(m_s)
            # theta_s is None if the population m_ss has died out.
            # This can happen since the model_perturbation_kernel
            # can return a model nr which has died out.
            if m_ss not in m:
                continue
        else:
            # only one model
            m_ss = m[0]
        theta_ss = transitions[m_ss].rvs()

        # check if positive under prior
        # if model_prior.pmf(m_ss) * parameter_priors[m_ss].pdf(theta_ss) > 0:
        mp = model_prior.pmf(m_ss)
        pp = parameter_priors[m_ss].pdf(theta_ss)
        # print("mp, pp=", mp, pp)
        if mp * pp > 0:
            return m_ss, theta_ss

        # unhealthy sampling detection
        n_sample += 1
        if n_sample == n_sample_soft_limit:
            logger.warning(
                "Unusually many (model, parameter) samples have prior "
                "density zero. The transition might be inappropriate."
            )

def evaluate_proposal(
    m_ss: int,
    theta_ss: Parameter,
    t: int,
    models: List[Model],
    summary_statistics: Callable,
    distance_function: Distance,
    eps: Epsilon,
    acceptor: Acceptor,
    x_0: dict,
    weight_function: Callable,
    proposal_id: int,
) -> Particle:
    """Evaluate a proposed parameter.

    Parameters
    ----------
    m_ss, theta_ss: The proposed (model, parameter) sample.
    t: The current time.
    models: List of all models.
    summary_statistics:
        Function to compute summary statistics from model output.
    distance_function: The distance function.
    eps: The epsilon threshold.
    acceptor: The acceptor.
    x_0: The observed summary statistics.
    weight_function: Function by which to reweight the sample.
    proposal_id: Id of the transition kernel.

    Returns
    -------
    particle: A particle containing all information.

    Data for the given parameters theta_ss are simulated, summary statistics
    computed and evaluated.
    """
    # simulate, compute distance, check acceptance
    model_result = models[m_ss].accept(
        t, theta_ss, summary_statistics, distance_function, eps, acceptor, x_0
    )

    # compute acceptance weight
    if model_result.accepted:
        weight = weight_function(m_ss, theta_ss, model_result.weight)
        if np.isnan(weight):
            weight = 0
            model_result.accepted = False
    else:
        weight = 0

    return Particle(
        m=m_ss,
        parameter=theta_ss,
        weight=weight,
        sum_stat=model_result.sum_stat,
        distance=model_result.distance,
        accepted=model_result.accepted,
        preliminary=False,
        proposal_id=proposal_id,
    )

def only_simulate_data_for_proposal(
    m_ss: int,
    theta_ss: Parameter,
    t: int,
    models: List[Model],
    summary_statistics: Callable,
    weight_function: Callable,
    proposal_id: int,
) -> Particle:
    """Simulate data for parameters.

    Similar to `evaluate_proposal`, however here for the passed parameters
    only data are simulated, but no distances calculated or acceptance
    checked. That needs to be done post-hoc then, not checked here."""

    # simulate
    model_result = models[m_ss].summary_statistics(
        t, theta_ss, summary_statistics
    )

    # dummies for distance and weight, need to be recomputed later
    distance = np.inf
    acceptance_weight = 1.0

    # needs to be accepted in order to be forwarded by the sampler, and so
    #  as a single particle
    accepted = True

    # compute weight
    weight = weight_function(m_ss, theta_ss, acceptance_weight)
    if np.isnan(weight):
        weight = 0
        accepted = False

    return Particle(
        m=m_ss,
        parameter=theta_ss,
        weight=weight,
        sum_stat=model_result.sum_stat,
        distance=distance,
        accepted=accepted,
        preliminary=True,
        proposal_id=proposal_id,
    )

def create_simulate_function(
    t: int,
    model_probabilities: pd.DataFrame,
    model_perturbation_kernel: ModelPerturbationKernel,
    transitions: List[Transition],
    model_prior: RV,
    parameter_priors: List[Distribution],
    models: List[Model],
    summary_statistics: Callable,
    x_0: dict,
    distance_function: Distance,
    eps: Epsilon,
    acceptor: Acceptor,
    evaluate: bool = True,
    proposal_id: int = 0,
) -> Callable:
    """
    Create a simulation function which performs the sampling of parameters,
    simulation of data and acceptance checking, and which is then passed
    to the sampler.

    Parameters
    ----------
    t: The time index to simulate for.
    model_probabilities: The last generation's model probabilities.
    model_perturbation_kernel: The model perturbation kernel.
    transitions: The parameter transition kernels.
    model_prior: The model prior.
    parameter_priors: The parameter priors.
    models: List of all models.
    summary_statistics:
        Function to compute summary statistics from model output.
    x_0: The observed summary statistics.
    distance_function: The distance function.
    eps: The epsilon threshold.
    acceptor: The acceptor.
    evaluate:
        Whether to actually evaluate the sample. Should be True except for
        certain preliminary settings.
    proposal_id:
        Identifier for the proposal distribution.

    Returns
    -------
    simulate_one: callable
        Function that samples parameters, simulates data, and checks
        acceptance.

    .. note::
        For some of the samplers, the sampling function needs to be
        serialized in order to be transported to where the sampling
        happens. Therefore, the returned function should be light, and
        in particular not contain references to the ABCSMC class.
    """
    # cache model_probabilities to not query the database so often
    m = np.array(model_probabilities.index)
    p = np.array(model_probabilities.p)

    # create prior and transition densities for weight function
    prior_pdf = create_prior_pdf(
        model_prior=model_prior, parameter_priors=parameter_priors
    )
    if t == 0:
        transition_pdf = prior_pdf
    else:
        transition_pdf = create_transition_pdf(
            transitions=transitions,
            model_probabilities=model_probabilities,
            model_perturbation_kernel=model_perturbation_kernel,
        )

    # create weight function
    weight_function = create_weight_function(
        prior_pdf=prior_pdf, transition_pdf=transition_pdf
    )

    # simulation function
    def simulate_one():
        proposed = False
        for n in range(10):
            parameter = generate_valid_proposal(
                t=t,
                m=m,
                p=p,
                model_prior=model_prior,
                parameter_priors=parameter_priors,
                model_perturbation_kernel=model_perturbation_kernel,
                transitions=transitions,
            )
            if evaluate:
                particle = evaluate_proposal(
                    *parameter,
                    t=t,
                    models=models,
                    summary_statistics=summary_statistics,
                    distance_function=distance_function,
                    eps=eps,
                    acceptor=acceptor,
                    x_0=x_0,
                    weight_function=weight_function,
                    proposal_id=proposal_id,
                )
                proposed = True
                break
            else:
                particle = only_simulate_data_for_proposal(
                    *parameter,
                    t=t,
                    models=models,
                    summary_statistics=summary_statistics,
                    weight_function=weight_function,
                    proposal_id=proposal_id,
                )
                if particle.accepted:
                    proposed = True
                    break
                else:
                    # discard this proposal
                    continue

        assert proposed

        return particle

    return simulate_one
