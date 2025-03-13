"""
    Solvers.ABC.CustomABC.py

        * custom_distance
        * loop callback

    Copyright (c) 2024, SAXS Team, KEK-PF    
"""

from typing import Callable, List, Tuple, TypeVar, Union
import numpy as np
from datetime import datetime, timedelta

from pyabc.inference.smc import (
    logger, model_output, identity, ABCSMC
)

# from pyabc.model import FunctionModel, Model
from pyabc.random_variables import RV, Distribution
from pyabc.sampler import Sampler
from pyabc.acceptor import (
    Acceptor,
    FunctionAcceptor,
    StochasticAcceptor,
    UniformAcceptor,
)
from pyabc.distance import (
    Distance,
    FunctionDistance,
    PNormDistance,
)
from .CustomDistance import CustomFunctionDistance

from pyabc.epsilon import Epsilon, MedianEpsilon
from pyabc.populationstrategy import ConstantPopulationSize, PopulationStrategy
from pyabc.transition import (
    ModelPerturbationKernel,
    MultivariateNormalTransition,
    Transition,
)
from importlib import reload
import Solvers.ABC.CustomInferenceUtil
reload(Solvers.ABC.CustomInferenceUtil)
from Solvers.ABC.CustomInferenceUtil import (
    create_simulate_function,
)
# from pyabc.platform_factory import DefaultSampler
from .CustomSamplers import DefaultSampler
from pyabc.storage import History
from .CustomModel import CustomObjectiveModel, CustomObjectiveFunction
from pyabc.weighted_statistics import effective_sample_size

class CustomABCSMC(ABCSMC):
    def __init__(
        self,
        models: Union[List[CustomObjectiveModel], CustomObjectiveModel, Callable],
        parameter_priors: Union[List[Distribution], Distribution, Callable],
        distance_function: Union[Distance, Callable] = None,
        population_size: Union[PopulationStrategy, int] = 100,
        summary_statistics: Callable[[model_output], dict] = identity,
        model_prior: RV = None,
        model_perturbation_kernel: ModelPerturbationKernel = None,
        transitions: Union[List[Transition], Transition] = None,
        eps: Epsilon = None,
        sampler: Sampler = None,
        acceptor: Acceptor = None,
        stop_if_only_single_model_alive: bool = False,
        max_nr_recorded_particles: int = np.inf,
        callback: Callable = None,
    ):
        def objective_function(parameter, x0):
            # assert False
            return 0

        self.objective_function = objective_function

        if not isinstance(models, list):
            models = [models]
        models = list(map(lambda x:CustomObjectiveFunction.to_model(self.objective_function, x), models))
        self.models = models

        if not isinstance(parameter_priors, list):
            parameter_priors = [parameter_priors]
        self.parameter_priors = parameter_priors

        # sanity checks
        if len(self.models) != len(self.parameter_priors):
            raise AssertionError(
                "Number models and number parameter priors have to agree."
            )

        assert distance_function is not None
 
        self.distance_function = CustomFunctionDistance.to_distance(
            distance_function,
        )

        self.summary_statistics = summary_statistics

        if model_prior is None:
            model_prior = RV("randint", 0, len(self.models))
        self.model_prior = model_prior

        if model_perturbation_kernel is None:
            model_perturbation_kernel = ModelPerturbationKernel(
                len(self.models), probability_to_stay=0.7
            )
        self.model_perturbation_kernel = model_perturbation_kernel

        if transitions is None:
            transitions = [MultivariateNormalTransition() for _ in self.models]
        if not isinstance(transitions, list):
            transitions = [transitions]
        self.transitions: List[Transition] = transitions

        if eps is None:
            eps = MedianEpsilon(median_multiplier=1)
        self.eps = eps

        if isinstance(population_size, int):
            population_size = ConstantPopulationSize(population_size)
        self.population_size = population_size

        if sampler is None:
            sampler = DefaultSampler()
        self.sampler = sampler

        if acceptor is None:
            acceptor = UniformAcceptor()
        self.acceptor = FunctionAcceptor.to_acceptor(acceptor)

        self.stop_if_only_single_model_alive = stop_if_only_single_model_alive
        self.max_nr_recorded_particles = max_nr_recorded_particles
        self.callback = callback

        # will be set later
        self.x_0 = None
        self.history = None
        self._initial_population = None
        self.minimum_epsilon = None
        self.max_nr_populations = None
        self.min_acceptance_rate = None
        self.max_t = None
        self.max_total_nr_simulations = None
        self.max_walltime = None
        self.min_eps_diff = None

        self.init_walltime = None
        self.analysis_id = None

        self._sanity_check()

    def _create_simulate_function(self, t: int):
        """
        Create a simulation function which performs the sampling of parameters,
        simulation of data and acceptance checking, and which is then passed
        to the sampler.

        Parameters
        ----------
        t: int
            Time index

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
        return create_simulate_function(
            t=t,
            model_probabilities=self.history.get_model_probabilities(t - 1),
            model_perturbation_kernel=self.model_perturbation_kernel,
            transitions=self.transitions,
            model_prior=self.model_prior,
            parameter_priors=self.parameter_priors,
            models=self.models,
            summary_statistics=self.summary_statistics,
            x_0=self.x_0,
            distance_function=self.distance_function,
            eps=self.eps,
            acceptor=self.acceptor,
        )

    def run_generation(
        self,
        t: int,
    ) -> dict:
        """Run a single generation.

        Parameters
        ----------
        t: Generation time index to run for.

        Returns
        -------
        ret:
            Dictionary with entries "successful" indicating whether the
            generation terminated successfully,
            and potentially "acceptance_rate".
        """
        # get epsilon for generation t
        current_eps = self.eps(t)
        if current_eps is None or np.isnan(current_eps):
            raise ValueError(
                f"The epsilon threshold {current_eps} is invalid."
            )
        logger.info(f"t: {t}, eps: {current_eps:.8e}.")

        # create simulate function
        simulate_one = self._create_simulate_function(t)

        # population size and maximum number of evaluations
        pop_size = self.population_size(t)
        max_eval = (
            np.inf
            if self.min_acceptance_rate == 0.0
            else pop_size / self.min_acceptance_rate
        )

        # perform the sampling
        logger.debug(f"Submitting population {t}.")
        sample = self.sampler.sample_until_n_accepted(
            n=pop_size,
            simulate_one=simulate_one,
            t=t,
            max_eval=max_eval,
            ana_vars=self._vars(t=t),
        )

        # check sample health
        if not sample.ok:
            logger.info("Stopping: sample not ok.")
            return {
                "successful": False,
            }

        # normalize accepted population weight to 1
        sample.normalize_weights()

        # retrieve accepted population
        population = sample.get_accepted_population()
        logger.debug(f"Population {t} done.")

        # save to database
        n_sim = self.sampler.nr_evaluations_
        model_names = [model.name for model in self.models]
        self.history.append_population(
            t, current_eps, population, n_sim, model_names
        )
        logger.debug(
            f"Total samples up to t = {t}: "
            f"{self.history.total_nr_simulations}."
        )

        # acceptance rate and ess
        pop_size = len(population)
        acceptance_rate = pop_size / n_sim
        ess = effective_sample_size(population.get_weighted_distances()['w'])
        logger.info(
            f"Accepted: {pop_size} / {n_sim} = "
            f"{acceptance_rate:.4e}, ESS: {ess:.4e}."
        )

        if self.callback is not None:
            params = self.history_to_params()
            self.callback(params, None, True)

        # prepare next iteration
        self._prepare_next_iteration(
            t=t + 1,
            sample=sample,
            population=population,
            acceptance_rate=acceptance_rate,
        )

        return {
            "successful": True,
            "acceptance_rate": acceptance_rate,
        }

    def history_to_params(self):
        """
        see get_min_dist_parameter in the spike notebook
        """
        history = self.history

        population = history.get_population(history.max_t)
        distance_df = population.get_weighted_distances()
        
        k = np.argmin(distance_df['distance'].values)
        df, w = history.get_distribution(m=0, t=history.max_t)
        return df.iloc[k].values