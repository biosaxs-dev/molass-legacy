"""
    ABC.CustomSamplers.py

    see pyabc.platform_factory.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np
from typing import Union
from jabbar import jabbar
from pyabc.sampler import SingleCoreSampler

class DefaultSampler(SingleCoreSampler):
    def __init__(self, check_max_eval: bool = False):
        from importlib import reload
        import Solvers.ABC.CustomPopulation
        reload(Solvers.ABC.CustomPopulation)
        from Solvers.ABC.CustomPopulation import CustomSampleFactory

        self.nr_evaluations_: int = 0
        self.sample_factory: CustomSampleFactory = CustomSampleFactory(
            record_rejected=False
        )
        self.show_progress: bool = False
        self.analysis_id: Union[str, None] = None

        self.check_max_eval = check_max_eval

    def sample_until_n_accepted(
        self,
        n,
        simulate_one,
        t,
        *,
        max_eval=np.inf,
        all_accepted=False,
        ana_vars=None,
    ):
        nr_simulations = 0
        sample = self._create_empty_sample()

        for _ in jabbar(range(n), enable=self.show_progress, keep=False):
            while True:
                if self.check_max_eval and nr_simulations >= max_eval:
                    break
                new_sim = simulate_one()
                sample.append(new_sim)
                nr_simulations += 1
                if new_sim.accepted:
                    break
        self.nr_evaluations_ = nr_simulations

        if sample.n_accepted < n:
            sample.ok = False

        return sample