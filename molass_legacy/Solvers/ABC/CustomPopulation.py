"""
    Solvers.ABC.CustomPopulation.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from pyabc.population import Population, Sample, SampleFactory
import molass_legacy.KekLib.DebugPlot as plt

debug = False

class CustomSample(Sample):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_accepted_population(self) -> Population:
        """
        Returns
        -------
        population: Population
            A population of only the accepted particles.
        """
        global debug
        if debug:
            print("CustomSample: len(self.accepted_particles)=", len(self.accepted_particles))
            print("CustomSample: len(self.rejected_particles)=", len(self.rejected_particles))
            weights = [p.weight for p in self.accepted_particles]
            print("accepted_particles weights=", weights)

            with plt.Dp(button_spec=["OK", "Cancel"]):
                fig, ax = plt.subplots()
                ax.set_title("get_accepted_population")
                ax.plot(weights)
                fig.tight_layout()
                ret = plt.show()
                if not ret:
                    debug = False

        return Population(self.accepted_particles.copy())

class CustomSampleFactory(SampleFactory):
    def __call__(self, is_look_ahead: bool = False):
        """Create a new empty sample."""
        return CustomSample(
            record_rejected=self._record_rejected,
            max_nr_rejected=self._max_nr_rejected,
            is_look_ahead=is_look_ahead,
        )
