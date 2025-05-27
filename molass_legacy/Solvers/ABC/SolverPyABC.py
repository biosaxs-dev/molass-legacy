"""
    ABC.SolverPyABC.py

    Copyright (c) 2024, SAXS Team, KEK-PF    
"""
import os
import logging
from importlib import reload
from molass_legacy._MOLASS.WorkUtils import get_temp_folder
from molass_legacy.Optimizer.StateSequence import save_opt_params
from molass_legacy.Optimizer.OptimizerUtils import OptimizerResult

prior_module_dict = {

}

class SolverPyABC:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.num_pure_components = optimizer.num_pure_components
        self.cb_fh = optimizer.cb_fh
        self.nfev = 0
        self.logger = logging.getLogger(__name__)
    
    def minimize(self, objective, init_params, niter=100, seed=1234, bounds=None, callback=None):
        import Solvers.ABC.ConstrainedPrior
        reload(Solvers.ABC.ConstrainedPrior)
        from Solvers.ABC.ConstrainedPrior import ConstrainedPrior
        import Solvers.ABC.ParameterUtils
        reload(Solvers.ABC.ParameterUtils)
        from Solvers.ABC.ParameterUtils import parameter_to_vector
        import Solvers.ABC.CustomABC
        reload(Solvers.ABC.CustomABC)
        from Solvers.ABC.CustomABC import CustomABCSMC as ABCSMC
        import Solvers.ABC.HistoryUtils
        reload(Solvers.ABC.HistoryUtils)
        from Solvers.ABC.HistoryUtils import get_min_dist_parameter

        self.objective = objective

        lower = bounds[:,0]
        upper = bounds[:,1]
        constrained_prior = ConstrainedPrior(
            (self.num_pure_components, 4),  # for EGH
            lower, upper,
            offset=1,                       # for EGH
            )
       
        def objective_func_wrapper(x, x0, t, par):
            # print("objective_func_wrapper: par=", par)
            params = parameter_to_vector(par)
            fv = objective(params)
            # print("objective_func_wrapper: fv=", fv)
            return fv

        # task: remove objective_function=objective

        abc = ABCSMC(None, constrained_prior, objective_func_wrapper, population_size=200, callback=self.callback)
        db_path = os.path.join(get_temp_folder(), "abc-history.db")
        print("db_path=", db_path)
        observation = self.optimizer.xr_curve.y
        abc.new("sqlite:///" + db_path, {"data": observation})
        history = abc.run(minimum_epsilon=0.1, max_nr_populations=30)
        print("history.max_t=", history.max_t)
        parameter = get_min_dist_parameter(history)
        opt_params = parameter_to_vector(parameter)

        return OptimizerResult(x=opt_params, nit=niter, nfev=self.nfev)

    def callback(self, norm_params, f, accept):
        fv = self.objective(norm_params)
        real_params = self.optimizer.to_real_params(norm_params)
        self.logger.info("callback: fv=%.3g", fv)
        save_opt_params(self.cb_fh, real_params, fv, accept, self.nfev)
        return False
