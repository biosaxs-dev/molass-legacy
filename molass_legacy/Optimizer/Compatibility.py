"""
Optimizer.Compatibility.py
"""
import os
import shutil
import matplotlib.pyplot as plt
import logging

def test_subprocess_optimizer_impl(mpl_monitor):
    working_folder = mpl_monitor.runner.working_folder.replace("/", "\\") 
    test_folder = os.path.abspath(working_folder[:-3] + "test_" + working_folder[-3:])      # last 3 chars should be numeric like 000
    if os.path.exists(test_folder):
        shutil.rmtree(test_folder)
    shutil.copytree(mpl_monitor.runner.working_folder, test_folder)
    log_file = os.path.join(test_folder, "optimizer.log")
    if os.path.exists(log_file):
        os.remove(log_file)
    mpl_monitor.run_impl(mpl_monitor.optimizer, mpl_monitor.init_params, work_folder=test_folder, optimizer_test=True, debug=True, devel=True)

def test_optimizer_compatibility(optimizer, params=None):
    """ Test the compatibility of the given optimizer instance.

    This function runs a series of checks to ensure that the optimizer's methods
    and attributes are functioning as expected. It is intended for debugging and
    validation purposes.

    Parameters
    ----------
    optimizer : Optimizer
        An instance of the Optimizer class to be tested.

    init_params : array-like, optional
        Initial parameters to be used in the test. If None, the optimizer's

    Returns
    -------
    None
    """
    logger = logging.getLogger("OptimizerCompatibilityTest")
    logger.info("Starting compatibility test for optimizer: %s", optimizer.__class__.__name__)
    optimizer.prepare_for_optimization(params)
    optimizer.objective_func(params, plot=True)
    plt.show()
    logger.info("Compatibility test completed.")