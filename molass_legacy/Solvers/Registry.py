"""
    molass_legacy.Solvers.Registry

    Single source of truth for solver registration.

    Adding a new solver requires ONE file change: add an entry here.
    All other files (OptimizerUtils, Scripting, InProcessRunner, BasicOptimizer)
    derive their data from this registry automatically.

    Entry format:
        user_name  : str   — uppercase user-facing name (e.g. 'DE')
        impl_name  : str   — lowercase impl name used in BasicOptimizer.solve (e.g. 'de')
        int_code   : int   — integer stored in SerialSettings 'optimization_method'
        module     : str   — dotted module path to import
        class_name : str   — class name within the module
        settings_keys : list[str] — SerialSettings keys to pass as kwargs to solver __init__

    MCMC (emcee/pyabc/pymc) are legacy solvers with special nnn-based alternating dispatch;
    they are kept in the registry for completeness but BasicOptimizer handles them separately.

    Copyright (c) 2026, SAXS Team, KEK-PF
"""
from collections import namedtuple

SolverEntry = namedtuple('SolverEntry', [
    'impl_name',      # lowercase impl name
    'int_code',       # integer code stored in SerialSettings
    'module',         # dotted module path
    'class_name',     # class name
    'settings_keys',  # list of SerialSettings keys → solver __init__ kwargs
])

# ── Registry ─────────────────────────────────────────────────────────────────
# Ordered so that int_code matches list position (important for MCMC/SMC legacy logic).
SOLVER_REGISTRY = {
    'BH':    SolverEntry('bh',        0, 'molass_legacy.Solvers.BH.SolverBH',                   'SolverBH',       []),
    'NS':    SolverEntry('ultranest', 1, 'molass_legacy.Solvers.UltraNest.SolverUltraNest',      'SolverUltraNest', []),
    'MCMC':  SolverEntry('emcee',     2, 'molass_legacy.Solvers.MCMC.SolverEmcee',               'SolverEmcee',    []),
    'SMC':   SolverEntry('pyabc',     3, 'molass_legacy.Solvers.ABC.SolverPyABC',                'SolverPyABC',    []),
    'PYMC':  SolverEntry('pymc',      4, 'molass_legacy.Solvers.SMC.SolverPyMC',                 'SolverPyMC',     []),
    'CMA':   SolverEntry('cma',       5, 'molass.Solvers.CMA.SolverCMA',                         'SolverCMA',      []),
    'DE':    SolverEntry('de',        6, 'molass.Solvers.DE.SolverDE',                           'SolverDE',       ['de_pop_size', 'de_variant', 'de_F', 'de_CR']),
    'NSGA2': SolverEntry('nsga2',     7, 'molass.Solvers.NSGA2.SolverNSGA2',                     'SolverNSGA2',    []),
}

# Derived lists (for backward compat with OptimizerUtils)
METHOD_NAMES      = list(SOLVER_REGISTRY.keys())
IMPL_METHOD_NAMES = [e.impl_name for e in SOLVER_REGISTRY.values()]

# Map lowercase impl_name → user name (for _resolve_solver in InProcessRunner)
IMPL_TO_USER = {e.impl_name: user for user, e in SOLVER_REGISTRY.items()}

# Solvers with direct 1:1 impl mapping (as opposed to nnn-alternating MCMC/SMC)
DIRECT_SOLVERS = {user: e.impl_name for user, e in SOLVER_REGISTRY.items()
                  if user not in ('MCMC', 'SMC')}


def get_solver_instance(impl_name, optimizer):
    """Instantiate the solver for *impl_name* ('bh', 'de', 'nsga2', …).

    Loads the module (with reload for dev convenience), passes any
    SerialSettings-stored hyperparameters as kwargs.

    Parameters
    ----------
    impl_name : str
        Lowercase implementation name.
    optimizer : BasicOptimizer
        Passed as first positional arg to the solver constructor.

    Returns
    -------
    solver
        Ready-to-call solver with a ``minimize()`` method.
    """
    from importlib import import_module, reload as _reload

    # Look up by impl_name
    user_name = IMPL_TO_USER.get(impl_name)
    if user_name is None:
        raise ValueError(
            f"Unknown impl_name {impl_name!r}. "
            f"Register it in molass_legacy/Solvers/Registry.py."
        )
    entry = SOLVER_REGISTRY[user_name]

    # Reload for development convenience (mirrors existing pattern)
    mod = import_module(entry.module)
    _reload(mod)
    cls = getattr(mod, entry.class_name)

    # Collect solver-specific hyperparameters from SerialSettings
    if entry.settings_keys:
        from molass_legacy._MOLASS.SerialSettings import get_setting
        kw = {k.split('_', 1)[1]: v   # strip solver prefix: 'de_pop_size' → 'pop_size'
              for k in entry.settings_keys
              if (v := get_setting(k)) is not None}
    else:
        kw = {}

    return cls(optimizer, **kw)
