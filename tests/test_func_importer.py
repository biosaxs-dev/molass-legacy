"""
Tests for FuncImporter.import_objective_function's error-handling contract
(molass-legacy#102): it must return ``None`` only when the requested model's
ObjectiveFunctions module genuinely doesn't exist, and must re-raise (not
silently swallow) any other failure -- e.g. BasicOptimizer or one of its
dependencies failing to reload, which previously got misreported by
``construct_legacy_optimizer`` as "model not supported".
"""
import sys
from importlib import reload
from unittest.mock import patch

import pytest

from molass_legacy.Optimizer.FuncImporter import import_objective_function


def test_returns_none_for_genuinely_missing_model():
    # No ObjectiveFunctions/G9999.py exists on disk.
    assert import_objective_function("G9999") is None


def test_returns_class_for_existing_model():
    # G1500 (GRM) is a real, existing ObjectiveFunctions module.
    class_ = import_objective_function("G1500")
    assert class_ is not None
    assert class_.__name__ == "G1500"


def test_propagates_basic_optimizer_reload_failure():
    """A BasicOptimizer reload failure must raise, not return None.

    Reproduces the stale-module scenario: BasicOptimizer (shared by every
    model) fails to reload for a reason unrelated to which model was asked
    for, so it must never be reported as "model not supported".
    """
    with patch(
        "molass_legacy.Optimizer.FuncImporter.reload",
        side_effect=ImportError("cannot import name 'compute_uv_domain_mask'"),
    ):
        with pytest.raises(ImportError):
            import_objective_function("G1500")


def test_propagates_failure_from_existing_module_import():
    """The ObjectiveFunctions module existing but failing on import (e.g. one
    of *its* imports is missing) must raise, not be swallowed into None."""
    module_name = "molass_legacy.ObjectiveFunctions.G1500"
    # Simulate "the module exists but a dependency it imports does not" by
    # raising a ModuleNotFoundError whose .name differs from module_name.
    missing = ModuleNotFoundError("No module named 'bogus_dependency'")
    missing.name = "bogus_dependency"

    def fake_import_module(name, *args, **kwargs):
        if name == module_name:
            raise missing
        import importlib
        return importlib.import_module(name, *args, **kwargs)

    with patch(
        "molass_legacy.Optimizer.FuncImporter.import_module",
        side_effect=fake_import_module,
    ):
        with pytest.raises(ModuleNotFoundError):
            import_objective_function("G1500")
