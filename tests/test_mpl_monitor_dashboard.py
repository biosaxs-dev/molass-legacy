"""
Phase 2 regression test for MplMonitor.create_dashboard() button visibility.

Verifies that the Resume and Terminate buttons are hidden for in-process
(_RunInfoSource) sources and shown for subprocess (_SubprocessSource) sources.

See: https://github.com/biosaxs-dev/molass-library/issues/139
"""
import threading
import pytest
from unittest.mock import MagicMock

ipywidgets = pytest.importorskip("ipywidgets", reason="ipywidgets not installed")


def _make_mon_with_source(source):
    """Create a bare MplMonitor bypassing __init__ (needs filesystem + SerialSettings)."""
    from molass_legacy.Optimizer.MplMonitor import MplMonitor
    mon = object.__new__(MplMonitor)
    mon.source = source
    mon.terminate_event = threading.Event()
    return mon


def _button_descriptions(mon):
    """Return the 'description' of every widget in controls.children that has one."""
    return [w.description for w in mon.controls.children if hasattr(w, "description")]


# ---------------------------------------------------------------------------
# In-process source (_RunInfoSource) — Resume + Terminate must be absent
# ---------------------------------------------------------------------------

def test_in_process_dashboard_hides_resume_button():
    """Resume Job must not appear in controls for a _RunInfoSource."""
    from molass_legacy.Optimizer.MplMonitor import _RunInfoSource
    ri = MagicMock()
    ri.is_alive = False
    ri.work_folder = "/tmp/fake_run"

    mon = _make_mon_with_source(_RunInfoSource(ri))
    mon.create_dashboard()

    assert "Resume Job" not in _button_descriptions(mon)


def test_in_process_dashboard_hides_terminate_button():
    """Terminate Job must not appear in controls for a _RunInfoSource."""
    from molass_legacy.Optimizer.MplMonitor import _RunInfoSource
    ri = MagicMock()
    ri.is_alive = False
    ri.work_folder = "/tmp/fake_run"

    mon = _make_mon_with_source(_RunInfoSource(ri))
    mon.create_dashboard()

    assert "Terminate Job" not in _button_descriptions(mon)


def test_in_process_dashboard_shows_export_button():
    """Export Data must still appear in controls for a _RunInfoSource."""
    from molass_legacy.Optimizer.MplMonitor import _RunInfoSource
    ri = MagicMock()
    ri.is_alive = False
    ri.work_folder = "/tmp/fake_run"

    mon = _make_mon_with_source(_RunInfoSource(ri))
    mon.create_dashboard()

    assert "Export Data" in _button_descriptions(mon)


# ---------------------------------------------------------------------------
# Subprocess source (_SubprocessSource) — Resume + Terminate must be present
# ---------------------------------------------------------------------------

def test_subprocess_dashboard_shows_resume_button():
    """Resume Job must appear in controls for a _SubprocessSource."""
    from molass_legacy.Optimizer.MplMonitor import _SubprocessSource
    runner = MagicMock()
    runner.poll.return_value = None  # alive

    mon = _make_mon_with_source(_SubprocessSource(runner))
    mon.create_dashboard()

    assert "Resume Job" in _button_descriptions(mon)


def test_subprocess_dashboard_shows_terminate_button():
    """Terminate Job must appear in controls for a _SubprocessSource."""
    from molass_legacy.Optimizer.MplMonitor import _SubprocessSource
    runner = MagicMock()
    runner.poll.return_value = None

    mon = _make_mon_with_source(_SubprocessSource(runner))
    mon.create_dashboard()

    assert "Terminate Job" in _button_descriptions(mon)


def test_subprocess_dashboard_shows_export_button():
    """Export Data must appear in controls for a _SubprocessSource."""
    from molass_legacy.Optimizer.MplMonitor import _SubprocessSource
    runner = MagicMock()
    runner.poll.return_value = None

    mon = _make_mon_with_source(_SubprocessSource(runner))
    mon.create_dashboard()

    assert "Export Data" in _button_descriptions(mon)


# ---------------------------------------------------------------------------
# Widget attributes exist regardless of source type
# ---------------------------------------------------------------------------

def test_create_dashboard_always_sets_resume_button_attribute():
    """self.resume_button must be set even when not shown, so callers don't break."""
    from molass_legacy.Optimizer.MplMonitor import _RunInfoSource
    ri = MagicMock()
    ri.is_alive = False
    ri.work_folder = "/tmp/fake_run"

    mon = _make_mon_with_source(_RunInfoSource(ri))
    mon.create_dashboard()

    assert hasattr(mon, "resume_button")
    assert hasattr(mon, "terminate_button")
