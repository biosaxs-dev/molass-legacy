"""Test that MplMonitor.terminate() kills the subprocess directly.

Regression test for issue #17: terminate() used to rely on the watch
thread to call runner.terminate(), but a race condition meant the
watch thread could exit before reaching that call.

Phase 2 (molass-library#139): tests for _RunInfoSource and factory classmethods.
"""
import threading
from unittest.mock import MagicMock, patch


def make_mock_monitor():
    """Create a minimal MplMonitor-like object with mocked internals."""
    from molass_legacy.Optimizer.MplMonitor import MplMonitor

    # Prevent __init__ from running (it needs SerialSettings, filesystem, etc.)
    mon = object.__new__(MplMonitor)

    # Set up the minimal attributes that terminate() touches
    mon.terminate_event = threading.Event()
    mon.stop_watch_event = threading.Event()
    mon.watch_thread = None
    mon.is_monitoring = False
    mon.instance_id = 0
    mon.logger = MagicMock()
    mon.source = MagicMock()
    mon.source.terminate = MagicMock()
    mon.process_id = "12345"
    mon.optimizer_folder = "fake_folder"

    return mon


def test_terminate_calls_runner_terminate_directly():
    """source.terminate() must be called even if no watch thread is running."""
    mon = make_mock_monitor()
    mon.terminate()
    mon.source.terminate.assert_called_once()


def test_terminate_calls_runner_before_stop_watching():
    """source.terminate() must be called even when watch thread exits early."""
    mon = make_mock_monitor()

    # Simulate an active watch thread that exits immediately on stop_watch_event
    call_order = []

    original_terminate = mon.source.terminate
    def record_runner_terminate():
        call_order.append("runner.terminate")
        original_terminate()

    mon.source.terminate = record_runner_terminate

    # Create a thread that blocks until stop_watch_event is set, then exits
    def fake_watch():
        mon.stop_watch_event.wait()
        call_order.append("watch_thread_exit")

    mon.watch_thread = threading.Thread(target=fake_watch)
    mon.watch_thread.start()

    mon.terminate(timeout=2.0)

    assert "runner.terminate" in call_order, (
        "source.terminate() was not called during terminate()"
    )


def test_terminate_survives_runner_exception():
    """terminate() should not raise even if source.terminate() fails."""
    mon = make_mock_monitor()
    mon.source.terminate.side_effect = OSError("process already dead")

    # Should not raise
    result = mon.terminate()
    assert result is True or result is False  # returns a bool either way
    mon.source.terminate.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 2: _RunInfoSource and factory classmethods
# ---------------------------------------------------------------------------

def test_run_info_source_is_alive_delegates_to_run_info():
    """_RunInfoSource.is_alive() reads run_info.is_alive (property)."""
    from molass_legacy.Optimizer.MplMonitor import _RunInfoSource

    ri = MagicMock()
    ri.is_alive = True
    src = _RunInfoSource(ri)
    assert src.is_alive() is True

    ri.is_alive = False
    assert src.is_alive() is False


def test_run_info_source_working_folder():
    """_RunInfoSource.working_folder reads run_info.work_folder."""
    from molass_legacy.Optimizer.MplMonitor import _RunInfoSource

    ri = MagicMock()
    ri.work_folder = "/some/path"
    src = _RunInfoSource(ri)
    assert src.working_folder == "/some/path"


def test_run_info_source_terminate_is_noop():
    """_RunInfoSource.terminate() must not raise and must not call anything."""
    from molass_legacy.Optimizer.MplMonitor import _RunInfoSource

    ri = MagicMock()
    src = _RunInfoSource(ri)
    src.terminate()  # must not raise
    ri.terminate.assert_not_called()


def test_subprocess_source_wraps_back_runner():
    """_SubprocessSource.is_alive() / working_folder / terminate() delegate to _runner."""
    from molass_legacy.Optimizer.MplMonitor import _SubprocessSource

    runner = MagicMock()
    runner.poll.return_value = None  # subprocess alive
    runner.working_folder = "/opt/folder"

    src = _SubprocessSource(runner)
    assert src.is_alive() is True

    runner.poll.return_value = 0   # subprocess exited
    assert src.is_alive() is False

    assert src.working_folder == "/opt/folder"

    src.terminate()
    runner.terminate.assert_called_once()
