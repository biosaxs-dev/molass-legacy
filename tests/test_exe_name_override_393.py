import pytest
from molass_legacy.KekLib.BasicUtils import exe_name


def test_exe_name_defaults_to_the_script_name(monkeypatch):
    monkeypatch.delenv("MOLASS_SETTINGS_NAME", raising=False)
    monkeypatch.setattr("sys.argv", ["C:/work/run_56f.py"])
    assert exe_name() == "run_56f"


def test_exe_name_can_be_chosen_by_environment_variable(monkeypatch):
    monkeypatch.setattr("sys.argv", ["C:/work/run_56f.py"])
    monkeypatch.setenv("MOLASS_SETTINGS_NAME", "run_56f_job2")
    assert exe_name() == "run_56f_job2"
    monkeypatch.setenv("MOLASS_SETTINGS_NAME", "a/b")
    with pytest.raises(ValueError):
        exe_name()
