"""
    Jupyter.JupyterLab.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""
import re

def run_jupyterlab(parent=None, try_install=True):
    if try_install:
        from .Installer import install_jupyterlab
        ok = install_jupyterlab(parent)
    else:
        ok = True

    if not ok:
       return

    import sys
    import os
    import subprocess

    this_dir = os.path.dirname(__file__)
    home_dir = os.path.abspath(this_dir + "/../../")
    notebooks_path_txt = os.path.join(home_dir, "notebooks-path.txt")

    with open(notebooks_path_txt) as fh:
        notebooks_path = re.sub(r"\s+", "", fh.read())
    # print("notebooks_path='%s'" % notebooks_path)
    if not os.path.exists(notebooks_path):
        from molass_legacy.KekLib.TkCustomWidgets import MessageBox
        # HookableMessageBox would be required if this must be really the custom widget.
        MessageBox.showinfo("Notebooks path notification",
            'The folder path "%s",\n'
            'which is written in "%s",\n'
            "does not exist.\n"
            "This folder will be created soon after this confirmation\n"
            "and will be used by Jupyter as the current folder.\n\n"
            "Please rewrite it and retry if you wish to change it."
            % (notebooks_path, notebooks_path_txt),
            parent=parent
            )
        os.makedirs(notebooks_path)

    os.chdir(notebooks_path)

    jupyterlab_exe = os.path.join(os.path.dirname(sys.executable), "Scripts\\jupyter-lab.exe")
    # note that "jupyter.exe lab" invocation insead of "jupyter-lab.exe"
    # would fail if the folder has been moved from the created location.

    # quotes are required for paths like "C:\Program Files\Python310\Scripts\jupyter-lab.exe".
    jupyterlab_exe_quoted = '"%s"' % jupyterlab_exe
    print("jupyterlab_exe='%s'" % jupyterlab_exe)
    os.spawnl(os.P_DETACH, jupyterlab_exe, jupyterlab_exe_quoted)   # run in background
