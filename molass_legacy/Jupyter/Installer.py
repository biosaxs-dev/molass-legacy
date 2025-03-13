"""
    Jupyter.Installer.py

    Copyright (c) 2022-2024, SAXS Team, KEK-PF
"""

from molass_legacy.KekLib.RunPython import run_this_python

PACKAGES = ["jupyterlab", "ipympl"]
MAX_ITER = 80

def install_jupyterlab(parent=None):
    jupyterlab = PACKAGES[0]
    result = run_this_python("-m", "pip", "show", jupyterlab)
    if False:
        print("returncode=", result.returncode)
        print(result.stdout)
        print(result.stderr)
    if result.returncode == 0:
        return True

    import os
    import sys
    import psutil
    from molass_legacy.KekLib.TkCustomWidgets import MessageBox
    site_packages_path = os.path.join(os.path.dirname(sys.executable), "lib\\site-packages")

    hdd = psutil.disk_usage('/')

    GB = 2**30
    print ("Total: %d GiB" % (hdd.total/GB))
    print ("Used: %d GiB" % (hdd.used/GB))
    print ("Free: %d GiB" % (hdd.free/GB))

    yn = MessageBox.askyesno("Jupyter Lab Install Confirmation",
        "%s package in not installed in the following folder.\n"
        "    %s\n\n"
        "It may require up to 200MB (=0.2GB) depending on the state of package installation, "
        "while the current disc usage seems like\n\n"
        "    Total: %5.0f GB\n"
        "    Used : %5.0f GB\n"
        "    Free : %5.0f GB.\n\n"
        "Would you like to accept and procced?"
        % (jupyterlab, site_packages_path, hdd.total/GB, hdd.used/GB, hdd.free/GB),
        parent=parent)

    if not yn:
        return False

    if parent is None:
        for package in PACKAGES:
            print("installing", package)
            result = run_this_python("-m", "pip", "install", package)
            print("returncode=", result.returncode)
    else:
        from molass_legacy.KekLib.RunPython import run_this_python_stdout_lines
        from molass_legacy.KekLib.ProgressMinDialog import run_with_progress

        def install_jupyterlab(exe_queue):
            n = 0
            for i, package in enumerate(PACKAGES):
                print("installing", package)
                for j, line in enumerate(run_this_python_stdout_lines("-m", "pip", "install", package)):
                    print([j], line)
                    n += 1
                    exe_queue.put([0, n])
            exe_queue.put([1, n])
            return True

        def on_return(ret):
            print("--------------- on_return: ret=", ret)

        try:
            ret, error_info = run_with_progress(parent, install_jupyterlab, max_iter=MAX_ITER,
                                        title="Jupyter.Installer", on_return=on_return)
        except:
            from molass_legacy.KekLib.ExceptionTracebacker import log_exception
            log_exception(None, "run_with_progress failure: ")
            ret = None

    return True
