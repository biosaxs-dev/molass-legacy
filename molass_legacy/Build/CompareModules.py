"""
    CompareModules.py

    Copyright (c) 2023, Masatsuyo Takahashi, KEK-PF
"""
import os
import sys
import re
import subprocess
from packaging import version
import ctypes

# https://stackoverflow.com/questions/1026431/cross-platform-way-to-check-admin-rights-in-a-python-script-under-windows
try:
    is_admin = os.getuid() == 0
except AttributeError:
    is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0

assert is_admin

this_python_exe = sys.executable

this_dir = os.path.dirname(os.path.abspath( __file__ ))
home_dir = os.path.abspath(os.path.join(this_dir, "..\\..\\"))
executables_dir = os.path.join(home_dir, "embeddables")
embed_pathon_exe = os.path.join(executables_dir, "python.exe")

assert os.path.exists(embed_pathon_exe)

requirements_txt = os.path.join(home_dir, "build\\build_embeddables\\requirements.txt")

truncate_re = re.compile(r"\s+#?.*")
version_re = re.compile(r"\bVersion:\s(\S+)\s")

with open(requirements_txt) as fh:
    for line in fh:
        if line[0] == "#":
            continue

        name = re.sub(truncate_re, '', line)
        print("Comparing", name)

        versions = []
        for python_exe in this_python_exe, embed_pathon_exe:
            result = subprocess.run([python_exe, "-m", "pip", "show", name], capture_output=True)
            stdout = result.stdout.decode()
            m = version_re.search(stdout)
            if m:
                versions.append(m.group(1))

        if len(versions) == 2 and versions[0] == versions[1]:
            print("ok", versions[0])
        else:
            print("different", versions)
            # synchronize to the later version
            if version.parse(versions[0]) < version.parse(versions[1]):
                ver = versions[1]
                python_exe = this_python_exe
            else:
                ver = versions[0]
                python_exe = embed_pathon_exe
            versioned_name = "%s==%s" % (name, ver)
            print('"%s" -m pip install %s' % (python_exe, versioned_name))
            result = subprocess.run([python_exe, "-m", "pip", "install", versioned_name], capture_output=True)
            if result.returncode == 0:
                print("ok")
            else:
                stdout = result.stdout.decode()
                print(stdout)
                stderr = result.stderr.decode()
                print(stderr)
