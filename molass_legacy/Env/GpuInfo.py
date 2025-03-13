"""
    GpuInfo.py

    Copyright (c) 2019-2024, SAXS Team, KEK-PF
"""
import sys
import os
import subprocess
import re

class DxdiagInfo:
    def __init__(self):
        import os
        import tempfile
        import subprocess

        self.info_list = []
        path = tempfile.gettempdir()
        # print(path)
        file = path + '/dxdiag.txt'
        ret = subprocess.run(['dxdiag', '/t', file])
        if ret.returncode == 0:
            with open(file) as fh:
                for line in fh.readlines():
                    # print(line)
                    kw = 'Card name:'
                    n = line.find(kw)
                    if n > 0:
                        n += len(kw) + 1
                        self.cardname = line[n:-1]
                        # print(self.cardname)
                        self.info_list.append(line[:-1])
                        break
        else:
            assert False

        os.remove(file)

    def get_info_list(self):
        return self.info_list

def get_cupy_version():
    CUDA_PATH = os.environ.get('CUDA_PATH')
    if CUDA_PATH is None:
        cupy_cuda = 'cupy-cuda11x'
    else:
        cuda_ver = CUDA_PATH.split('\\')[-1]
        # print('cuda_ver=', cuda_ver)
        # cuda_ver is like v11.8, v12.3, etc.
        cupy_cuda = 'cupy-cuda%sx' % cuda_ver[1:3]
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'show', cupy_cuda], capture_output=True)
        assert result.returncode == 0
        stdout = result.stdout.decode()
        version_re = re.compile(r'Version: (\S+)\r\n')
        m = version_re.search(stdout)
        if m:
            version = (cupy_cuda, m.group(1))
        else:
            version = None
    except:
        from molass_legacy.KekLib.ExceptionTracebacker import log_exception
        log_exception(None, "get_cupy_version falure: ")
        version = None
    # print("get_cupy_version: version=", version)
    return version

def get_nvcc_version():
    try:
        result = subprocess.run(['nvcc', '-V'], capture_output=True)
        stdout = result.stdout.decode()
        # print('get_nvcc_version: stdout=', stdout)
        version_re = re.compile('(V\d+\.\d+)\.\d+')
        m = version_re.search(stdout)
        if m:
            version = m.group(1)
        else:
            version = version
    except:
        version = None
    # print('get_nvcc_version: version=', version)
    return version

def get_available_cuda_versions():
    try:
        result = subprocess.run(['where', 'nvcc'], capture_output=True)
        stdout = result.stdout.decode()
    except:
        stdout = ""

    versions = []
    v11x = []
    for line in stdout.split("\n"):
        nodes = line.split("\\")
        if len(nodes) > 2:
            versions.append(nodes[-3])

    return versions + v11x

class GpuInfo:
    def __init__(self):
        self.get_graphics_card_info()
        self.cuda_path = os.environ.get('CUDA_PATH')
        self.cupy_version = get_cupy_version()
        self.nvcc_version = get_nvcc_version()
        self.cuda_versions = get_available_cuda_versions()
        self.cuda_version = self.cuda_versions[0] if len(self.cuda_versions) > 0 else None
        self.cuda_is_availale = self.cuda_version is not None
        self._cupy_ok = None
        try:
            import cupy
            self.cupy_is_installed = True
        except:
            self.cupy_is_installed = False

    def get_graphics_card_info(self):
        from pythoncom import CoInitialize, CoUninitialize
        CoInitialize()      # required inside a thread
        import wmi
        wmi_infos = wmi.WMI().Win32_VideoController()

        # Name = "NVIDIA GeForce GTX 960";
        vp_re = re.compile(r'\s+Name = "(.+)"')
        self.card_infos = []

        exists_compatible = False
        self.index_compatible = None
        for k, i in enumerate(wmi_infos):
            info = str(i)
            m = vp_re.search(info)
            if m:
                name = m.group(1)
            else:
                name = None

            if info.find('NVIDIA') > 0:
                compatible = True
                exists_compatible = True
                if self.index_compatible is None:
                    # set the first-found
                    self.index_compatible = k
            else:
                compatible = False

            self.card_infos.append((name, compatible))

        self.nvidia_compatible = exists_compatible
        k = 0 if self.index_compatible is None else self.index_compatible
        self.card_name = self.card_infos[k][0]
        CoUninitialize()    # required inside a thread

    def check_cupy_cuda_consistency(self):
        cupy_ver_num = self.cupy_version[1].replace('cuda', '')
        non_digit_re = re.compile(r'\D+')
        cuda_ver_num = re.sub(non_digit_re, '', self.nvcc_version)
        # print('cupy_ver_num=', cupy_ver_num, 'cuda_ver_num', cuda_ver_num)
        return cupy_ver_num[0:2] == cuda_ver_num[0:2]   # "12x"[0:2] == "123"[0:2]

    def cuda_ok(self):
        return self.nvidia_compatible and self.cuda_is_availale

    def cupy_ok(self):
        if self._cupy_ok is not None:
            return self._cupy_ok

        if self.cuda_ok():
            if not self.cupy_is_installed:
                self.cupy_ng_reason = "However, no consistent cupy is installed."
                self._cupy_ok = False
                return False
        else:
            self.cupy_ng_reason = "However, cuda is not available."
            self._cupy_ok = False
            return False

        ok = False
        if self.cupy_version is None:
            cupy_cuda_version = "v11.x"
        else:
            cupy_cuda_version = re.sub(r"cupy-cuda(\d{2})(\w)",  lambda m: "v%s.%s" % (m.group(1), m.group(2)), self.cupy_version[0])
            # print("cupy_cuda_version=", cupy_cuda_version)
            for k, v in enumerate(self.cuda_versions):
                # print([k], v)
                if cupy_cuda_version[0:3] == v[0:3]:    # eg., "12.x"[0:3] == "12.3"[0:3]
                    ok = True
                    break
        if ok:
            self.cupy_ng_reason = None
        else:
            self.cupy_ng_reason = "However, cupy cuda version %s is not installed" % cupy_cuda_version

        self._cupy_ok = ok
        return ok

    def get_reason_texts(self):
        texts = []

        # GPU Card compatibility
        not_ = '' if self.nvidia_compatible else 'not '
        texts.append(self.card_name + ' is %sNVIDIA compatible.' % not_)
        if not self.nvidia_compatible:
            # unless it is nvidia_compatible, there is no need to add further information.
            return texts

        # CUDA installation
        conn_text = 'And, ' if self.cuda_is_availale == self.nvidia_compatible else 'However, '
        if self.cuda_version is None:
            if self.nvcc_version is None:
                cuda_text = 'CUDA is not found installed.'
            else:
                versions = (self.cupy_version, self.nvcc_version)
                if self.check_cupy_cuda_consistency():
                    cuda_text = 'Something is wrong with Cupy %s and CUDA %s.' % versions
                else:
                    cuda_text = 'Cupy %s is inconsistent with CUDA %s.' % versions
        else:
            be = "is" if len(self.cuda_versions) == 1 else "are"
            cuda_text = 'CUDA %s %s installed.' % (",".join(self.cuda_versions), be)
        texts.append(conn_text + cuda_text)

        # CUPY avaiability
        if not self.cupy_ok():
            texts.append(self.cupy_ng_reason)

        return texts
