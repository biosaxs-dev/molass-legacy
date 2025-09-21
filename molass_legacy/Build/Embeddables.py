"""
    Embeddables.py

    Copyright (c) 2019-2024, Masatsuyo Takahashi, KEK-PF
"""
import sys
import os
import copy
import platform
import re
import zipfile
import subprocess
from shutil import copytree, copyfile
from time import sleep
import logging
from KekLib.StdoutLogger import StdoutLogger
from .Github import get_method
# from .SystemPath import add_to_path

REQUIRED_MINIMAL = ['requests', 'bs4', 'wget', 'selenium', 'chromedriver_binary']
EMBEDDABLES_FOLDER = 'embeddables'
TRY_ADDING_TO_PATH = False
SPECIFIED_VERSION = {
    "numpy": "1.26.4",
}

class DummyUnofficialSite:
    def __init__(self):
        pass

    def is_ready_for(self, name):
        return False

class Embeddables:
    def __init__(self, skip_install=False):
        self.logger = StdoutLogger("build.log")
        self.python00 = None
        self.python_version_re = re.compile(r'python-(\d+\.\d+\.\w+)')
        if not skip_install:
            self.install_required_minimum()
        self.emb_folder = EMBEDDABLES_FOLDER
        self.site_folder = os.path.join(self.emb_folder, r'Lib\site-packages')
        self.uo = DummyUnofficialSite()
        self.get_downloadpage_parser('https://www.python.org/downloads/windows/')

    def install_required_minimum(self):
        self.python_exe = sys.executable
        for name in REQUIRED_MINIMAL:
            print('trying to install %s.' % name)
            if self.already_installed(name):
                print('%s is already installed.' % name)
            else:
                if name == 'chromedriver_binary':
                    from .Chrome import get_available_driver_version
                    driver_version = get_available_driver_version()
                    name += '==%s' % driver_version
                args = [name]
                print('pip install', *args)
                ret = self.pip('install', *args)
                if ret.returncode != 0:
                    print(ret.stderr.decode())
                    assert False

    def get_this_ver(self):
        this_ver = 'Python %s' % platform.python_version()
        print('this_ver=', this_ver)
        return this_ver

    def clean(self):
        import shutil
        import glob
        for f in ['embeddables']:
            if os.path.exists(f):
                shutil.rmtree(f)
        for f in glob.glob('*.zip'):
            os.remove(f)

    def build(self, stable=True, requirements="requirements.txt"):
        """
        TODO: separate official site parsing
        """
        this_ver = self.get_this_ver()
        latest_ver = self.get_latest_version()
        if latest_ver == this_ver:
            want_ver = latest_ver
        else:
            print("Latest '%s' != this '%s'" % (latest_ver, this_ver))
            yn = input("Build from the latest version? ([y]/n) > ")
            if yn == "":
                yn = "y"
            print(yn)
            if yn[0] == "y":
                want_ver = latest_ver
            else:
                want_ver = this_ver

        self.build_ver = [int(v) for v in want_ver.split(' ')[-1].split('.')]
        print("build_ver=", self.build_ver)
        pair = self.get_latestrelease_top_pair()
        print('pair=', pair)

        url_pair = []
        for element in pair:
            url = self.get_embeddable_zipfile_url(element, want_ver)
            print('url=', url)
            url_pair.append(url)
            if url is not None:
                break

        # url = url_pair[0 if stable else 1]
        url = url_pair[0]
        file = url.split('/')[-1]

        if not os.path.exists('embeddables'):
            file = self.download(url, file)
            self.unzip(file)

        self.add_standard_dirs_to_path()
        self.prepare_python_exe()
        self.add_pip()
        if TRY_ADDING_TO_PATH:
            this_dir = os.path.dirname(os.path.abspath( __file__ ))
            script_path = os.path.join(os.path.join(this_dir, EMBEDDABLES_FOLDER), 'Scripts')
            print("script_path=", script_path)
            add_to_path(script_path)
        self.get_local_python_folder()
        self.copy_tkinter()
        self.copy_devel_folders()
        # self.prepare_unofficial_site()
        self.install_requirements(requirements)

    def prepare_python_exe(self):
        self.python_exe = os.path.join(self.emb_folder, 'python.exe')
        assert os.path.exists(self.python_exe)

    def get_downloadpage_parser(self, url):
        import requests
        from bs4 import BeautifulSoup
        response = requests.get(url)
        print('response=', response)
        page_html = response.text
        soup = BeautifulSoup(page_html, "html.parser")
        self.parser = soup
        return soup

    def get_latest_version(self):
        for k, h1 in enumerate(self.parser.findAll('h1')):
            # print([k], h1)
            if h1.text.find("Python Releases for Windows") >= 0:
                break

        for k, ul in enumerate(h1.find_all_next('ul')):
            # print([k], ul)
            break

        for k, li in enumerate(ul.find_all_next('li')):
            print([k], li)
            break

        python_version_re = re.compile(r"Latest Python 3 Release - (Python \S+)")
        m = python_version_re.search(li.text)
        if m:
            ret_ver = m.group(1)
        else:
            ret_ver = None
        return ret_ver

    def get_latestrelease_top_pair(self):
        pair = []
        for k, element in enumerate(self.parser.findAll('h2')):
            # print([k], element)
            pair.append(element)
        return pair

    def get_want_ver_title_element(self, top_element, want_ver):
        title_element = None
        for k, element in enumerate(top_element.find_all_next('a')):
            # print([k], element)
            if element.text.find(want_ver) >= 0 or k > 30:
                title_element = element
                break
        return title_element

    def get_embeddable_zipfile_url(self, top_element, want_ver):
        title_element = self.get_want_ver_title_element(top_element, want_ver)
        print('title_element=', title_element)

        version_num = re.sub(r"Python\s+(\S+)\s*", lambda m : m.group(1), want_ver)
        print("version_num='%s'" % version_num)

        if platform.architecture()[0].find('64') >= 0:
            key_name = 'embeddable package (64-bit)'
            old_key_name = 'x86-64 embeddable zip'
        else:
            key_name = 'embeddable package (32-bit)'
            old_key_name = 'x86 embeddable zip'

        url = None
        for name in [key_name, old_key_name]:
            for k, element in enumerate(title_element.find_all_next('a')):
                # print([k], element)
                if element.text.find(name) > 0:
                    url = element['href']
                    # print([k], element)
                    print([k], url)
                    if url.find(version_num) < 0:
                        continue

                    if self.python00 is None:
                        m = self.python_version_re.search(url)
                        if m:
                            v = m.group(1).split('.')
                            self.python00 = 'python' + v[0] + v[1]
                            print('python00=', self.python00)
                    break
            if url is None:
                continue
            else:
                break
        return url

    def download(self, url, out_file=None):
        import wget

        if out_file is None:
            filename = url.split('/')[-1]
            out_file = filename
        wget.download(url, out=out_file)
        return out_file

    def unzip(self, zipfile_):
        """
        see https://stackoverflow.com/questions/9813243/extract-files-from-zip-file-and-retain-mod-date-python-2-7-1-on-windows-7
        and improve this into a time-attributes-retaining one
        """
        with zipfile.ZipFile(zipfile_) as zh:
            zh.extractall(self.emb_folder)
        return self.emb_folder

    def add_standard_dirs_to_path(self):
        pth_file = self.emb_folder + '/' + self.python00 + '._pth'
        lines = []
        already_added = False
        with open(pth_file) as fh:
            for line in fh.readlines():
                lines.append(line)
                if line.find('import') == 0:
                    already_added = True
                    break
        if already_added:
            print('it seems that standard dirs have been already added.')
            return

        lines.insert(2, '.\\DLLs\n')
        lines.insert(2, '.\\Lib\n')
        lines.insert(2, '.\\Scripts\n')
        lines[-1] = lines[-1].replace('#', '')
        print(lines)
        new_pth_file = pth_file.replace('_pth', '_pth.tmp')
        with open(new_pth_file, 'w') as fh:
            fh.write(''.join(lines))

        bak_pth_file = pth_file.replace('_pth', '_pth.bak')
        os.rename(pth_file, bak_pth_file)
        os.rename(new_pth_file, pth_file)

    def python(self, *args):
        result = subprocess.run(
                [self.python_exe] + list(args),
                capture_output=True,
                )
        return result

    def pip(self, *args):
        result = self.python('-m', 'pip', *args)
        return result

    def add_pip(self):
        result = self.pip('show', 'pip')
        installed = result.returncode == 0
        if installed:
            print('it seems that pip has been installed.')
            return

        print('we must install pip.')
        file = self.download('https://bootstrap.pypa.io/get-pip.py')
        print("\nexecuting 'python %s'" % file)
        result = self.python(file)
        os.remove(file)

    def get_local_python_folder(self):
        self.local_python_folder, _ = os.path.split(sys.executable)
        print('local python is installed in "%s"' % self.local_python_folder)

    def copy_folders(self, folders):
        for folder in folders:
            src_folder = os.path.join(self.local_python_folder, folder)
            tgt_folder = os.path.join(self.emb_folder, folder)
            if os.path.exists(tgt_folder):
                print(tgt_folder, 'exists.')
            else:
                print('copying to', tgt_folder)
                copytree(src_folder, tgt_folder)

    def copy_tkinter(self):
        self.copy_folders(['tcl', 'Lib/tkinter', 'Lib/idlelib'])

        dll_files = ['_tkinter.pyd', 'tcl86t.dll', 'tk86t.dll']
        if self.build_ver >= [3,12,0]:
            dll_files += ['zlib1.dll']      # zlib1.dll is required from Python312
        src_dll_folder = os.path.join(self.local_python_folder, 'DLLs')
        tgt_dll_folder = os.path.join(self.emb_folder, 'DLLs')
        if not os.path.exists(tgt_dll_folder):
            os.makedirs(tgt_dll_folder)

        for file in dll_files:
            src_file = os.path.join(src_dll_folder, file)
            tgt_file = os.path.join(tgt_dll_folder, file)
            if os.path.exists(tgt_file):
                print(tgt_file, 'exists.')
            else:
                print('copying to', tgt_file)
                copyfile(src_file, tgt_file)

    def copy_devel_folders(self):
        self.copy_folders(['include', 'libs'])

    def already_installed(self, name):
        print('pip show', name)
        ret = self.pip('show', name)
        return ret.returncode == 0

    def install_requirements(self, requirements):
        """
        TODO:   check version compatibility other than patch numbers.
                e.g.,
                    3.1.2 ==> 3.1.3 ok
                    3.1.2 ==> 3.2.0 not ok, or be careful at least
        """
        truncate_re = re.compile(r'\s+#?.*')
        spec_re = re.compile(r'spec=(\S+)')
        with open(requirements) as fh:
            for line in fh.readlines():
                if line[0] == '#':
                    continue
                name = re.sub(truncate_re, '', line)
                print('trying to install %s.' % name)
                if self.already_installed(name):
                    print('%s is already installed.' % name)
                else:
                    method = get_method(name)
                    if method is None:
                        if self.uo.is_ready_for(name):
                            m = spec_re.search(line)
                            if m:
                                spec = m.group(1)
                            else:
                                spec = None
                            arg_name = self.uo.download(name, spec=spec)
                        else:
                            arg_name = name
                        
                        version = SPECIFIED_VERSION.get(arg_name)
                        if version is not None:
                            arg_name += "==%s" % version
                        print('pip install "%s"' % arg_name)
                        ret = self.pip('install', arg_name)
                        if ret.returncode != 0:
                            print(ret.stderr.decode())
                            assert False
                    else:
                        method()
                if self.uo.is_ready_for(name):
                    extra_step = self.uo.get_extra_step(name)
                    if extra_step is not None:
                        extra_step(self)

    def prepare_unofficial_site(self):
        from .UnofficialSite import UnofficialSite
        NN = self.python00.replace('python', '')
        self.uo = UnofficialSite(NN)
