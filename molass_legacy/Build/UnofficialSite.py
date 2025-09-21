"""
    UnofficialSite.py

    Copyright (c) 2020-2023, Masatsuyo Takahashi, KEK-PF
"""
import sys
import os
import platform
import re
import requests
import getpass
from time import sleep
from bs4 import BeautifulSoup
import glob
from shutil import copy
"""
learned at
1) https://qiita.com/Azunyan1111/items/b161b998790b1db2ff7a
2) https://qiita.com/memakura/items/20a02161fa7e18d8a693

    ChromeDriverをpipでバージョン指定してインストールする手順
        https://qiita.com/ten_to_ten/items/d0b39f7d1da1d2627d1b

    pip install chromedriver-binary==91.0.4472.101
"""
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import chromedriver_binary
from KnownPaths import get_downloads_folder

PAGE_URL = 'https://www.lfd.uci.edu/~gohlke/pythonlibs/'
"""
# MODULES_FROM_UO_SITE_WITH_LIST = ['pywin32', 'numpy', 'scipy', 'matplotlib', 'cupy',
MODULES_FROM_UO_SITE_WITH_LIST = ['pywin32', 'scipy', 'matplotlib',
                'pillow', 'kiwisolver', 'pandas', 'statsmodels', 'scikit-learn', 'fabio', 'lxml',
                'opencv']
# temporarily remove 'numba'
# temporarily remove 'llvmlite' to get llvmlite 0.34 for numba-0.51.2-cp39-cp39-win_amd64.whl

MODULES_FROM_UO_SITE_WITHOUT_LIST = ['future', 'asteval', 'et_xmlfile']
MODULES_FROM_UO_SITE = MODULES_FROM_UO_SITE_WITH_LIST + MODULES_FROM_UO_SITE_WITHOUT_LIST
"""
MODULES_FROM_UO_SITE = []

"""
crape data with beautifulsoup results in 404
https://stackoverflow.com/questions/41909065/scrape-data-with-beautifulsoup-results-in-404?rq=1
"""
HEADERS = {"User-Agent":"Mozilla/5.0"}

def get_downloadpage_parser(url):
    response = requests.get(url, headers=HEADERS)
    print('response=', response)
    page_html = response.text
    soup = BeautifulSoup(page_html, "html.parser")
    return soup

def get_downloadpage_driver(url):
    options = Options()
    # options.add_argument('--headless')        # --headless disturbes click download
    try:
        driver = webdriver.Chrome(options=options)
        driver.get(url)
    except:
        print(r"run as admin build\build_embeddables\tests\test_110_ChromeDriver.py")
        assert False
    # page_html = driver.page_source.encode('utf-8')
    return driver

class UnofficialSite:
    def __init__(self, NN):
        self.parser = get_downloadpage_parser(PAGE_URL)
        self.driver = get_downloadpage_driver(PAGE_URL)
        username = getpass.getuser()
        folder = get_downloads_folder().replace('Default', username)
        # note that the above replace does not take place unless is includes 'Default'

        print('folder=', folder)
        if not os.path.exists(folder):
            folder = folder.replace(username, 'Owner')
            print('folder=', folder)
            assert os.path.exists(folder)

        self.user_downloads_folder = folder

        self.rc_re = re.compile(r'rc\d*-')

        self.dl_a_element_dict = {}

        for name in MODULES_FROM_UO_SITE:
            if name == 'et_xmlfile':
                name_ = 'et-xmlfile'
            else:
                name_ = name
            a_element = self.parser.find(id=name_)
            print([name], a_element)
            self.dl_a_element_dict[name] = a_element

        if platform.architecture()[0].find('64') >= 0:
            self.uo_postfix = '‑cp{0}‑cp{0}‑win_amd64.whl'.format(NN)
        else:
            self.uo_postfix = '‑cp{0}‑cp{0}‑win32.whl'.format(NN)

    def is_ready_for(self, name):
        return self.dl_a_element_dict.get(name) is not None

    def get_a_element_text(self, name, spec=None):
        text = None
        start_a_element = self.dl_a_element_dict.get(name)

        if name in MODULES_FROM_UO_SITE_WITH_LIST:
            # find next ul element
            for j, ul_element in enumerate(start_a_element.find_all_next('ul')):
                # first-found ul_element
                break

            # find whl files
            for j, a_element in enumerate(ul_element.find_all_next('a')):
                text  = a_element.text
                print([j], text)
                if spec is not None:
                    if text.find(spec) < 0:
                        continue

                if text.find(self.uo_postfix) >= 0:
                    if self.rc_re.search(text.replace('‑','-')):
                        # skip release candidates.
                        # be careful not to be confused with hyphen-like chars.
                        print([j], 'skipping a release candidate.')
                        continue
                    else:
                        break
                if j > 20:
                    break
        else:
            for j, a_element in enumerate(start_a_element.find_all_next('a')):
                text  = a_element.text
                print([j], text)
                if text.find(name) >= 0 and text.find('py3'):
                    break
                if j > 20:
                    break

        print('get_a_element_text', text)
        return text

    def download(self, name, devel=False, spec=None):
        page_text = self.get_a_element_text(name, spec=spec)
        name_text = page_text.replace('‑', '-')

        file_path = os.path.join(self.user_downloads_folder, name_text)
        print('file_path=', file_path)
        if os.path.exists(file_path):
            print('there exists %s.' % file_path)
        else:
            # a_element = self.driver.find_element_by_link_text(page_text)
            # https://stackoverflow.com/questions/72754331/webdriver-object-has-no-attribute-find-element-by-link-text-selenium-scrip
            a_element = self.driver.find_element("link text", page_text)
            print(a_element)
            a_element.click()   # does not seem to work
            print("click done.")
            self.wait_for_download_complete(file_path)

        return file_path

    def wait_for_download_complete(self, file_path):
        dl_temp_file = file_path + '.crdownload'
        while (not os.path.exists(file_path) or os.path.exists(dl_temp_file)):
            sys.stdout.write('.')
            sys.stdout.flush()
            sleep(1)
        print('download complete in %s.' % file_path)

    def get_extra_step(self, name):
        if name == 'pywin32':
            extra_step = pywin32_extra_step
        else:
            extra_step = None

        return extra_step

def pywin32_extra_step(caller):
    target_folder = caller.emb_folder
    pywin32_dll_folder = os.path.join(caller.site_folder, 'pywin32_system32')
    for path in glob.glob(pywin32_dll_folder + r'\*.dll'):
        _, file = os.path.split(path)
        target_path = os.path.join(target_folder, file)
        if os.path.exists(target_path):
            print('%s exists, ok.' % file)
        else:
            print('copying %s.' % file)
            copy(path, target_folder)
        assert os.path.exists(target_path)
