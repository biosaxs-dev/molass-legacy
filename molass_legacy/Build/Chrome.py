# coding: utf-8
"""
    Chrome.py

    Copyright (c) 2021-2022, Masatsuyo Takahashi, KEK-PF
"""
import os
import re
from .EasyScraper import EasyScraper

def get_installed_chrome_version():
    """
    cf. https://stackoverflow.com/questions/50880917/how-to-get-chrome-version-using-command-prompt-in-windows
    """
    dir_name_re = re.compile(r"^([\.\d]+)$")
    version = None
    for path in [   r"C:\Program Files\Google\Chrome\Application",
                    r"C:\Program Files (x86)\Google\Chrome\Application"]:
        if not os.path.exists(path):
            continue
        for line in os.listdir(path):
            m = dir_name_re.match(line)
            if m:
                version = m.group(1)
                break
    return version

def get_available_driver_version(chorme_version=None):
    if chorme_version is None:
        chorme_version = get_installed_chrome_version()

    url = "https://pypi.org/project/chromedriver-binary/#history"
    key_text = '.'.join(chorme_version.split('.')[0:3])
    es = EasyScraper(url)
    text = es.get_element('a', key_text)
    version_re = re.compile("(%s)" % (key_text.replace('.', '\\.') + '\\S+'))
    m = version_re.search(text)
    return m.group(1)
