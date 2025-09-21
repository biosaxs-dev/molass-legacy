# coding: utf-8
"""
    SakuraUtils.py

    Copyright (c) 2021, Masatsuyo Takahashi, KEK-PF
"""
import os
import shutil

def clean_all_skrolds():
    this_dir, _ = os.path.split(os.path.abspath(__file__))
    root_dir = os.path.abspath(this_dir + '/../..')
    print(root_dir)

    for root, dirs, files in os.walk(root_dir):
        # print("root=", root)
        for file in files:
            if file.find('.skrold') > 0:
                path = os.path.join(root, file)
                print("removing", path)
                os.remove(path)
