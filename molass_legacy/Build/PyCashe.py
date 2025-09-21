# coding: utf-8
"""
    PyCashe.py

    Copyright (c) 2020, Masatsuyo Takahashi, KEK-PF
"""
import os
import shutil

def clean_all_pycashes():
    this_dir, _ = os.path.split(os.path.abspath(__file__))
    root_dir = os.path.abspath(this_dir + '/..')
    print(root_dir)

    for root, dirs, files in os.walk(root_dir):
        # print("root=", root)
        for dir_ in dirs:
            # print("\tdir_=", dir_)
            if dir_ == "__pycache__":
                path = os.path.join(root, dir_)
                print("removing", path)
                shutil.rmtree(path)
