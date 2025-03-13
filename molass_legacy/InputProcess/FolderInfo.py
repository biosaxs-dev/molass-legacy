# coding: utf-8
"""
    FolderInfo.py

    Copyright (c) 2019-2020, SAXS Team, KEK-PF
"""
import os
import re

filename_re = re.compile(r'^(\w+_)(\d+)([^\d]+)')

class FolderInfo:
    def __init__(self, files, conc_file, comments):
        self.conc_file = conc_file
        self.num_files = len(files)
        self.comments = comments

        self.folder, sample_name = os.path.split(files[0])

        m = filename_re.search(sample_name)
        if m:
            # pattern will be like SAMPLE_#####_sub.dat
            self.pattern = m.group(1) + '#'*len(m.group(2)) + m.group(3)
            start = int(m.group(2))
            stop = start + self.num_files
        else:
            self.pattern = 'Irregular Name:' + sample_name
            start = 0
            stop = self.num_files

        self.num_slice = slice(start, stop)

    def get_files(self):
        files = []
        for k in range(self.num_slice.start, self.num_slice.stop):
            name = self.pattern.replace('#####', '%05d' % k)
            files.append(os.path.join(self.folder, name))
        return files

    def get_comments(self):
        return self.comments
