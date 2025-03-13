"""
    OptJobInfo.py

    Copyright (c) 2022, SAXS Team, KEK-PF
"""
import os
from .OptLogFile import OptLogFile

JOB_STATES = ["to be run", "running", "finished", "dead"]

class OptJobInfo:
    def __init__(self, name=None, state=None, nc=None, drift_type=None, niter=None, seed=None, pid=None):
        self.name = name
        self.state = state
        self.nc = nc
        self.drift_type = drift_type
        self.niter = niter
        self.seed = seed
        self.pid = pid
        self.items  = [name, state, nc, drift_type, niter, seed]    # for backward compatibility

    def __repr__(self):
        return 'OptimizerJob' + str(self.items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, item):
        return self.items[item]

    def update_state(self, state):
        self.state = state

    def load(self, folder):
        folder_name = os.path.split(folder)[1]
        if self.name is None:
            self.name = folder_name
        else:
            assert folder_name == self.name

        path = os.path.join(folder, "optimizer.log")
        log_file = OptLogFile(path)
        optdict = log_file.optdict

        self.state = "done"
        self.nc = int(optdict['-n'])
        self.drift_type = optdict['-d']
        self.niter = int(optdict['-m'])
        self.seed = int(optdict['-s'])
        self.pid = None
        self.items  = [self.name, self.state, self.nc, self.drift_type, self.niter, self.seed]
        # self.items will be reprecated
