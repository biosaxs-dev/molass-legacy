"""
    Selective.Proportions.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
import numpy as np

PROPORTIONS_FUNC_SPEC = [
    [("def proportions(p):\n"
     "    return p, 1-p"), (0.05, 0.2)],
    [("def proportions(p):\n"
     "    return p, 1-2*p, p"), (0.025, 0.15)],
    [("def proportions(p):\n"
     "    return p, 0.5-p, 0.5-p, p"), (0.025, 0.1)],
    [("def proportions(p):\n"
     "    return 2*p, 0.3-p, 0.4-2*p, 0.3-p, 2*p"), (0.01, 0.05)],
    [("def proportions(p):\n"
     "    return 2*p, 0.25-p, 0.25-p, 0.25-p, 0.25-p, 2*p"), (0.01, 0.05)],
    [("def proportions(p):\n"
     "    return 3*p, 0.15-p, 0.15-p, 0.4-2*p, 0.15-p, 0.15-p, 3*p"), (0.007, 0.04)],
]

class Proportions_:         # renamed so as not to be confused with "proportions" function
    def __init__(self):
        pass

    def generate_funccode(self, num_components):
        return PROPORTIONS_FUNC_SPEC[num_components-2][0]

    def get_min_max_values(self, n):
        """
        Returns a tuple of (min, max) values for the proportions function parameter p.
        """
        return PROPORTIONS_FUNC_SPEC[n-2][1]

    def guess_optimal_prop(self, n):
        minv, maxv = self.get_min_max_values(n)
        return minv*0.7 + maxv*0.3

    def get_optimal_props_text(self, n):
        return " proportions(%.3g)" % self.guess_optimal_prop(n)

    def get_with_props_values(self, n):
        # the eval below must be preceeded by the "proportions" function definition,
        # which can be attained by calling self.get_proportions_func(...)
        opt_props_text = self.get_optimal_props_text(n)
        return [opt_props_text, str(eval(opt_props_text))]
    
    def get_proportions_func(self, code_text, debug=False):
        if debug:
            print(code_text)
        context = {}
        exec(code_text, context)
        # note this code_text must define a function named "proportions"
        proportions = context['proportions']
        globals()['proportions'] = proportions      # "proportions" will be called in the eval above
        if debug:
            for p in np.linspace(0.05, 0.2, 20):
                print(proportions(p))
        return proportions
