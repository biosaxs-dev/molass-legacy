"""
    Sympy/StochasticUtils.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""

from sympy import symbols, exp, diff, simplify, I

w = symbols('w')

def raw_moment(cf, k):
    return simplify((-I)**k * diff(cf, w, k).subs(dict(w=0)))

def central_moment(cf, k):
    m = raw_moment(cf, 1)
    return simplify(raw_moment(cf/exp(I*w*m),k))

def demo_monopore_cf():
    n1, t1 = symbols('n1, t1')
    cf1 = exp(n1*(1/(1 - I*w*t1) - 1))
    print("raw_moment(cf1, 1)=", raw_moment(cf1, 1))
    print("central_moment(cf1, 2)=", central_moment(cf1, 2))
    print("central_moment(cf2, 3)=", central_moment(cf1, 3))

def demo_dipore_cf():
    n1, t1, n2, t2 = symbols('n1, t1, n2, t2')
    cf2 = exp(n1*(1/(1 - I*w*t1) - 1)) * exp(n2*(1/(1 - I*w*t2) - 1))
    print("raw_moment(cf2, 1)=", raw_moment(cf2, 1))
    print("central_moment(cf2, 2)=", central_moment(cf2, 2))
    print("central_moment(cf2, 3)=", central_moment(cf2, 3))

if __name__ == "__main__":
    # demo_monopore_cf()
    demo_dipore_cf()