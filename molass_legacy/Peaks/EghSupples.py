"""

    Peaks.EghSupples.py

    Copyright (c) 2021-2022, SAXS Team, KEK-PF

"""
import numpy as np
from molass_legacy.Peaks.ElutionModels import egh

"""

See also, 2001, Kevin Lan, James W. Jorgenson

from sympy import symbols
from sympy.solvers import solve
s, t, a, A, B = symbols("s, t, a, A, B")
solve((s**2 + B*A/(2*a), t + (B-A)/a), A, B)
[(a*t/2 - sqrt(-a*(-a*t**2 + 8*s**2))/2,
  -a*t/2 - sqrt(-a*(-a*t**2 + 8*s**2))/2),
 (a*t/2 + sqrt(-a*(-a*t**2 + 8*s**2))/2,
  -a*t/2 + sqrt(-a*(-a*t**2 + 8*s**2))/2)]

"""
def compute_AB(a, s, t):
    # a: np.log(alpha)
    A = a*t/2 + np.sqrt(-a*(-a*t**2 + 8*s**2))/2
    B = -a*t/2 + np.sqrt(-a*(-a*t**2 + 8*s**2))/2
    return A, B

def d_egh(x, H, tR, sigma, tau):
    x_  = x - tR
    s2  = 2 * sigma**2
    z   = s2 + tau*x_
    z_neg   = z <= 0
    z_pos   = z > 0

    zero_part = np.zeros( len(x) )[z_neg]

    # t*(x - m)**2/(2*s**2 + t*(x - m))**2 - 2*(x - m)/(2*s**2 + t*(x - m))
    xp = x_[z_pos]
    denom = s2 + tau*xp 
    posi_part = (tau*xp **2/denom**2 - 2*xp/denom) * egh(x[z_pos], H, tR, sigma, tau)

    if tau > 0:
        parts = [ zero_part, posi_part ]
    else:
        parts = [ posi_part, zero_part ]

    return np.hstack( parts )
