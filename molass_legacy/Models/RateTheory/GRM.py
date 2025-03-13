"""
    Models/RateTheory/GRM.py

    Copyright (c) 2023, SAXS Team, KEK-PF





"""
import numpy as np

class GRM:
    def __init__(self, u, a, e, ep, D_L. Dp, Ds, Rp):
        self.u = u
        self.D_L = D_L
        self.F = (1 - e)/e
        self.Deff = ep*Dp + (1- ep)*Ds*a
        self.Rp = Rp
        self.a = a
        self.a_ = ep + (1 - ep)*a
        self.Bp = kext * Rp / self.Deff

    def alpha(self, s):
        return self.a_*s/self.Deff

    def f(self, s):
        ras = np.sqrt(self.alpha(s))
        return self.Bp /( self.Bp - 1 + ras*self.Rp / np.tanh(ras*self.Rp) )    # np.coth(...) == 1/np.tanh(...)

    def phi(self, s):
        return s + self.kext * 3/self.Rp * self.F * (1 - self.f(s))

    def c_(self, s, z):
        U = self.u / (2*self.D_L)
        V = np.sqrt(1 + 4*self.D_L*self.phi(s))
        lam1 = U*(1 - V)
        lam2 = U*(1 + V)
        return self.A * np.exp(lam1*z) + self.B * np.exp(lam2*z)

    def c_p(s, z, r):
        ras = np.sqrt(self.alpha(s))
        # return 1/r * ( self.d1(s,z)/ras * np.sinh(ras*r) + self.d2(s,z) * np.cosh(ras*r) )
        return 1/r * ( self.d1(s,z)/ras * np.sinh(ras*r)    # d2 == 0

    def d1(self, s, z, ras):
        ras = np.sqrt(self.alpha(s))
        nume = self.Bp * senl.c_(s, z) / np.sinh(ras*self.Rp)
        deno = (self.Bp - 1)/(ras*self.Rp) + 1 / np.tanh(ras*self.Rp
        return  nume/deno
