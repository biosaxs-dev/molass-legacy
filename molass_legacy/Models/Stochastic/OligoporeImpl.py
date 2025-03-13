"""
    Models.Stochastic.OligoporeImpl.py

    Copyright (c) 2024, SAXS Team, KEK-PF
"""
from SecTheory.SecCF import sec_oligopore_phi_impl, gec_oligopore_phi_impl
from SecTheory.SecPDF import FftInvPdf

USE_DISPERSIVE = False

sec_oligopore_pdf = FftInvPdf(sec_oligopore_phi_impl)
gec_oligopore_pdf = FftInvPdf(gec_oligopore_phi_impl)

def oligopore_pdf(x, nt_pairs, pszp, x0, N0=None):
    if N0 is None:
        return gec_oligopore_pdf(x - x0, nt_pairs, pszp, 0)     # using x0 directly may induce divergence
    else:
        return sec_oligopore_pdf(x - x0, nt_pairs, pszp, 0, N0) # using x0 directly may induce divergence