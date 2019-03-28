#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimate TRAPPIST-1's dereddened B-V color given 2MASS magnitudes

Script output:

TRAPPIST-1 (B-V)_0: 1.5170599999999983

That is, TRAPPIST-1's B-V is about 1.52, which makes sense given it's a very
low mass late M dwarf, nearly a brown drawf.

"""

import numpy as np


def bvFromJHK(J, H, Ks):
    """
    Compute (B-V)_0 color from J, H, and Ks bands using fit from Bilir+2008
    for [M/H] > -0.4 since TRAPPIST-1 has a nearly solar metallicity.
    """

    # Constants
    alpha = 1.64
    beta = 1.033
    gamma = 0.05

    # Compute colors
    JH = J - H
    HKs = H - Ks

    return alpha * JH + beta * HKs + gamma
# end function


# Compute B-V for TRAPPIST-1 Using 2MASS magnitudes given in Burgasser+2017
# originally from Skrutskie et al. 2006
J = 11.35
H = 10.72
Ks = 10.3

BV = bvFromJHK(J, H, Ks)
print("TRAPPIST-1 (B-V)_0:", BV)

# Done!
