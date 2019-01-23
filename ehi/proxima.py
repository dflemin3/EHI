#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data and priors for Rup 147 stellar binary with data from Torres et al. (2018)
"""

import numpy as np
from scipy.stats import norm
from . import utils

__all__ = ["kwargsPROXIMA", "LnPriorPROXIMA", "samplePriorPROXIMA"]

# Observational constraints adopted from Barnes et al. (2016)
# see here https://github.com/rodluger/proxima/blob/master/uncert/uncert.py

# Stellar properties
mProxima = 0.120                 # Anglada-Escude et al. (2016) [Msun]
mProximaSig = 0.015              # Anglada-Escude et al. (2016) [Msun]
lumProxima = 0.00165             # Demory et al. (2009) [Lsun]
lumProximaSig = 0.00015          # Demory et al. (2009) [Lsun]
logLXUVProxima = -6.36           # Ribas et al. (2016; see above)
logLXUVProximaSig = 0.3          # See above
ageProxima = 4.8                 # Barnes et al. (2017) [Gyr]
ageProximaSig = 1.4              # ibid [Gyr]
betaProxima = -1.23              # Ribas et al. (2005)
betaProximaSig = 0.1             # Arbitrary

# Planet properties
porbProximab = 11.186            # Anglada-Escude et al. (2016) [d]
porbProximabSig = 0.002          # Anglada-Escude et al. (2016) [d]
mpsiniProximab = 1.27            # Anglada-Escude et al. (2016) [Mearth]
mpsiniProximabSig = 0.18         # Anglada-Escude et al. (2016) [Mearth]


### Prior, likelihood, MCMC functions ###


def LnPriorPROXIMA(x, **kwargs):
    """
    log prior
    """

    # Get the current vector
    dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x

    # Generous bounds for stellar mass [Msun]
    if (dMass < 0.1) or (dMass > 0.15):
        return -np.inf

    # log flat prior on saturation fraction (log10)
    if (dSatXUVFrac < -5) or (dSatXUVFrac > -2):
        return -np.inf

    # log flat prior on saturation timescale log10[Gyr]
    if (-dSatXUVTime < -0.3) or (-dSatXUVTime > 1.0):
        return -np.inf

    # Large bound for age of system [Gyr]
    if (dStopTime < 1.0e-3) or (dStopTime > 8):
        return -np.inf

    # Age prior
    lnprior = norm.logpdf(dStopTime, ageProxima, ageProximaSig)

    # Beta prior
    lnprior += norm.logpdf(dXUVBeta, betaProxima, betaProximaSig)

    return lnprior
# end function


def samplePriorPROXIMA(size=1, **kwargs):
    """
    Sample dMass, dSatXUVFrac, dSatXUVTime, dStopTime, and dXUVBeta from their
    prior distributions.
    """

    ret = []
    for ii in range(size):
        while True:
            guess = [np.random.uniform(low=0.1, high=0.15),
                     np.random.uniform(low=-5.0, high=-2.0),
                     np.random.uniform(low=-0.3, high=1.0),
                     norm.rvs(loc=ageProxima, scale=ageProximaSig, size=1)[0],
                     norm.rvs(loc=betaProxima, scale=betaProximaSig, size=1)[0]]
            if not np.isinf(LnPriorPROXIMA(guess, **kwargs)):
                ret.append(guess)
                break

    if size > 1:
        return ret
    else:
        return ret[0]
# end function

# Dict to hold all constraints
kwargsPROXIMA = {"PATH" : ".",
                 "LUM" : lumProxima,
                 "LUMSIG" : lumProxima,
                 "LnPrior" : LnPriorPROXIMA,
                 "LOGLUMXUV" : logLXUVProxima,
                 "LOGLUMXUVSIG" : logLXUVProximaSig,
                 "PORB" : porbProximab,
                 "PORBSIG" : porbProximabSig,
                 "PriorSample" : samplePriorPROXIMA}
