#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Proxima Centauri b MCMC run
"""

import os
from ehi import pool
from ehi import proxima, mcmcUtils

# Define run parameters
ndim = 5
nwalk = 40
nsteps = 10
nsamples = 0
restart = False
npzCache = "proxima.npz"
planetList = ["proximab.in"]

# Open a pool, and let it rip!
with pool.Pool(pool='SerialPool') as pool:

    # Options
    kwargs = proxima.kwargsPROXIMA
    kwargs["nsteps"] = nsteps
    kwargs["nsamples"] = nsamples
    kwargs["nwalk"] = nwalk
    kwargs["pool"] = pool
    kwargs["restart"] = restart
    kwargs["LnPrior"] = proxima.LnPriorPROXIMA
    kwargs["PriorSample"] = proxima.samplePriorPROXIMA
    PATH = os.path.dirname(os.path.abspath(__file__))
    kwargs["PATH"] = PATH
    kwargs["npzCache"] = npzCache
    kwargs["planetList"] = planetList

    # Check for output dir, make it if it doesn't already exist
    if not os.path.exists(os.path.join(PATH, "output")):
        os.makedirs(os.path.join(PATH, "output"))

    # Run
    mcmcUtils.RunMCMC(**kwargs)

# Done!
