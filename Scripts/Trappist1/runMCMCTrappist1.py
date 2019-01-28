#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trappist1 b-h MCMC run
"""

import os
from ehi import pool
from ehi import trappist1, mcmcUtils

# Define run parameters
ndim = 5
nwalk = 40
nsteps = 10
nsamples = 0
restart = False
backend = "trappist1.h5"
planetList = ["trappist1b.in", "trappist1c.in", "trappist1d.in", "trappist1e.in",
              "trappist1f.in", "trappist1g.in", "trappist1h.in"]

# Open a pool, and let it rip!
with pool.Pool(pool='SerialPool') as pool:

    # Options
    kwargs = trappist1.kwargsTRAPPIST1
    kwargs["nsteps"] = nsteps
    kwargs["nsamples"] = nsamples
    kwargs["nwalk"] = nwalk
    kwargs["pool"] = pool
    kwargs["restart"] = restart
    kwargs["LnPrior"] = trappist1.LnPriorTRAPPIST1
    kwargs["PriorSample"] = trappist1.samplePriorTRAPPIST1
    PATH = os.path.dirname(os.path.abspath(__file__))
    kwargs["PATH"] = PATH
    kwargs["backend"] = backend
    kwargs["planetList"] = planetList

    # Check for output dir, make it if it doesn't already exist
    if not os.path.exists(os.path.join(PATH, "output")):
        os.makedirs(os.path.join(PATH, "output"))

    # Run
    mcmcUtils.RunMCMC(**kwargs)

# Done!
