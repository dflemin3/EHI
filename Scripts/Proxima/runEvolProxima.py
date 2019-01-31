#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Proxima Centauri b runs with initial conditions sampled from posterior dist
"""

import os
import numpy as np
import vplot
import emcee
from ehi import proxima, mcmcUtils

# Define run parameters
nsamples = 100
planetList = ["proximab.in"]

# RNG seed
seed = 90
np.random.seed(seed)

# Options
kwargs = proxima.kwargsPROXIMA
kwargs["LnPrior"] = proxima.LnPriorPROXIMA
kwargs["PriorSample"] = proxima.samplePriorPROXIMA
PATH = os.path.dirname(os.path.abspath(__file__))
kwargs["PATH"] = PATH
kwargs["planetList"] = planetList

# Get the input files, save them as strings
planet_ins = []
for planet in planetList:
    with open(os.path.join(PATH, planet), 'r') as f:
        planet_ins.append(f.read())
    kwargs["PLANETIN"] = planet_ins
with open(os.path.join(PATH, "star.in"), 'r') as f:
    star_in = f.read()
    kwargs["STARIN"] = star_in
with open(os.path.join(PATH, "vpl.in"), 'r') as f:
    vpl_in = f.read()
    kwargs["VPLIN"] = vpl_in

# Check for output dir, make it if it doesn't already exist
if not os.path.exists(os.path.join(PATH, "output")):
    os.makedirs(os.path.join(PATH, "output"))

# Draw nsmaples from the posterior distributions
reader = emcee.backends.HDFBackend("../../Data/proxima.h5")
tau = reader.get_autocorr_time()
burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

# Get random initial conditions from posterior
sampleInds = np.random.choice(np.arange(len(samples)), size=nsamples, replace=False)

# Containers
waterMass = []
oxygenMass = []
lum = []
lumXUV = []
radius = []
temp = []
hzlimrun = []
time = []

ii = 0
while ii < nsamples:

    # Run simulations and collect output
    output = mcmcUtils.GetEvol(samples[sampleInds[ii],:], **kwargs)

    # If simulation succeeded, extract data, move on to the next one
    if output is not None:
        # Extract simulation data
        time.append(output.star.Time)

        waterMass.append(output.proximab.SurfWaterMass)
        oxygenMass.append(output.proximab.OxygenMass + output.proximab.OxygenMantleMass)

        lum.append(output.star.Luminosity)
        lumXUV.append(output.star.LXUVStellar)
        radius.append(output.star.Radius)
        temp.append(output.star.Temperature)
        hzlimrun.append(output.star.HZLimRunaway)

        ii = ii + 1

# Cache results
np.savez("../../Data/proximaEvol.npz", time=time, SurfWaterMass=waterMass,
         OxygenMass=oxygenMass, Luminosity=lum, LXUVStellar=lumXUV,
         Radius=radius, Temperature=temp, HZLimRunaway=hzlimrun)

# Done!
