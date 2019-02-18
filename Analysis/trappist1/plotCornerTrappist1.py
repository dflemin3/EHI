#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import corner
import emcee
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt


#filename = "../../Data/trappist120EOEpsBolmont.h5"
filename = "../../Data/apRun3.h5"

plotBlobs = False

# Open file
reader = emcee.backends.HDFBackend(filename)

print(reader.iteration)
tau = reader.get_autocorr_time(tol=0)
if np.any(~np.isfinite(tau)):
    tau = 500

burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))

print("Burnin, thin:", burnin, thin)

chain = reader.get_chain(discard=burnin, flat=True, thin=thin)
if plotBlobs:
    tmp = reader.get_blobs(discard=burnin, flat=True, thin=thin)
    blobs = []
    for bl in tmp:
        blobs.append([bl[ii] for ii in range(len(bl))])
    blobs = np.array(blobs)
    print(blobs.shape)

    inds = np.arange(len(blobs[0]))
    mask = np.array(list(inds[:2]) + list(inds[15:22]) + list(inds[28:]))
    print(mask)
    xxx
    samples = np.concatenate((chain, blobs[:,mask]), axis=1)
    #blobs = np.array(dPorbs + dPlanetMasses + dRGTimes + dEnvMasses + dWaterMasses + dOxygenMasses)

    labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta", "Lum",
              "logLumXUV", "dRGTimeb", "dRGTimec", "dRGTimed", "dRGTimee",
              "dRGTimef", "dRGTimeg", "dRGTimeh", "WaterMassb", "WaterMassc",
              "WaterMassd", "WaterMasse", "WaterMassf", "WaterMassg", "WaterMassh",
              "OxygenMassb", "OxygenMassc", "OxygenMassd", "OxygenMasse", "OxygenMassf",
              "OxygenMassg", "OxygenMassh"]

    # Convert RG Time to Myr
    samples[:,7] = samples[:,9]/1.0e6
    samples[:,8] = samples[:,9]/1.0e6
    samples[:,9] = samples[:,9]/1.0e6
    samples[:,10] = samples[:,9]/1.0e6
    samples[:,11] = samples[:,9]/1.0e6
    samples[:,12] = samples[:,9]/1.0e6
    samples[:,13] = samples[:,9]/1.0e6

    # Make luminosity units more palatable
    samples[:,5] = samples[:,5]*1.0e3
else:
    samples = chain
    labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta"]

#samples = np.load("apFModelCache.npz")["theta"]
#print(samples.shape)

fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 12})

fig.savefig("trappist1CornerAP.png", bbox_inches="tight")

# Done!
