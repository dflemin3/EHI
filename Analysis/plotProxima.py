#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import corner
import emcee
import matplotlib.pyplot as plt


filename = "../Data/proxima.h5"

# Open file
reader = emcee.backends.HDFBackend(filename)

print(reader.iteration)
tau = reader.get_autocorr_time(tol=0)
if np.any(~np.isfinite(tau)):
    tau = 500

burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
chain = reader.get_chain(discard=burnin, flat=True, thin=thin)
tmp = reader.get_blobs(discard=burnin, flat=True, thin=thin)
blobs = []
for bl in tmp:
    blobs.append([bl[ii] for ii in range(len(bl))])
blobs = np.array(blobs)
mask = np.array([0, 1, 2, 3, 4, 6, 7])
samples = np.concatenate((chain, blobs[:,mask]), axis=1)
### Corner plot ###
labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta", "dPorb",
          "dPlanetMass", "dLum", "dLogLumXUV", "dRGTime", "dWaterMass",
          "dOxygenMass"]

# Convert RG Time to Myr
samples[:,9] = samples[:,9]/1.0e6

# Make luminosity units more palatable
samples[:,7] = samples[:,7]*1.0e3

# MLE solution

fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 12})

fig.savefig("proximaCorner.png", bbox_inches="tight")

# Done!
