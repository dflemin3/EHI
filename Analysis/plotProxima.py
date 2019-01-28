#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import corner
import emcee
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt


#filename = "../Data/proxima.h5"
filename = "apRun9.h5"

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
    ehi = np.array(blobs[:,6] > 0)

    # Estimate error on EHI
    ehiEst = np.mean(ehi)
    ehiErrDown, ehiErrUp = proportion_confint(np.sum(ehi), len(ehi), alpha=0.32, method="agresti_coull")
    print("EHI +/-: %e %e/%e" % (ehiEst, ehiErrUp-ehiEst, ehiEst-ehiErrDown))

    mask = np.array([0, 1, 2, 3, 4, 6, 7])
    samples = np.concatenate((chain, blobs[:,mask], ehi[:, None]), axis=1)

    labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta", "dPorb",
              "dPlanetMass", "dLum", "dLogLumXUV", "dRGTime", "dWaterMass",
              "dOxygenMass", "EHI"]

    # Convert RG Time to Myr
    samples[:,9] = samples[:,9]/1.0e6

    # Make luminosity units more palatable
    samples[:,7] = samples[:,7]*1.0e3
else:
    samples = chain
    labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta"]

#samples = np.load("apFModelCache.npz")["theta"]
#print(samples.shape)

fig = corner.corner(samples, quantiles=[0.16, 0.5, 0.84], labels=labels,
                    show_titles=True, title_kwargs={"fontsize": 12})

fig.savefig("proximaAPCorner.png", bbox_inches="tight")

# Done!
