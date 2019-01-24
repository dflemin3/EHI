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
blobs = reader.get_blobs(discard=burnin, flat=True, thin=thin)
print(blobs[0], blobs.shape)

samples = np.concatenate((chain, [blob for blob in blobs]), axis=1)


### Corner plot ###
#labels = ["Mass", "SatXUVFrac", "SatXUVTime", "Age", "XUVBeta"]
#range = [(1.0,1.1),(1.0,1.1),(0,15),(0,15),(-2,3),(-2,3),(3,13),
#         (0,0.6),(1.0e9,4.0e9)]
labels = None
range = None

# MLE solution

fig = corner.corner(blobs, labels=labels, range=range,
                    quantiles=[0.16, 0.5, 0.84],
                    show_titles=True, title_kwargs={"fontsize": 12})

fig.savefig("proximaCorner.png", bbox_inches="tight")

### Chains

chain = reader.get_chain(discard=burnin, flat=False, thin=thin)

fig, axes = plt.subplots(nrows=9, figsize=(10,60))

for ii, ax in enumerate(axes.flatten()):

    ax.plot(chain[:,:, ii], "k", alpha=0.3, lw=1.5)
    ax.set_ylabel(labels[ii])
    ax.set_xlim(0,len(chain))

axes[-1].set_xlabel("Iterations")

fig.tight_layout()
fig.savefig("proximaChains.png")
