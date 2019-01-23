#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import corner
import emcee
import matplotlib.pyplot as plt


#filename = "../Data/Rup147.h5"
filename = "../Data/apRun18.h5"
#filename = "../Data/Rup147LargeNWalk.h5"

# Open file
reader = emcee.backends.HDFBackend(filename)

print(reader.iteration)
tau = reader.get_autocorr_time(tol=0)
if np.any(~np.isfinite(tau)):
    tau = 2500

burnin = int(2*np.max(tau))
thin = int(0.5*np.min(tau))
chain = reader.get_chain(discard=burnin, flat=True, thin=thin)

print(burnin)

### Corner plot ###

labels = ["m1", "m2", "Prot1I", "Prot2I", "Tau1", "Tau2", "PorbI", "EccI",
          "Age"]
#range = [(1.0,1.1),(1.0,1.1),(0,15),(0,15),(-2,3),(-2,3),(3,13),
#         (0,0.6),(1.0e9,4.0e9)]
range = None

# MLE solution
mle = np.array([1.08, 1.07, 7.28, 7.47, -0.17, -0.39, 7.27, 0.28, 2.49])

fig = corner.corner(chain, labels=labels, range=range,
                    quantiles=[0.16, 0.5, 0.84], truths=mle,
                    show_titles=True, title_kwargs={"fontsize": 12})

fig.savefig("rup147Corner.png", bbox_inches="tight")

### Chains

chain = reader.get_chain(discard=burnin, flat=False, thin=thin)

fig, axes = plt.subplots(nrows=9, figsize=(10,60))

for ii, ax in enumerate(axes.flatten()):

    ax.plot(chain[:,:, ii], "k", alpha=0.3, lw=1.5)
    ax.set_ylabel(labels[ii])
    ax.set_xlim(0,len(chain))

axes[-1].set_xlabel("Iterations")

fig.tight_layout()
fig.savefig("rup147Chains.png")
