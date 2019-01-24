#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
docs
"""

import vplot as vpl
import numpy as np
import emcee
import subprocess
import re
import os, sys
import random
import argparse
from . import utils
from . import proxima


__all__ = ["FunctionWrapper", "LnLike", "GetEvol", "RunMCMC"]

class FunctionWrapper(object):
    """"
    A simple function wrapper class. Stores :py:obj:`args` and :py:obj:`kwargs` and
    allows an arbitrary function to be called with a single parameter :py:obj:`x`
    """

    def __init__(self, f, *args, **kwargs):
        """
        """

        self.f = f
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x):
        """
        """

        return self.f(x, *self.args, **self.kwargs)
# end class

def LnLike(x, **kwargs):
    """
    loglikelihood function: runs VPLanet simulation!
    """

    # Get the current vector
    dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x
    dSatXUVFrac = 10 ** dSatXUVFrac # Unlog
    dStopTime *= 1.e9 # Convert from Gyr -> yr
    dOutputTime = dStopTime # Output only at the end of the simulation
    dSatXUVTime = 10 ** dSatXUVTime # Unlog

    # Get the prior probability
    lnprior = kwargs["LnPrior"](x, **kwargs)
    if np.isinf(lnprior):
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Get strings containing VPLanet input files (they must be provided!)
    try:
        planet_in = kwargs.get("PLANETIN")
        star_in = kwargs.get("STARIN")
        vpl_in = kwargs.get("VPLIN")
    except KeyError as err:
        print("ERROR: Must supply PLANETIN, STARIN, VPLIN.")
        raise

    # Get PATH
    try:
        PATH = kwargs.get("PATH")
    except KeyError as err:
        print("ERROR: Must supply PATH.")
        raise

    # Randomize file names
    sysName = 'vpl%012x' % random.randrange(16**12)
    planetName = 'pl%012x' % random.randrange(16**12)
    starName = 'st%012x' % random.randrange(16**12)
    sysFile = sysName + '.in'
    planetFile = planetName + '.in'
    starFile = starName + '.in'
    logfile = sysName + '.log'
    planetFwFile = '%s.planet.forward' % sysName
    starFwFile = '%s.star.forward' % sysName

    # Get the planet mass (all prior)
    dPlanetMass = -np.inf
    while -dPlanetMass > 10:
        inc = np.arccos(1 - np.random.random())
        msini = proxima.mpsiniProximab + proxima.mpsiniProximabSig * np.random.randn()
        dPlanetMass = -msini / np.sin(inc)

    # Get planet Porb (all prior)
    dPorbInit = kwargs.get("PORB") + kwargs.get("PORBSIG") * np.random.randn()

    # Populate the planet input file (periods negative to make units Mearth in VPLanet)
    planet_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", -dPlanetMass), planet_in)
    planet_in = re.sub("%s(.*?)#" % "dOrbPeriod", "%s %.6e #" % ("dOrbPeriod", -dPorbInit), planet_in)

    with open(os.path.join(PATH, "output", planetFile), 'w') as f:
        print(planet_in, file = f)

    # Populate the star input file
    star_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVFrac", "%s %.6e #" % ("dSatXUVFrac", dSatXUVFrac), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVTime", "%s %.6e #" % ("dSatXUVTime", -dSatXUVTime), star_in)
    star_in = re.sub("%s(.*?)#" % "dXUVBeta", "%s %.6e #" % ("dXUVBeta", -dXUVBeta), star_in)
    with open(os.path.join(PATH, "output", starFile), 'w') as f:
        print(star_in, file = f)

    # Populate the system input file
    vpl_in = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), vpl_in)
    vpl_in = re.sub('%s(.*?)#' % "dOutputTime", '%s %.6e #' % ("dOutputTime", dOutputTime), vpl_in)
    vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysName, vpl_in)
    vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s %s #' % (starFile, planetFile), vpl_in)
    with open(os.path.join(PATH, "output", sysFile), 'w') as f:
        print(vpl_in, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", sysFile], cwd = os.path.join(PATH, "output"))
    output = vpl.GetOutput(os.path.join(PATH, "output"), logfile = logfile)

    try:
        os.remove(os.path.join(PATH, "output", planetFile))
        os.remove(os.path.join(PATH, "output", starFile))
        os.remove(os.path.join(PATH, "output", sysFile))
        os.remove(os.path.join(PATH, "output", planetFwFile))
        os.remove(os.path.join(PATH, "output", starFwFile))
        os.remove(os.path.join(PATH, "output", logfile))
    except FileNotFoundError:
        # Run failed!
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Ensure we ran for as long as we set out to
    if not output.log.final.system.Age / utils.YEARSEC >= dStopTime:
        return -np.inf, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Get output parameters
    dPorb = dPorbInit # XXX hack since we don't evolve this quantity
    dEnvMass = float(output.log.final.planet.EnvelopeMass)
    dWaterMass = float(output.log.final.planet.SurfWaterMass)
    dOxygenMass = float(output.log.final.planet.OxygenMass) + float(output.log.final.planet.OxygenMantleMass)
    dLum = float(output.log.final.star.Luminosity)
    dLogLumXUV = np.log10(float(output.log.final.star.LXUVStellar)) # Logged!
    dRGTime = float(output.log.final.planet.RGDuration)

    # Extract constraints
    # Must have orbital period, error, for planet
    porb = kwargs.get("PORB")
    porbSig = kwargs.get("PORBSIG")
    try:
        lum = kwargs.get("LUM")
        lumSig = kwargs.get("LUMSIG")
    except KeyError:
        lum = None
        lumSig = None
    try:
        logLumXUV = kwargs.get("LOGLUMXUV")
        logLumXUVSig = kwargs.get("LOGLUMXUVSIG")
    except KeyError:
        logLumXUV = None
        logLumXUVSig = None

    # Compute the likelihood using provided constraints, assuming we have
    # radius constraints for both stars
    #lnlike = ((dPorb - porb) / porbSig) ** 2
    if lum is not None:
        lnlike = ((dLum - lum) / lumSig) ** 2
    if logLumXUV is not None:
        lnlike += ((dLogLumXUV - logLumXUV) / logLumXUVSig) ** 2
    lnlike = -0.5 * lnlike + lnprior

    # Return likelihood and blobs
    return lnlike, dPorb, dPlanetMass, dLum, dLogLumXUV, dRGTime, dEnvMass, dWaterMass, dOxygenMass
# end function


def GetEvol(x, **kwargs):
    """
    Run a VPLanet simulation for this initial condition vector, x
    """

    # Get the current vector
    dMass, dSatXUVFrac, dSatXUVTime, dStopTime, dXUVBeta = x
    dSatXUVFrac = 10 ** dSatXUVFrac # Unlog
    dStopTime *= 1.e9 # Convert from Gyr -> yr
    dOutputTime = dStopTime # Output only at the end of the simulation
    dSatXUVTime = 10 ** dSatXUVTime # Unlog

    # Get the prior probability
    lnprior = kwargs["LnPrior"](x, **kwargs)
    if np.isinf(lnprior):
        return None

    # Get strings containing VPLanet input files (they must be provided!)
    try:
        planet_in = kwargs.get("PLANETIN")
        star_in = kwargs.get("STARIN")
        vpl_in = kwargs.get("VPLIN")
    except KeyError as err:
        print("ERROR: Must supply PLANETIN, STARIN, VPLIN.")
        raise

    # Get PATH
    try:
        PATH = kwargs.get("PATH")
    except KeyError as err:
        print("ERROR: Must supply PATH.")
        raise

    # Randomize file names
    sysName = 'vpl%012x' % random.randrange(16**12)
    planetName = 'pri%012x' % random.randrange(16**12)
    starName = 'sec%012x' % random.randrange(16**12)
    sysFile = sysName + '.in'
    planetFile = planetName + '.in'
    starFile = starName + '.in'
    logfile = sysName + '.log'
    planetFwFile = '%s.planet.forward' % sysName
    starFwFile = '%s.star.forward' % sysName

    # Get the planet mass (all prior)
    dPlanetMass = -np.inf
    while -dPlanetMass > 10:
        inc = np.arccos(1 - np.random.random())
        msini = Mpsini + sigMpsini * np.random.randn()
        dPlanetMass = -msini / np.sin(inc)

    # Get planet Porb (all prior)
    dPorbInit = kwargs.get("PORB") + kwargs.get("PORBSIG") * np.random.randn()

    # Populate the planet input file (periods negative to make units Mearth in VPLanet)
    planet_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", -dPlanetMass), planet_in)
    planet_in = re.sub("%s(.*?)#" % "dOrbPeriod", "%s %.6e #" % ("dOrbPeriod", -dPorbInit), planet_in)

    with open(os.path.join(PATH, "output", planetFile), 'w') as f:
        print(planet_in, file = f)

    # Populate the star input file
    star_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVFrac", "%s %.6e #" % ("dSatXUVFrac", dSatXUVFrac), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVTime", "%s %.6e #" % ("dSatXUVTime", -dSatXUVTime), star_in)
    star_in = re.sub("%s(.*?)#" % "dXUVBeta", "%s %.6e #" % ("dXUVBeta", -dXUVBeta), star_in)
    with open(os.path.join(PATH, "output", starFile), 'w') as f:
        print(star_in, file = f)

    # Populate the system input file
    vpl_in = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), vpl_in)
    vpl_in = re.sub('%s(.*?)#' % "dOutputTime", '%s %.6e #' % ("dOutputTime", dOutputTime), vpl_in)
    vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysName, vpl_in)
    vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s %s #' % (starFile, planetFile), vpl_in)
    with open(os.path.join(PATH, "output", sysFile), 'w') as f:
        print(vpl_in, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", sysFile], cwd = os.path.join(PATH, "output"))
    output = vpl.GetOutput(os.path.join(PATH, "output"), logfile = logfile)

    try:
        os.remove(os.path.join(PATH, "output", planetFile))
        os.remove(os.path.join(PATH, "output", starFile))
        os.remove(os.path.join(PATH, "output", sysFile))
        os.remove(os.path.join(PATH, "output", planetFwFile))
        os.remove(os.path.join(PATH, "output", starFwFile))
        os.remove(os.path.join(PATH, "output", logfile))
    except FileNotFoundError:
        # Run failed!
        return None

    # Ensure we ran for as long as we set out to
    if not output.log.final.system.Age / utils.YEARSEC >= dStopTime:
        return None

    # Return output
    return output
# end function


def RunMCMC(x0=None, ndim=5, nwalk=100, nsteps=5000, pool=None, backend=None,
            restart=False, **kwargs):
    """
    """

    # Ensure LnPrior, prior sample are in kwargs
    try:
        kwargs["PriorSample"]
    except KeyError as err:
        print("ERROR: Must supply PriorSample function!")
        raise
    try:
        kwargs["LnPrior"]
    except KeyError as err:
        print("ERROR: Must supply LnPrior function!")
        raise

    # Extract path
    PATH = kwargs["PATH"]

    print("Running MCMC...")

    # Get the input files, save them as strings
    with open(os.path.join(PATH, "planet.in"), 'r') as f:
        planet_in = f.read()
        kwargs["PLANETIN"] = planet_in
    with open(os.path.join(PATH, "star.in"), 'r') as f:
        star_in = f.read()
        kwargs["STARIN"] = star_in
    with open(os.path.join(PATH, "vpl.in"), 'r') as f:
        vpl_in = f.read()
        kwargs["VPLIN"] = vpl_in

    # Set up backend to save results
    if backend is not None:
        # Set up the backend
        handler = emcee.backends.HDFBackend(backend)

        # If restarting from a previous interation, initialize backend
        if not restart:
            handler.reset(nwalk, ndim)

    # Populate initial conditions for walkers using random samples over prior
    if not restart:
        # If MCMC isn't initialized, just sample from the prior
        if x0 is None:
            x0 = np.array([kwargs["PriorSample"](**kwargs) for w in range(nwalk)])

    ### Run MCMC ###

    # Define blobs, blob data types
    dtype = [("dPorb", np.float64), ("dPlanetMass", np.float64), ("dLum", np.float64),
             ("dLogLXUV", np.float64), ("dRGTime", np.float64), ("dEnvMass", np.float64),
             ("dWaterMass", np.float64), ("dOxygenMass", np.float64)]

    # Initialize the sampler object
    sampler = emcee.EnsembleSampler(nwalk, ndim, LnLike, kwargs=kwargs, pool=pool,
                                    blobs_dtype=dtype, backend=handler)

    # Actually run the MCMC
    if restart:
        sampler.run_mcmc(None, nsteps)
    else:
        for ii, result in enumerate(sampler.sample(x0, iterations=nsteps)):
            print("MCMC: %d/%d..." % (ii + 1, nsteps))

    print("Done!")
# end function
