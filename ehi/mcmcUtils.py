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
        return -np.inf, np.nan, np.nan, [np.nan for _ in kwargs["PLANETLIST"]], \
        [np.nan for _ in kwargs["PLANETLIST"]], [np.nan for _ in kwargs["PLANETLIST"]], \
        [np.nan for _ in kwargs["PLANETLIST"]], [np.nan for _ in kwargs["PLANETLIST"]], \
        [np.nan for _ in kwargs["PLANETLIST"]]

    # Get strings containing VPLanet input files (they must be provided!)
    try:
        planet_ins = kwargs.get("PLANETIN")
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
    planetNames = ['pl%012x' % random.randrange(16**12) for _ in kwargs["PLANETLIST"]]
    starName = 'st%012x' % random.randrange(16**12)
    sysFile = sysName + '.in'
    planetFiles = [pname + '.in' for pname in planetNames]
    starFile = starName + '.in'
    logfile = sysName + '.log'
    planetFwFiles = ['%s.%s.forward' % (pname, sysName) for name in kwargs["PLANETLIST"]]
    starFwFile = '%s.star.forward' % sysName

    # Get masses, initial eccentricities, Porbs in order from inner -> outer
    planetMasses = [kwargs["PlanetMassSample"](name) for name in kwargs["PLANETLIST"]]
    planetRadii = [kwargs["PlanetRadiusSample"](name) for name in kwargs["PLANETLIST"]]
    planetEccs = [kwargs["PlanetEccSample"](name) for name in kwargs["PLANETLIST"]]
    planetPorbs = [kwargs["PlanetPorbSample"](name) for name in kwargs["PLANETLIST"]]

    # Populate the planet input files for each planet.  Note that Porbs negative
    # to make units days in VPLanet, and same for mass/rad but for Earth units
    for ii, planet_in in enumerate(planet_ins):
        planet_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", -planetMasses[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dRadius", "%s %.6e #" % ("dRadius", -planetRadii[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dEcc", "%s %.6e #" % ("dEcc", planetEccs[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dOrbPeriod", "%s %.6e #" % ("dOrbPeriod", -planetPorbs[ii]), planet_in)
        with open(os.path.join(PATH, "output", planetFiles[ii]), 'w') as f:
            print(planet_in, file = f)

    # Populate the star input file
    star_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVFrac", "%s %.6e #" % ("dSatXUVFrac", dSatXUVFrac), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVTime", "%s %.6e #" % ("dSatXUVTime", -dSatXUVTime), star_in)
    star_in = re.sub("%s(.*?)#" % "dXUVBeta", "%s %.6e #" % ("dXUVBeta", -dXUVBeta), star_in)
    with open(os.path.join(PATH, "output", starFile), 'w') as f:
        print(star_in, file = f)

    # Populate the system input file

    # Populate list of planets
    saBodyFiles = str(starFile)
    for pFile in planetFiles:
        saBodyFiles += str(pFile) + " "
    saBodyFiles = saBodyFiles.strip()

    vpl_in = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), vpl_in)
    vpl_in = re.sub('%s(.*?)#' % "dOutputTime", '%s %.6e #' % ("dOutputTime", dOutputTime), vpl_in)
    vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysName, vpl_in)
    vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s #' % saBodyFiles, vpl_in)
    with open(os.path.join(PATH, "output", sysFile), 'w') as f:
        print(vpl_in, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", sysFile], cwd = os.path.join(PATH, "output"))
    output = vpl.GetOutput(os.path.join(PATH, "output"), logfile = logfile)

    try:
        for pFile in planetFiles:
            os.remove(os.path.join(PATH, "output", pFile))
        os.remove(os.path.join(PATH, "output", starFile))
        os.remove(os.path.join(PATH, "output", sysFile))
        for pFile in planetFwFiles:
            os.remove(os.path.join(PATH, "output", pFile))
        os.remove(os.path.join(PATH, "output", starFwFile))
        os.remove(os.path.join(PATH, "output", logfile))
    except FileNotFoundError:
        # Run failed!
        return -np.inf, np.nan, np.nan, [np.nan for _ in kwargs["PLANETLIST"]], \
        [np.nan for _ in kwargs["PLANETLIST"]], [np.nan for _ in kwargs["PLANETLIST"]], \
        [np.nan for _ in kwargs["PLANETLIST"]], [np.nan for _ in kwargs["PLANETLIST"]], \
        [np.nan for _ in kwargs["PLANETLIST"]]

    # Ensure we ran for as long as we set out to
    if not output.log.final.system.Age / utils.YEARSEC >= dStopTime:
        return -np.inf, np.nan, np.nan, [np.nan for _ in kwargs["PLANETLIST"]], \
        [np.nan for _ in kwargs["PLANETLIST"]], [np.nan for _ in kwargs["PLANETLIST"]], \
        [np.nan for _ in kwargs["PLANETLIST"]], [np.nan for _ in kwargs["PLANETLIST"]], \
        [np.nan for _ in kwargs["PLANETLIST"]]

    # Get planet output parameters. Porb and masses are determined by priors
    dEnvMasses = []
    dWaterMasses = []
    dOxygenMasses = []
    dPorbs = []
    dPlanetMasses = []
    dRGTimes = []
    for ii, pName in enumerate(kwargs["PLANETLIST"]):
        dPlanetMasses.append(planetMasses[ii]) # Prior
        dPorbs.append(planetPorbs[ii]) # Prior
        dEnvMasses.append(float(output.log.final.__dict__[pName].EnvelopeMass))
        dWaterMasses.append(float(output.log.final.__dict__[pName].SurfWaterMass))
        dOxygenMasses.append(float(output.log.final.__dict__[pName].OxygenMass) + float(output.log.final.__dict__[pName].OxygenMantleMass))
        dRGTimes.append(float(output.log.final.__dict__[pName].RGDuration))

    # Get stellar properties
    dLum = float(output.log.final.star.Luminosity)
    dLogLumXUV = np.log10(float(output.log.final.star.LXUVStellar)) # Logged!

    # Extract constraints
    # Must have luminosity, err for star
    lum = kwargs.get("LUM")
    lumSig = kwargs.get("LUMSIG")
    try:
        logLumXUV = kwargs.get("LOGLUMXUV")
        logLumXUVSig = kwargs.get("LOGLUMXUVSIG")
    except KeyError:
        logLumXUV = None
        logLumXUVSig = None

    # Compute the likelihood using provided constraints, assuming we have
    # luminosity constraints for host star
    lnlike = ((dLum - lum) / lumSig) ** 2
    if logLumXUV is not None:
        lnlike += ((dLogLumXUV - logLumXUV) / logLumXUVSig) ** 2
    lnlike = -0.5 * lnlike + lnprior

    dtype [("dLum", np.float64), ("dLogLXUV", np.float64), ("dPorbs", list),
           ("dPlanetMasses", list), ("dRGTimes", list), ("dEnvMasses", list),
             ("dWaterMasses", list), ("dOxygenMasses", list)]

    # Return likelihood and blobs
    return lnlike, dLum, dLogLumXUV, dPorbs, dPlanetMasses, dRGTimes, dEnvMasses, dWaterMasses, dOxygenMasses

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
        planet_ins = kwargs.get("PLANETIN")
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
    planetNames = ['pl%012x' % random.randrange(16**12) for _ in kwargs["PLANETLIST"]]
    starName = 'st%012x' % random.randrange(16**12)
    sysFile = sysName + '.in'
    planetFiles = [planetName + '.in' for planetName in planetNames]
    starFile = starName + '.in'
    logfile = sysName + '.log'
    planetFwFiles = ['%s.%s.forward' % (sysName, planetName) for planetName in planetNames]
    starFwFile = '%s.star.forward' % sysName

    # Sample masses, radii, ecc, porbs for each planet in the system
    planetMasses = [kwargs["PlanetMassSample"](name) for name in kwargs["PLANETLIST"]]
    planetRadii = [kwargs["PlanetRadiusSample"](name) for name in kwargs["PLANETLIST"]]
    planetEccs = [kwargs["PlanetEccSample"](name) for name in kwargs["PLANETLIST"]]
    planetPorbs = [kwargs["PlanetPorbSample"](name) for name in kwargs["PLANETLIST"]]

    # Write input file for each planet
    for ii, planet_in in enumerate(planet_ins):
        # Populate the planet input file (periods negative to make units Mearth in VPLanet)
        planet_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", -planetMasses[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dRadius", "%s %.6e #" % ("dRadius", -planetRadii[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dEcc", "%s %.6e #" % ("dEcc", planetEccs[ii]), planet_in)
        planet_in = re.sub("%s(.*?)#" % "dOrbPeriod", "%s %.6e #" % ("dOrbPeriod", -planetPorbs[ii]), planet_in)
        with open(os.path.join(PATH, "output", planetFiles[ii]), 'w') as f:
            print(planet_in, file = f)

    # Populate the star input file
    star_in = re.sub("%s(.*?)#" % "dMass", "%s %.6e #" % ("dMass", dMass), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVFrac", "%s %.6e #" % ("dSatXUVFrac", dSatXUVFrac), star_in)
    star_in = re.sub("%s(.*?)#" % "dSatXUVTime", "%s %.6e #" % ("dSatXUVTime", -dSatXUVTime), star_in)
    star_in = re.sub("%s(.*?)#" % "dXUVBeta", "%s %.6e #" % ("dXUVBeta", -dXUVBeta), star_in)
    with open(os.path.join(PATH, "output", starFile), 'w') as f:
        print(star_in, file = f)

    # Make list of body files
    saBodyFiles = "star.in "
    for pFile in planetFiles:
        saBodyFiles += str(pFile) + " "
    saBodyFiles = saBodyFiles.strip()

    # Populate the system input file
    vpl_in = re.sub('%s(.*?)#' % "dStopTime", '%s %.6e #' % ("dStopTime", dStopTime), vpl_in)
    vpl_in = re.sub('%s(.*?)#' % "dOutputTime", '%s %.6e #' % ("dOutputTime", dOutputTime), vpl_in)
    vpl_in = re.sub('sSystemName(.*?)#', 'sSystemName %s #' % sysName, vpl_in)
    vpl_in = re.sub('saBodyFiles(.*?)#', 'saBodyFiles %s #' % saBodyFiles, vpl_in)
    with open(os.path.join(PATH, "output", sysFile), 'w') as f:
        print(vpl_in, file = f)

    # Run VPLANET and get the output, then delete the output files
    subprocess.call(["vplanet", sysFile], cwd = os.path.join(PATH, "output"))
    output = vpl.GetOutput(os.path.join(PATH, "output"), logfile = logfile)

    try:
        for pFile in planetFiles:
            os.remove(os.path.join(PATH, "output", pFile))
        os.remove(os.path.join(PATH, "output", starFile))
        os.remove(os.path.join(PATH, "output", sysFile))
        for pFile in planetFwFiles:
            os.remove(os.path.join(PATH, "output", pFile))
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
            restart=False, planetList=["planet.in"], **kwargs):
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
    dtype [("dLum", np.float64), ("dLogLXUV", np.float64), ("dPorbs", list),
           ("dPlanetMasses", list), ("dRGTimes", list), ("dEnvMasses", list),
             ("dWaterMasses", list), ("dOxygenMasses", list)]

    # Initialize the sampler object
    sampler = emcee.EnsembleSampler(nwalk, ndim, LnLike, kwargs=kwargs,
                                    pool=pool, blobs_dtype=dtype,
                                    backend=handler)

    # Actually run the MCMC
    if restart:
        sampler.run_mcmc(None, nsteps)
    else:
        for ii, result in enumerate(sampler.sample(x0, iterations=nsteps)):
            print("MCMC: %d/%d..." % (ii + 1, nsteps))

    print("Done!")
# end function
