# SynTH
### Synthesis of Transmission from Hydrodynamical simulations (SynTH)

This package provides utilities for synthesizing mock intergalactic resonant absoprtion (or transmission) from a Hydrodynamical simulation. 

Assuming an elementary composition of the gas with 75% Hydrogen ($X = 0.75$) and 25% Helium ($Y = 0.25$) by mass, and assuming ionization equilibrium among the various atomic species in the gas, densities and fractions of neutral Hydrogen (and Helium) can be estimated for any $N$-d collection of voxels from the simulation.

Further, using the known information of gas peculiar velocities, an optical depth spectrum of a resonant scattering (such as Lyman-$\alpha$) can be synthesized corresponding to a line of sight (skewer) through the simulation.



## Getting started

Importing the utilities,
```Python
import h5py
import numpy as np
from synth import SynTH as synth 
```

Getting the input skewers file and reading in the cosmological parameters,
```Python
input_hfile = 'my_sim.h5'  # contains skewers from a hydro simulations
Cosmology, Domain = synth.cosmology(input_hfile)
```

Reading in a skewer,
```Python
skewer_id = 0
with h5py.File(input_hfile) as f:
    rhob_skewer = f['rhob'][skewer_id] # in units of the mean baryon density
    vlos_skewer = f['vlos'][skewer_id] # line-of-sight component of the peculiar velocity, in km/s
    T_skewer    = f['temp'][skewer_id] # in K
```

Computing neutral Hydrogen density $n_\mathrm{HI}$ and fraction $x_\mathrm{HI}$,
```Python
# Use the Hydrogen class for computation of neutral/ionized fractions
Hydrogen = synth.Hydrogen(Cosmology, X = 0.75, TREECOOL = None)
nHI_skewer, xHI_skewer = Hydrogen.eval_HI(rhob_skewer, T_skewer, mode = 'approximate') # nHI in CGS
```

Computing the optical depth of Lyman-alpha forest abosption,
```Python
# Use the Lyman class for computation of tau
Lyman = synth.Lyman(input_hfile, X = 0.75, Y = 0.25, TREECOOL = None)
tau_lya_skewer, v_h_skewer = Lyman.eval_tau_skewer(
    rhob_skewer, 
    T_skewer, 
    vlos_skewer, 
    n_neutral = nHI_skewer, 
    element = 'H', 
    transition = 'lya', 
    profile = 'doppler', 
    return_v_h_skewer = True,
) # v_h_skewer in km/s
```
_Et voilà!_


## Rescaling optical depth values

Oftentimes it's necessary to rescale the pre-computed optical depth $\tau$ values for a set of spectra in order to match their mean transmission $\bar{F}$ to the observed value (recent literature measurements, supported by the package, are Turner et al. 2024 and Becker et al. 2013). This is necessitated by the fact that exact values of the intergalactic UV background ionization rates may not be accurately known while running the original hydro sim.

First finding the observed mean transmission at a given redshift,
```Python
z = Cosmology['redshift']
tau_eff_obs = synth.tau_effective_lya(z, fitter = 'Becker13')
F_mean_obs  = np.exp(-tau_eff_obs)
```

Then, computing a scaling factor $A$ such that $\langle \exp(-A\tau) \rangle = \bar{F}_\mathrm{obs}$,
```Python
rescaler = synth.TauRescaler(taus, F_mean_to_match = F_mean_obs)
A_scalar = rescaler.eval_scalar_A(init_guess = 0.8)
```

Hence, all the $\tau$ values can be rescaled, `taus *= A_scalar`.


## Computing summary statistics

SynTH provides utilities to compute the 1d (line-of-sight) power spectrum and the probability density function (PDF) of the Lyman-$\alpha$ transmission for a set of mock spectra.

```Python
Summary = synth.SummaryStats(taus, v_h_skewer)
```

### 1d power spectrum 
```Python
pks, k = Summary.compute_p1d(F_mean = F_mean_obs, per_spectrum = True) # this returns a p1d estimate per input tau spectrum
Delta2  = k * np.mean(pks, axis = 0) / np.pi
```
or equivalently,
```Python
Pk, k = Summary.compute_p1d(F_mean = F_mean_obs, per_spectrum = False)
Delta2  = k * Pk / np.pi
```

### PDF
```Python
bins = 50
bins = np.linspace(0, 1, bins+1)
pdfs = Summary.compute_flux_PDF(bins = bins, density = True, per_spectrum = True)
PDF  = np.mean(pdfs, axis = 0)
```
or equivalently,
```Python
bins = 50 
PDF  = Summary.compute_flux_PDF(bins = bins, density = True, per_spectrum = False)
```


## Get in touch!
The package is developed and maintained by Parth Nayak. Write an email to [parth3e8@gmail.com](mailto:parth3e8@gmail.com) to get in touch regarding any questions/issues.  