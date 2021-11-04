# KilonovaNet: Kilonova Surrogate Modelling

A conditional variational autoencoder (cVAE) framework for producing continuous
surrogate spectra for kilonova models.

This package provides the interface to predict spectra. It does not provide an
interface to do the data prep for training and training itself. The currently
trained and provided models are:

- [M. Bulla BNS Models](https://github.com/mbulla/kilonova_models/tree/master/bns_m3_3comp)
- [M. Bulla BHNS Models](https://github.com/mbulla/kilonova_models/tree/master/bhns_m1_2comp)
- [D.Kasen BNS Models](https://github.com/dnkasen/Kasen_Kilonova_Models_2017)

This work requires the use of [pyphot](https://github.com/mfouesneau/pyphot) and
therefore there are a number of dependencies that are needed to just get pyphot working,
i.e. astropy, pytables, etc. My `environment.yml` for conda environments should handle
those but you will need to install [pyphot](https://github.com/mfouesneau/pyphot) before
using this work.

## Installation
Install from source: download and run `python setup.py install`.

Install via pip: `pip install kilonovanet`
## Usage
In order to produce surrogate spectra (see *eventual paper* for discussion about
how good these spectra are or are not, though), use:

```python
import kilonovanet
import numpy as np

metadata_file = "data/metadata_bulla_bns.json"
torch_file = "models/bulla-bns-latent-20-hidden-1000-CV-4-2021-04-21-epoch-200.pt"
times = np.array([1.2, 2.2])
physical_parameters = np.array([1.0e-2, 9.0e-2, 3.0e1, 3.0e-1])

model = kilonovanet.Model(metadata_file, torch_file)
spectra = model.predict_spectra(physical_parameters, times)
```

In order to produce some photometric observations, the following have to be specified:
- the model
- the corresponding parameters of the model (see their papers, repositories, etc.)
- the times post-merger to produce the observations
- the filters in which to produce the observations

The general use is then as follows:

```python
import kilonovanet
import numpy as np
 
metadata_file = "data/metadata_bulla_bns.json"
torch_file = "models/bulla-bns-latent-20-hidden-1000-CV-4-2021-04-21-epoch-200.pt"
filter_lib = "data/filter_data"

times = np.array([1.2, 1.2, 1.2, 2.2, 2.2, 2.2, 2.2])
filters = np.array(["LSST_u", "LSST_z", "LSST_y", "LSST_u", "LSST_z", "LSST_y"])
distance = 40.0 * 10 ** 6 * 3.086e18 # 40 Mpc in cm
physical_parameters = np.array([1.0e-2, 9.0e-2, 3.0e1, 3.0e-1])

model = kilonovanet.Model(metadata_file, torch_file, filter_library_path=filter_lib)
mags = model.predict_magnitudes(physical_parameters, times=times, filters=filters,
distance=distance)
```

If you intend to use the same set of observations often, e.g. when doing an
MCMC-based fit, you can specify all of them in an `Observations` object and
then simply call `model.predict_magnitudes(physical_parameters)`. 

### Warnings
- All specified model parameter values have to lie within the ranges of the original
radiative transport simulations! This code will not throw errors if you do not do this
but will instead return nonsense results, so be mindful to read their papers.
