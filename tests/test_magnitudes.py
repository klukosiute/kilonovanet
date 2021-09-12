import kilonovanet
import numpy as np
import torch
import pytest


torch.manual_seed(42)
np.random.seed(42)


DISTANCE = 40.0 * 10 ** 6 * 3.086e18
FILTERS = np.array(["LSST_u", "LSST_z", "LSST_y"])
FILTER_LIB = "data/filter_data"


# bulla_bns, bulla_bhns, kasen_bns
METADATA = [
    "data/metadata_bulla_bns.json",
    "data/metadata_bulla_bhns.json",
    "data/metadata_kasen_bns.json",
]

MODELS = [
    "models/bulla-bns-latent-20-hidden-1000-CV-4-2021-04-21-epoch-200.pt",
    "models/bulla-bhns-latent-2-hidden-500-CV-4-2021-04-17-epoch-200.pt",
    "models/kasen-bns-latent-10-hidden-500-CV-4-2021-04-17-epoch-200.pt",
]


PHYSPARAMS = [
    np.array([1.0e-2, 9.0e-2, 3.0e1, 3.0e-1]),
    np.array([0.05, 0.01, 0.3]),
    np.array([2.50e-3, 3.0e-2, 1.0e-1]),
]

TIMES = [
    np.array([2.2, 2.2, 2.2]),
    np.array([2.2, 2.2, 2.2]),
    np.array([1.9499999284744263, 1.9499999284744263, 1.9499999284744263]),
]


RESULTS = [
    np.array([20.514, 17.270, 17.238]),
    np.array([20.68908, 18.6828, 18.63107]),
    np.array([31.1308, 22.8100, 21.9839]),
]


TOLERANCES = [0.03, 0.06, 0.06]


INPUTS1 = [
    (md, tf, pm, ti, op, to)
    for md, tf, pm, ti, op, to in zip(
        METADATA, MODELS, PHYSPARAMS, TIMES, RESULTS, TOLERANCES
    )
]
INPUTS2 = [
    (md, tf, pm, ti) for md, tf, pm, ti in zip(METADATA, MODELS, PHYSPARAMS, TIMES)
]


@pytest.mark.parametrize(
    "metadata, torch_file, params, times, output, tolerance", INPUTS1
)
def test_correct_output_no_obs(metadata, torch_file, params, times, output, tolerance):
    # Test that the output is correct when no Observations object provided
    model = kilonovanet.Model(metadata, torch_file, filter_library_path=FILTER_LIB)
    mags = model.predict_magnitudes(
        params, times=times, distance=DISTANCE, filters=FILTERS,
    )
    np.testing.assert_allclose(mags, output, atol=tolerance)


@pytest.mark.parametrize("metadata, torch_file, params, times", INPUTS2)
def test_matching_outputs(metadata, torch_file, params, times):
    # Test that the no-Obs and Obs ways of producing output does the same thing
    obs = kilonovanet.Observations(
        times, FILTERS, np.array([0.0, 0.0, 0]), np.array([0.0, 0.0, 0]), DISTANCE,
    )
    model1 = kilonovanet.Model(metadata, torch_file, filter_library_path=FILTER_LIB)
    mags1 = model1.predict_magnitudes(
        params, times=times, distance=DISTANCE, filters=FILTERS,
    )

    model2 = kilonovanet.Model(
        metadata, torch_file, filter_library_path=FILTER_LIB, observations=obs
    )
    mags2 = model2.predict_magnitudes(params)

    np.testing.assert_allclose(mags1, mags2, atol=0.06)
