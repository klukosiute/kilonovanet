import ksm
import numpy as np
import torch
import pandas as pd

# WIP: This is a set of tests to ensure that the nested sampling fit will work correctly.
# But there are more tests to ensure this, I've just not yet written them.

torch.manual_seed(42)
np.random.seed(42)


DISTANCE = 40.0 * 10 ** 6 * 3.086e18
FILTERS = np.array(["LSST_u", "LSST_z", "LSST_y"])
FILTER_LIB = "data/filter_data"


def import_data():
    df = pd.read_csv("data/GW170817.dat", delimiter=" ")
    df["date"] = pd.to_datetime(df["date"])
    df["date_offset"] = pd.to_datetime("2017-08-17T12:41:00.00")
    df["delta_t"] = df["date"] - df["date_offset"]
    df = df[df["delta_t"] < np.timedelta64(14, "D")]

    all_bands = ["u", "g", "r", "i", "z", "y", "J", "H", "K"]
    band_rules = ["LSST_"] * 6 + ["CTIO_"] * 3

    band_rules_dict = {i: j for (i, j) in zip(all_bands, band_rules)}

    times = np.array(df.delta_t.values / np.timedelta64(1, "D"), dtype=np.float32)
    mags = np.array(df.mag.values, dtype=float)
    err = np.array(df.mag_err.values, dtype=float)

    bands = []
    for band in df.band.values:
        bands.append(band_rules_dict[band] + band)
    bands = np.array(bands)

    observations = ksm.Observations(times, bands, mags, err, 1.23e26)
    return observations


def create_model():
    observations = import_data()
    model = ksm.Model(
        "data/metadata_bulla_bns.json",
        "models/bulla-bns-latent-20-hidden-1000-CV-4-2021-04-21-epoch-200.pt",
        filter_library_path=FILTER_LIB,
        observations=observations,
    )
    return model


def test_import_data():
    observations = import_data()

    assert np.all(
        observations.times[:10]
        == np.array(
            [
                0.47152778,
                0.47152778,
                0.47152778,
                0.47152778,
                0.47152778,
                0.47152778,
                0.47152778,
                0.47152778,
                0.70252778,
                0.70252778,
            ],
            dtype=np.float32,
        )
    )
    assert np.all(
        observations.magnitude_errors[:10]
        == np.array([0.02, 0.04, 0.03, 0.03, 0.01, 0.03, 0.15, 0.11, 0.06, 0.06])
    )
    assert not np.all(observations.upper_limit[:10])


def test_mag_upper_limits():
    # this should at minimum make observation index 41 be affected

    model = create_model()
    params = np.array([0.02, 0.13, 10, 0.5])
    lklhd = model.compute_likelihood_dynesty(params)
    assert not np.isfinite(lklhd)


def test_prior():
    model = create_model()
    params_min = np.array([0.001, 0.01, 0, 0])
    params_max = np.array([0.02, 0.13, 90, 1])
    prior_min = model.prior_transform_dynesty(np.array([0, 0, 0.0, 0.0]))
    prior_max = model.prior_transform_dynesty(np.array([1, 1, 1.0, 1.0]))

    np.testing.assert_allclose(prior_min, params_min)
    np.testing.assert_allclose(prior_max, params_max)
