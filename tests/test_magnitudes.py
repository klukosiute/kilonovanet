import ksm
import numpy as np

distance = 40. * 10**6 * 3.086E18
physical_params = np.array([1.e-2, 9.e-2, 3.e1, 3.e-1])

# How do I make this into command line arguments..?
metadata_location = '/home/kamile/deepnova/data/bulla_bns/metadata_bulla_bns.json'
filter_library_location = '/home/kamile/msc_thesis/kilonova_fitting/data/filter_data'


def test_correct_output_no_obs():
    # Test that the output is correct when no Observations object provided
    model = ksm.Model(metadata_location,
                      filter_library_path=filter_library_location)
    mags = model.predict_magnitudes(physical_params,
                                    times=np.array([2.2, 2.2, 2.2]),
                                    distance=distance,
                                    filters=np.array(['LSST_u', 'LSST_z', 'LSST_y']))

    np.testing.assert_allclose(mags, np.array([20.514, 17.270, 17.238]), atol=0.03)


def test_matching_outputs():
    # Test that the no-Obs and Obs ways of producing output does the same thing
    obs = ksm.Observations(np.array([2.2, 2.2, 2.2]),
                           np.array(['LSST_u', 'LSST_z', 'LSST_y']),
                           np.array([0., 0., 0]),
                           np.array([0., 0., 0]),
                           distance)
    model1 = ksm.Model(metadata_location,
                       filter_library_path=filter_library_location)
    mags1 = model1.predict_magnitudes(physical_params,
                                      times=np.array([2.2, 2.2, 2.2]),
                                      distance=distance,
                                      filters=np.array(['LSST_u', 'LSST_z', 'LSST_y']))

    model2 = ksm.Model(metadata_location,
                       filter_library_path=filter_library_location,
                       observations=obs)
    mags2 = model2.predict_magnitudes(physical_params)

    np.testing.assert_allclose(mags1, mags2, atol=0.06)


