import numpy as np
import pandas as pd
import pickle
import dynesty
import matplotlib.pyplot as plt
import ksm

import signal

class timeout:
    def __init__(self, seconds=1, error_message='Timeout'):
        self.seconds = seconds
        self.error_message = error_message
    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)
    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
    def __exit__(self, type, value, traceback):
        signal.alarm(0)

df = pd.read_csv('./GW170817.dat', delimiter=' ')
df['date'] = pd.to_datetime(df['date'])
df['date_offset'] =  pd.to_datetime('2017-08-17T12:41:00.00')
df['delta_t'] = df['date'] - df['date_offset']
df = df[df['delta_t'] < np.timedelta64(14, 'D')]


lsst_bands = ['u', 'g', 'r', 'i', 'z', 'y']
other = ['J', 'H', 'K']
all_bands = ['u', 'g', 'r', 'i', 'z', 'y','J', 'H', 'K']
band_rules = ['LSST_']*6 + ['CTIO_']*3

band_rules_dict = {i:j for (i,j) in zip(all_bands, band_rules) }

times = np.array(df.delta_t.values / np.timedelta64(1, 'D'), dtype=float)
mags = np.array(df.mag.values, dtype=float)
err = np.array(df.mag_err.values, dtype=float)

bands = []
for band in df.band.values:
    bands.append(band_rules_dict[band] + band)
bands = np.array(bands)

observations = ksm.Observations(times, bands, mags, err,1.23E26)

model = ksm.Model('/home/kamile/ksm/data/metadata_bulla_bns.json',
                 '/home/kamile/ksm/models/bulla-bns-latent-20-hidden-1000-CV-4-2021-04-21-epoch-200.pt',
                 filter_library_path='/home/kamile/ksm/data/filter_data',
                 observations=observations)



# fit using dynesty
dsampler = dynesty.NestedSampler(model.compute_likelihood_dynesty, model.prior_transform_dietrich_dynesty, ndim=4) #ndim=4 for bulla model


#try:
#    with timeout(seconds=28800):
#dsampler.run_nested()
#except TimeoutError:
#    pass

try:
    dsampler.run_nested()
except KeyboardInterrupt:
    results = dsampler.results
    pickle.dump(results, open('170817_results_dietrich_with1sigma.p', 'wb'))

results = dsampler.results
pickle.dump(results, open('170817_results_dietrich_with1sigma.p', 'wb'))




