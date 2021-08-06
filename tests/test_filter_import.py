import ksm

model = ksm.Model('/home/kamile/deepnova/data/bulla_bhns/metadata_bulla_bhns.json',
                  filter_library_path='/home/kamile/msc_thesis/kilonova_fitting/data/filter_data')
assert model.filters_loaded


model = ksm.Model('/home/kamile/deepnova/data/bulla_bhns/metadata_bulla_bhns.json')
#                  filter_library_path='/home/kamile/msc_thesis/kilonova_fitting/data/filter_data')

assert not model.filters_loaded