import ksm
import pytest


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


@pytest.mark.parametrize(
    "metadata, torch_file", [(m, t) for m, t, in zip(METADATA, MODELS)]
)
def test_filter_imports(metadata, torch_file):
    # Here testing that not loading the filters is fine and that prediction in spectra-only mode works too
    model = ksm.Model(metadata, torch_file, filter_library_path=FILTER_LIB)
    assert model.filters_loaded

    model2 = ksm.Model(metadata, torch_file)
    assert not model2.filters_loaded
