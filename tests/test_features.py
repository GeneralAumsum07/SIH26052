from vaani.dsp import features


def test_feature_names_contract():
    assert isinstance(features.FEATURE_NAMES, tuple)
    assert len(features.FEATURE_NAMES) == features.N_FEATURES == 18
