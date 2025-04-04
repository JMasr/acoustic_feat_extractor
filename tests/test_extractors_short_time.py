import tempfile
from pathlib import Path

import librosa
import loguru
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from gtm_feat.audio.short_time import OpenSmileExtractor, SpafeExtractor


@pytest.fixture
def config_opensmile_valid_dict():
    return {
        "feat_name": "compare_2016_voicing",
        "feat_type": "compare_2016_voicing",
        "resampling_rate": 22050,
        "compute_deltas": True,
        "compute_deltas_deltas": True,
    }

@pytest.fixture
def config_spafe_valid_dict():
    return {
        "feat_name": "mfcc",
        "feat_type": "spafe_mfcc",
        "resampling_rate": 22050,
        "compute_deltas": True,
        "compute_deltas_deltas": True,
    }


@pytest.fixture
def dummy_wav():
    dummy_wav, _ = librosa.load(librosa.example('brahms'))
    return dummy_wav



# OpenSmileExtractor Section
def test_opensmile_extractor_with_valid_audio(config_opensmile_valid_dict, dummy_wav):
    # Arrange
    config = config_opensmile_valid_dict

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(f"{temp_file.name}.wav", "wb") as f:
        sf.write(f, dummy_wav, 22050, subtype='PCM_24')

    # Act
    feat_extractor = OpenSmileExtractor(config)
    result = feat_extractor.extract(Path(f"{temp_file.name}.wav"))

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert 'F0final_sma' in result.columns
    assert len(result.columns) == 6 * 3  # All expected features plus deltas and deltas-deltas

def test_opensmile_extractor_with_valid_list_audios(config_opensmile_valid_dict, dummy_wav, num_wav=35):
    # Arrange
    config = config_opensmile_valid_dict

    list_temp_file = []
    for index  in range(num_wav):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(f"{temp_file.name}.wav", "wb") as f:
            sf.write(f, dummy_wav, 22050, subtype='PCM_24')
        list_temp_file.append(Path(f"{temp_file.name}.wav"))

    # Act
    feat_extractor = OpenSmileExtractor(config)
    result = feat_extractor.extract(list_temp_file)

    # Assert
    assert isinstance(result, list)
    assert len(result) == num_wav
    assert 'F0final_sma' in result[0].columns
    assert len(result[0].columns) == 6 * 3  # All expected features plus deltas and deltas-deltas

# SpafeExtractor Section
SPAFE_SUPPORTED_FEATS = [
        "spafe_mfcc",
        "spafe_imfcc",
        "spafe_bfcc",
        "spafe_cqcc",
        "spafe_gfcc",
        "spafe_lfcc",
        "spafe_lpc",
        "spafe_lpcc",
        "spafe_msrcc",
        "spafe_ngcc",
        "spafe_pncc",
        "spafe_psrcc",
        "spafe_plp",
        "spafe_rplp",
    ]

def test_spafe_extractor_with_valid_config(config_spafe_valid_dict, dummy_wav):
    # Arrange
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(f"{temp_file.name}.wav", "wb") as f:
        sf.write(f, dummy_wav, 22050, subtype='PCM_24')

    # Act

    for spafe_feat in SPAFE_SUPPORTED_FEATS:
        loguru.logger.info(f"Testing {spafe_feat} feature extraction...")
        config_spafe_valid_dict["feat_type"] = spafe_feat
        spafe_extractor = SpafeExtractor(config_spafe_valid_dict)
        result = spafe_extractor.extract(Path(f"{temp_file.name}.wav"))

        # Assert
        assert isinstance(result, pd.DataFrame)

        feat_array: np.ndarray = result.to_numpy()
        assert feat_array.shape[0] > 1
        assert feat_array.shape[1] > 1
        assert feat_array.sum() != 0

def test_spafe_extractor_with_valid_list_audios(config_spafe_valid_dict, dummy_wav, num_wav = 35):
    # Arrange
    list_temp_file = []
    for index in range(num_wav):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(f"{temp_file.name}.wav", "wb") as f:
            sf.write(f, dummy_wav, 22050, subtype='PCM_24')
        list_temp_file.append(Path(f"{temp_file.name}.wav"))

    # Act
    spafe_extractor = SpafeExtractor(config_spafe_valid_dict)
    result = spafe_extractor.extract(list_temp_file)

    # Assert
    assert isinstance(result, list)
    assert len(result) == num_wav

    assert isinstance(result[0], pd.DataFrame)
    assert result[0].shape[0] > 1
    assert result[0].shape[1] > 1