import tempfile
from pathlib import Path

import librosa
import pandas as pd
import pytest
import soundfile as sf

from audio.short_time import OpenSmileExtractor


@pytest.fixture
def config_valid_dict():
    return {
        "feat_name": "compare_2016_voicing",
        "feat_type": "compare_2016_voicing",
        "resampling_rate": 22050,
        "compute_deltas": True,
        "compute_deltas_deltas": True,
    }


@pytest.fixture
def dummy_wav():
    dummy_wav, _ = librosa.load(librosa.example('brahms'))
    return dummy_wav



# OpenSmileExtractor Section


def test_extract_features_from_valid_audio(config_valid_dict, dummy_wav):
    # Arrange
    config = config_valid_dict

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

def test_extract_features_from_valid_list_audios(config_valid_dict, dummy_wav):
    # Arrange
    config = config_valid_dict

    list_temp_file = []
    for index  in range(50):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(f"{temp_file.name}.wav", "wb") as f:
            sf.write(f, dummy_wav, 22050, subtype='PCM_24')
        list_temp_file.append(f"{temp_file.name}.wav")

    # Act
    feat_extractor = OpenSmileExtractor(config)
    result = feat_extractor.extract(list_temp_file)

    # Assert
    assert isinstance(result, list)
    assert len(result) == 50
    assert 'F0final_sma' in result[0].columns
    assert len(result[0].columns) == 6 * 3  # All expected features plus deltas and deltas-deltas
