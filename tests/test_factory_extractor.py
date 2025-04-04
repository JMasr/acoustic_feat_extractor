from pathlib import Path

import librosa
import pytest
import tempfile
import soundfile as sf
import numpy as np
import pandas as pd

from audio.factory import FeatureExtractorFactory
from config import TEST_DIR


@pytest.fixture
def dummy_wav():
    dummy_wav, _ = librosa.load(librosa.example('brahms'))
    return dummy_wav


@pytest.fixture
def dummy_df_train():
    num_wav, list_temp_file = 5, []

    dummy_wav, sr = librosa.load(librosa.example('brahms'))
    for index in range(num_wav):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        with open(f"{temp_file.name}.wav", "wb") as f:
            sf.write(f, dummy_wav, sr, subtype='PCM_24')
        list_temp_file.append(f"{temp_file.name}.wav")

    df_train_data = pd.DataFrame({
        "path": list_temp_file,
        "label": np.random.randint(0, 2, num_wav),
        "id": np.random.randint(0, 1000, num_wav),
    })
    return df_train_data

@pytest.fixture
def config_opensmile_valid_dict():
    return {
        "feat_name": "opensmile",
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
def config_ivector_valid_dict():
    return {
        "feat_name": "i-vector",
        "feat_type": "i-vector",
        "origin_model": TEST_DIR / "pretrained_models" / "i_vector-custom",
        "model_local_path": TEST_DIR / "pretrained_models" / "i_vector-custom",
        "tag": "custom_test",
        "resampling_rate": 16000,
        "n_mfcc": 20,
        "n_fft": 512,
        "hop_length": 256,
        "n_mels": 40,
        "knn_max_iter": 2,
        "ubm_gaussians": 2,
        "ubm_max_fitting_steps": 2,
        "ivector_dim": 2,
        "ivector_epochs": 2,
    }


def test_should_creates_opensmile_extractor(config_opensmile_valid_dict):
    # Arrange
    config = config_opensmile_valid_dict

    # Act
    extractor = FeatureExtractorFactory(config).create_extractor()

    # Assert
    assert extractor.__class__.__name__ == "OpenSmileExtractor"

def test_should_extract_opensmile_feat_with_valid_audio(config_opensmile_valid_dict, dummy_wav):
    # Arrange
    config = config_opensmile_valid_dict
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(f"{temp_file.name}.wav", "wb") as f:
        sf.write(f, dummy_wav, 22050, subtype='PCM_24')

    # Act
    feat_extractor = FeatureExtractorFactory(config).create_extractor()
    result = feat_extractor.extract(Path(f"{temp_file.name}.wav"))

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert 'F0final_sma' in result.columns

def test_should_creates_spafe_extractor(config_spafe_valid_dict):
    # Arrange
    config = config_spafe_valid_dict

    # Act
    extractor = FeatureExtractorFactory(config).create_extractor()

    # Assert
    assert extractor.__class__.__name__ == "SpafeExtractor"

def test_should_extract_spafe_feat_with_valid_audio(config_spafe_valid_dict, dummy_wav):
    # Arrange
    config = config_spafe_valid_dict
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(f"{temp_file.name}.wav", "wb") as f:
        sf.write(f, dummy_wav, 22050, subtype='PCM_24')

    # Act
    spafe_extractor = FeatureExtractorFactory(config).create_extractor()
    result = spafe_extractor.extract(Path(f"{temp_file.name}.wav"))

    # Assert
    assert isinstance(result, pd.DataFrame)
    assert result.std().sum() > 0


def test_should_creates_ivector_extractor(config_ivector_valid_dict):
    # Arrange
    config = config_ivector_valid_dict

    # Act
    extractor = FeatureExtractorFactory(config).create_extractor()

    # Assert
    assert extractor.__class__.__name__ == "IVectorExtractor"