import shutil
import tempfile

import librosa
import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from gtm_feat.audio.embedding import IVectorExtractor, XVectorExtractor
from gtm_feat.config import TEST_DIR


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


# I-Vector Section
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


def test_should_init_ivector_with_valid_config(config_ivector_valid_dict):
    # Arrange
    config = config_ivector_valid_dict

    # Act
    extractor = IVectorExtractor(config)

    # Assert
    assert extractor.config.__dict__ == config_ivector_valid_dict


def test_should_train_ivector_with_valid_dataset(config_ivector_valid_dict, dummy_df_train):
    # Arrange
    extractor = IVectorExtractor(config_ivector_valid_dict)

    # Act
    extractor.train(dummy_df_train)

    # Assert
    assert extractor.config.model_local_path.exists()
    assert extractor.load_model() == True
    # Clean
    shutil.rmtree(config_ivector_valid_dict.get("model_local_path").parent)


def test_should_extract_ivector_with_valid_dataset(config_ivector_valid_dict, dummy_df_train):
    # Arrange
    ivector_extractor = IVectorExtractor(config_ivector_valid_dict)

    # Act
    ivector_extractor.train(dummy_df_train)

    ivector = ivector_extractor.extract(dummy_df_train.path.iloc[0])
    ivector_list = ivector_extractor.extract(list(dummy_df_train.path))

    # Assert
    assert ivector.shape == (1, config_ivector_valid_dict["ivector_dim"])
    assert ivector.sum(axis=1)[0] != 0

    assert len(ivector_list) == len(list(dummy_df_train.path))
    # Clean
    shutil.rmtree(config_ivector_valid_dict.get("model_local_path").parent)

def test_should_extract_ivector_windowed_with_valid_audio(config_ivector_valid_dict, dummy_df_train):
    # Arrange
    ivector_extractor = IVectorExtractor(config_ivector_valid_dict)
    ivector_extractor.train(dummy_df_train)

    # Act
    ivector = ivector_extractor.extract_windowed(dummy_df_train.path.iloc[0])

    # Assert
    assert isinstance(ivector, list)

    # Clean
    shutil.rmtree(config_ivector_valid_dict.get("model_local_path").parent)


# x-Vector Section
@pytest.fixture
def config_xvector_valid_dict():
    return {
        "feat_name": "x-vector",
        "feat_type": "x-vector",
        "origin_model": "speechbrain/spkrec-xvect-voxceleb",
        "model_local_path": TEST_DIR / "pretrained_models" / "x_vector-custom",
        "resampling_rate": 16000,
        "xvector_dim": 512,
    }


def test_should_init_xvector_with_valid_config(config_xvector_valid_dict):
    # Arrange
    config = config_xvector_valid_dict

    # Act
    xvector_extractor = XVectorExtractor(config)

    # Assert
    assert xvector_extractor.config.__dict__ == config_xvector_valid_dict


def test_should_extract_xvector_with_valid_dataset(config_xvector_valid_dict, dummy_df_train):
    # Arrange
    xvector_extractor = XVectorExtractor(config_xvector_valid_dict)

    # Act
    x_vector = xvector_extractor.extract(dummy_df_train.path.iloc[0])
    x_vector_list = xvector_extractor.extract(list(dummy_df_train.path))

    # Assert
    assert len(x_vector) == config_xvector_valid_dict["xvector_dim"]
    assert x_vector.sum(axis=0)[0] != 0

    assert len(x_vector_list) == len(list(dummy_df_train.path))
    # Clean
    shutil.rmtree(config_xvector_valid_dict.get("model_local_path").parent)

def test_should_extract_xvector_windowed_with_valid_audio(config_xvector_valid_dict, dummy_df_train):
    # Arrange
    xvector_extractor = XVectorExtractor(config_xvector_valid_dict)

    # Act
    xvector = xvector_extractor.extract_windowed(dummy_df_train.path.iloc[0], window_length_ms= 40)

    # Assert
    assert isinstance(xvector, list)
    # Clean
    shutil.rmtree(config_xvector_valid_dict.get("model_local_path").parent)
