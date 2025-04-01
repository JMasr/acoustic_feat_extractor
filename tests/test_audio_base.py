import json
import tempfile
from pathlib import Path

import pytest

from audio.base import AcousticFeatConfiguration

# AcousticFeatConfiguration Section

@pytest.fixture
def config_valid_dict():
    return {
        "feat_name": "mfcc",
        "feat_type": "spectral",
        "resampling_rate": 16000,
        "extra_param": True,
    }

@pytest.fixture
def config_invalid_dict():
    return {
        "feat_name": "mfcc",
        # "feat_type": "spectral",
        "resampling_rate": 16000,
        "extra_param": True,
    }

def test_should_init_with_valid_dictionary(config_valid_dict):
    # Arrange
    config_dict = config_valid_dict

    # Act
    config = AcousticFeatConfiguration(config_dict)

    # Assert
    assert config.feat_name == "mfcc"
    assert config.feat_type == "spectral"
    assert config.resampling_rate == 16000
    assert config.extra_param == True

    assert config.get("feat_name") == "mfcc"
    assert config.get("", None) is None

def test_should_init_with_valid_json_file(config_valid_dict):
    # Arrange
    config_dict = config_valid_dict
    config_temp_json = json.dumps(config_dict)

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    with open(f"{temp_file.name}.json", "w") as f:
        f.write(config_temp_json)

    # Act
    path_temp_file = Path(f"{temp_file.name}.json")
    config = AcousticFeatConfiguration(path_temp_file)

    # Assert
    assert config.feat_name == "mfcc"
    assert config.feat_type == "spectral"
    assert config.resampling_rate == 16000
    assert config.extra_param == True

    assert config.get("feat_name") == "mfcc"
    assert config.get("", None) is None

def test_should_not_init_with_invalid_dict(config_invalid_dict):
    # Arrange
    config_dict = config_invalid_dict

    # Act & Assert
    with pytest.raises(ValueError):
        AcousticFeatConfiguration(config_dict)

def test_should_not_init_with_nonexistent_path():
    # Arrange
    nonexistent_path = Path("/path/to/nonexistent/config.json")

    # Act & Assert
    with pytest.raises(FileNotFoundError):
        AcousticFeatConfiguration(nonexistent_path)


