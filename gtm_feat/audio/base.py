from abc import ABC, ABCMeta, abstractmethod
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import torch

from gtm_feat.config import logger


class AcousticFeatConfiguration(ABC):
    mandatory_config_arguments = ["feat_name", "feat_type", "resampling_rate"]

    def __init__(self, config_object: Union[Path, Dict[str, Any]]):
        """
        AcousticFeatConfiguration class

        This configuration class is common for all feature extractors. It enforces
        that each configuration includes mandatory arguments: 'feat_name', 'feat_type',
        and 'resampling_rate'.

        Args:
            config_object (Union[Path, Dict[str, Any]]): A path to a JSON configuration file
                or a dictionary containing configuration parameters.
        """
        self.feat_name: str
        self.feat_type: str
        self.resampling_rate: int

        config = self._load_config_object(config_object)
        self._validate_config(config)
        self._apply_config(config)
        logger.info(f"Configuration loaded successfully: {config}")

    @staticmethod
    def _load_config_object(config_object: Union[Path, Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(config_object, Path):
            if config_object.exists() and config_object.suffix == ".json":
                try:
                    with config_object.open("r") as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load configuration from {config_object}")
                    raise IOError(f"Failed to load configuration: {e}")
            else:
                logger.error(f"Config file {config_object} does not exist or is not a JSON file.")
                raise FileNotFoundError
        elif isinstance(config_object, dict):
            return config_object
        else:
            logger.error(
                f"Configuration must be a dict or a valid Path to a JSON file, got {type(config_object)}."
            )
            raise TypeError

    def _validate_config(self, config: Dict[str, Any]) -> None:
        missing_arguments = [arg for arg in self.mandatory_config_arguments if arg not in config]
        if missing_arguments:
            logger.error(
                f"Missing mandatory configuration arguments: {', '.join(missing_arguments)}"
            )
            raise ValueError

    def _apply_config(self, config: Dict[str, Any]) -> None:
        for key, value in config.items():
            setattr(self, key, value)

    def get(self, parameter_name: str, subtitution: Any = None) -> Any:
        try:
            parameter_value = self.__getattribute__(parameter_name)
            if not parameter_value:
                return subtitution
        except Exception:
            logger.warning(
                f"Failed to get parameter *{parameter_name}*. Using default value *{subtitution}*"
            )
            return subtitution

        return parameter_value


class AcousticEmbeddingConfiguration(AcousticFeatConfiguration):
    mandatory_config_arguments = [
        "feat_name",
        "feat_type",
        "resampling_rate",
        "origin_model",
        "model_local_path",
    ]

    def __init__(self, config_object: Union[Path, Dict[str, Any]]):
        # Define the extended class parameters
        self.origin_model: str
        self.model_local_path: Path

        # Init the main class
        super().__init__(config_object)


class BaseFeatureExtractor(object, metaclass=ABCMeta):
    def __init__(self, config_object: Union[Path, Dict[str, Any]]):
        """Base Feature Extractor class

        All feature pre- and post-processors should subclass it.
        All subclass should overwrite:

        - Methods:``extract``, used for running the processing functionality.

        Args:
            config_object (Union[Path, Dict[str, Any]]): Path to a .json file or a dictionary with all feat parameters.
        """
        super().__init__()
        self.config: AcousticFeatConfiguration = AcousticFeatConfiguration(config_object)

    @abstractmethod
    def preprocessor(self, raw_audio_path: Path) -> Union[torch.Tensor, np.ndarray]:
        """Abstract method that implement the reading and preprocessing audio pipeline"""

    @abstractmethod
    def feature_transform(
        self, pre_audio: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Abstract method that implement the algorithm to transform raw audio to acoustic features."""

    @abstractmethod
    def postprocessor(
        self, acoustic_feats: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Abstract method that implement the postprocessing feat pipeline(calculation of delta, delta-delta, etc.)"""

    def _extract_single(self, raw_audio_path: Path) -> Union[torch.Tensor, np.ndarray]:
        """
        Default method for extracting the features.
        """
        try:
            pre_audio = self.preprocessor(raw_audio_path)
            acoustic_feats = self.feature_transform(pre_audio)
            post_feats = self.postprocessor(acoustic_feats)
        except Exception as e:
            logger.error(f"Failed to extract features from {raw_audio_path}. ERROR: {e}")
            raise RuntimeError
        return post_feats

    def _extract_parallel(
        self, raw_audios_paths: List[Path], n_jobs: int = None
    ) -> List[Union[torch.Tensor, np.ndarray]]:
        """
        Parallel version for extracting features from multiple audio files concurrently.
        If the length of the list is equal or minor that n_jobs, the extraction will be serial for efficiency reasons.

        Args:
            raw_audios_paths (List[Path]): List of audio file paths.
            n_jobs (int): Number of worker threads to use for parallel processing. Default value is the available cpus.

        Returns:
            List of extracted features (one for each audio file).
        """
        if n_jobs is None:
            max_workers = os.cpu_count()
            n_jobs = max_workers - 1

        try:
            results = []
            if len(raw_audios_paths) <= n_jobs:
                for raw_audio_path in raw_audios_paths:
                    results.append(self._extract_single(raw_audio_path))
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    # Map each audio file to the _extract_single method concurrently.
                    results = list(executor.map(self._extract_single, raw_audios_paths))
        except Exception as e:
            logger.error(f"Failed to extract features from {raw_audios_paths}. ERROR: {e}")
            raise RuntimeError

        return results

    def extract(self, raw_audio_path: Union[Path, List[Path]]):
        """
        Method to extract acoustic feat for

        """
        if isinstance(raw_audio_path, list):
            results = self._extract_parallel(raw_audio_path)
        elif isinstance(raw_audio_path, Path):
            results = self._extract_single(raw_audio_path)
        else:
            logger.error(f"Raw audio path {raw_audio_path} must be a Path or a list of Paths.")
            raise ValueError

        return results


class BaseDownloader(object, metaclass=ABCMeta):
    def __init__(
        self,
        file_id,
        path_local,
    ):
        """Base class for downloaders.

        All downloaders should subclass it.
        All subclass should overwrite:

        - Methods:``run``, supporting to run the download functionality.

        Args:
            file_id (str): ID of the hosted file (e.g. Google Drive File ID, HuggingFace URL, etc.).
            path_local (str): The file is downloaded to this local path.

        """
        super().__init__()
        self.file_id = file_id
        self.path_local = path_local

    @abstractmethod
    def run(self) -> None:
        """Abstract method that should implement the download functionality."""

    @abstractmethod
    def load(self) -> Any:
        """
        Abstract method that should implement the loading functionality.

        :return: A BaseModel object
        """


class BaseEmbeddingExtractor(BaseFeatureExtractor, ABC):
    def __init__(self, config_object: Union[Path, Dict[str, Any]]):
        """Base Embedding Extractor class

        All embedding feature pre- and post-processors should subclass it.
        All subclass should overwrite:

        - Methods:
            * ``extract``, used for running the processing functionality.
            * ``load_model``, used to load a pre-trained model extractor.
            * ``train``, used to train a new model extractor with custom data.


        Args:
            config_object (Union[Path, Dict[str, Any]]): Path to a .json file or a dictionary with all feat parameters.
        """
        super().__init__(config_object)

    @abstractmethod
    def load_model(self):
        """Loads the embedding models."""

    @abstractmethod
    def preprocessor(self, raw_audio_path: Path) -> Union[torch.Tensor, np.ndarray]:
        """Abstract method that implement the reading and preprocessing audio pipeline."""

    @abstractmethod
    def feature_transform(
        self, pre_audio: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Abstract method that implement the algorithm to transform raw audio to acoustic features."""

    @abstractmethod
    def postprocessor(
        self, acoustic_feats: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """Abstract method that implement the postprocessing feat pipeline(calculation of delta, delta-delta, etc.)"""

    @abstractmethod
    def train(self, df_custom_data: pd.DataFrame, train_config: dict[str, Any], path_local: Path):
        """Implementation of a custom training of the embedding model using a custom dataset."""
