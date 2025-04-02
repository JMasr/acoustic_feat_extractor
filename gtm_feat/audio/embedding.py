from pathlib import Path
from typing import Any, Union

from numpy import ndarray
from pandas import DataFrame
from torch import Tensor

from gtm_feat.audio.base import BaseEmbeddingExtractor


class IvectorExtractor(BaseEmbeddingExtractor):
    def __init__(self, config_object: Union[Path, dict]):
        super().__init__(config_object)
        self.load_model()

    def load_model(self):
        # TODO: Implement this
        pass

    def preprocessor(self, raw_audio_path: Path) -> Union[Tensor, ndarray]:
        # TODO: Implement this
        pass

    def feature_transform(self, pre_audio: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        # TODO: Implement this
        pass

    def postprocessor(self, acoustic_feats: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        # TODO: Implement this
        pass

    def train(self, df_custom_data: DataFrame, train_config: dict[str, Any], path_local: Path):
        # TODO: Implement this
        pass
