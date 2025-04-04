from pathlib import Path
from typing import Union, Dict, Any

from gtm_feat.audio.base import AcousticFeatConfiguration, BaseFeatureExtractor
from gtm_feat.audio.embedding import IVectorExtractor, XVectorExtractor
from gtm_feat.audio.short_time import OpenSmileExtractor, SpafeExtractor
from gtm_feat.config import logger


class FeatureExtractorFactory:

    def __init__(self, config_object: Union[Path, Dict[str, Any]]):
        self.config = AcousticFeatConfiguration(config_object)

        self.SPAFE_SUPPORTED_FEATS = [
            "mfcc",
            "imfcc",
            "bfcc",
            "cqcc",
            "gfcc",
            "lfcc",
            "lpc",
            "lpcc",
            "msrcc",
            "ngcc",
            "pncc",
            "psrcc",
            "plp",
            "rplp",
        ]
        self.OPENSMILE_SUPPORTED_FEATS = [
            "basic_spectral",
            "spectral",
            "energy",
            "voicing",
            "llds",
        ]

    def create_extractor(self) -> BaseFeatureExtractor:

        dict_config = self.config.to_dict()

        # Example logic to choose the extractor based on configuration
        if ("compare_2016" in self.config.get('feat_name') or ("opensmile" in self.config.get('feat_name'))
                or (self.config.get('feat_name') in self.OPENSMILE_SUPPORTED_FEATS)):

            return OpenSmileExtractor(dict_config)

        elif ("spafe" in (self.config.get('feat_name'))
              or (self.config.get('feat_name') in self.SPAFE_SUPPORTED_FEATS)):

            return SpafeExtractor(dict_config)

        elif "ivector" in self.config.get('feat_name') or "i-vector" in self.config.get('feat_name'):

            return IVectorExtractor(dict_config)

        elif "xvector" in self.config.get('feat_name') or "x-vector" in self.config.get('feat_name'):

            return XVectorExtractor(dict_config)

        else:
            logger.error(f"Unsupported feature name: {self.config.get('feat_name')}")
            raise ValueError
