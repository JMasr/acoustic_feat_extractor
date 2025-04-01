import os
from pathlib import Path
from typing import Union

import librosa
import numpy as np
import opensmile
import pandas as pd
from numpy import max, concatenate, abs
from torch import Tensor, from_numpy
from torchaudio.functional import compute_deltas

from gtm_feat.audio.base import BaseFeatureExtractor
from gtm_feat.config import logger


def check_wav_file(input_path: Union[Path, str]):
    if isinstance(input_path, str):
        input_path = Path(input_path)

    if not input_path.exists():
        raise ValueError(f"File {input_path} does not exist.")

    if os.path.getsize(input_path) <= 44:
        raise ValueError(f"File {input_path} is too small to be a valid wav file.")

    if not input_path.suffix == ".wav":
        raise ValueError(f"File {input_path} is not a valid wav file.")


def read_a_wav_file(wav_path: Path, fr: int = 1600, top_db: int = 30, pre_emphasis: float = 0.97) -> np.ndarray:
    """
    Reads, processes, and returns a WAV file's audio tensor and sampling rate.
    """
    check_wav_file(wav_path)

    try:
        # Load the audio file
        audio_ndarray, original_sr = librosa.load(wav_path, sr=None, mono=True)

        # Resample audio if necessary
        if 0 < fr < original_sr:
            audio_ndarray = librosa.resample(y=audio_ndarray, orig_sr=original_sr, target_sr=fr)

        # Apply speech activity detection and concatenate segments
        speech_segments = librosa.effects.split(audio_ndarray, top_db=top_db)
        audio_ndarray = concatenate([audio_ndarray[start:end] for start, end in speech_segments])

        # Apply a pre-emphasis filter
        audio_ndarray = librosa.effects.preemphasis(audio_ndarray, coef=pre_emphasis)

        # Normalize
        audio_ndarray /= max(abs(audio_ndarray))

        return audio_ndarray

    except Exception as e:
        raise RuntimeError(f"Error reading audio file at {wav_path}: {e}")


class OpenSmileExtractor(BaseFeatureExtractor):
    def __init__(self, config_object: Union[Path, dict]):
        super().__init__(config_object)

    def preprocessor(self, raw_audio_path: Path) -> np.ndarray:
        return read_a_wav_file(raw_audio_path,
                               fr=self.config.get("resampling_rate", 16000),
                               top_db=self.config.get("top_db", 30),
                               pre_emphasis=self.config.get("pre_emphasis", 0.97), )

    def feature_transform(self, pre_audio: np.ndarray) -> pd.DataFrame:
        feature_transform = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
            sampling_rate=self.config.resampling_rate,
        )

        # opensmile expects a numpy array; if necessary, convert the tensor.
        if isinstance(pre_audio, Tensor):
            pre_audio = pre_audio.numpy()

        return feature_transform.process_signal(pre_audio, self.config.resampling_rate)

    def opensmile_subset_selector(self, acoustic_feats: pd.DataFrame):
        # If the acoustic_feats is not a DataFrame, try converting it.
        if not isinstance(acoustic_feats, pd.DataFrame):
            try:
                acoustic_feats = pd.DataFrame(acoustic_feats)
            except Exception as e:
                logger.error(f"Cannot convert acoustic_feats to DataFrame: {e}")
                raise RuntimeError(f"Postprocessing error: {e}")

        # Select a subset of features based on the feature type.
        opensmile_feats_subset = []
        feature_type = self.config.feat_type.lower()

        if feature_type == "compare_2016_voicing":
            opensmile_feats_subset = [
                "F0final_sma",
                "voicingFinalUnclipped_sma",
                "jitterLocal_sma",
                "jitterDDP_sma",
                "shimmerLocal_sma",
                "logHNR_sma",
            ]

        elif feature_type == "compare_2016_energy":
            opensmile_feats_subset = [
                "audspec_lengthL1norm_sma",
                "audspecRasta_lengthL1norm_sma",
                "pcm_RMSenergy_sma",
                "pcm_zcr_sma",
            ]

        elif feature_type == "compare_2016_spectral":
            opensmile_feats_subset = [
                "audSpec_Rfilt_sma[0]",
                "audSpec_Rfilt_sma[1]",
                "audSpec_Rfilt_sma[2]",
                "audSpec_Rfilt_sma[3]",
                "audSpec_Rfilt_sma[4]",
                "audSpec_Rfilt_sma[5]",
                "audSpec_Rfilt_sma[6]",
                "audSpec_Rfilt_sma[7]",
                "audSpec_Rfilt_sma[8]",
                "audSpec_Rfilt_sma[9]",
                "audSpec_Rfilt_sma[10]",
                "audSpec_Rfilt_sma[11]",
                "audSpec_Rfilt_sma[12]",
                "audSpec_Rfilt_sma[13]",
                "audSpec_Rfilt_sma[14]",
                "audSpec_Rfilt_sma[15]",
                "audSpec_Rfilt_sma[16]",
                "audSpec_Rfilt_sma[17]",
                "audSpec_Rfilt_sma[18]",
                "audSpec_Rfilt_sma[19]",
                "audSpec_Rfilt_sma[20]",
                "audSpec_Rfilt_sma[21]",
                "audSpec_Rfilt_sma[22]",
                "audSpec_Rfilt_sma[23]",
                "audSpec_Rfilt_sma[24]",
                "audSpec_Rfilt_sma[25]",
                "pcm_fftMag_fband250-650_sma",
                "pcm_fftMag_fband1000-4000_sma",
                "pcm_fftMag_spectralRollOff25.0_sma",
                "pcm_fftMag_spectralRollOff50.0_sma",
                "pcm_fftMag_spectralRollOff75.0_sma",
                "pcm_fftMag_spectralRollOff90.0_sma",
                "pcm_fftMag_spectralFlux_sma",
                "pcm_fftMag_spectralCentroid_sma",
                "pcm_fftMag_spectralEntropy_sma",
                "pcm_fftMag_spectralVariance_sma",
                "pcm_fftMag_spectralSkewness_sma",
                "pcm_fftMag_spectralKurtosis_sma",
                "pcm_fftMag_spectralSlope_sma",
                "pcm_fftMag_psySharpness_sma",
                "pcm_fftMag_spectralHarmonicity_sma",
                "mfcc_sma[1]",
                "mfcc_sma[2]",
                "mfcc_sma[3]",
                "mfcc_sma[4]",
                "mfcc_sma[5]",
                "mfcc_sma[6]",
                "mfcc_sma[7]",
                "mfcc_sma[8]",
                "mfcc_sma[9]",
                "mfcc_sma[10]",
                "mfcc_sma[11]",
                "mfcc_sma[12]",
                "mfcc_sma[13]",
                "mfcc_sma[14]",
            ]

        elif feature_type == "compare_2016_mfcc":
            opensmile_feats_subset["subset"] = [
                "mfcc_sma[1]",
                "mfcc_sma[2]",
                "mfcc_sma[3]",
                "mfcc_sma[4]",
                "mfcc_sma[5]",
                "mfcc_sma[6]",
                "mfcc_sma[7]",
                "mfcc_sma[8]",
                "mfcc_sma[9]",
                "mfcc_sma[10]",
                "mfcc_sma[11]",
                "mfcc_sma[12]",
                "mfcc_sma[13]",
                "mfcc_sma[14]",
            ]

        elif feature_type == "compare_2016_rasta":
            opensmile_feats_subset["subset"] = [
                "audSpec_Rfilt_sma[0]",
                "audSpec_Rfilt_sma[1]",
                "audSpec_Rfilt_sma[2]",
                "audSpec_Rfilt_sma[3]",
                "audSpec_Rfilt_sma[4]",
                "audSpec_Rfilt_sma[5]",
                "audSpec_Rfilt_sma[6]",
                "audSpec_Rfilt_sma[7]",
                "audSpec_Rfilt_sma[8]",
                "audSpec_Rfilt_sma[9]",
                "audSpec_Rfilt_sma[10]",
                "audSpec_Rfilt_sma[11]",
                "audSpec_Rfilt_sma[12]",
                "audSpec_Rfilt_sma[13]",
                "audSpec_Rfilt_sma[14]",
                "audSpec_Rfilt_sma[15]",
                "audSpec_Rfilt_sma[16]",
                "audSpec_Rfilt_sma[17]",
                "audSpec_Rfilt_sma[18]",
                "audSpec_Rfilt_sma[19]",
                "audSpec_Rfilt_sma[20]",
                "audSpec_Rfilt_sma[21]",
                "audSpec_Rfilt_sma[22]",
                "audSpec_Rfilt_sma[23]",
                "audSpec_Rfilt_sma[24]",
                "audSpec_Rfilt_sma[25]",
            ]

        elif feature_type == "compare_2016_basic_spectral":
            opensmile_feats_subset = [
                "pcm_fftMag_fband250-650_sma",
                "pcm_fftMag_fband1000-4000_sma",
                "pcm_fftMag_spectralRollOff25.0_sma",
                "pcm_fftMag_spectralRollOff50.0_sma",
                "pcm_fftMag_spectralRollOff75.0_sma",
                "pcm_fftMag_spectralRollOff90.0_sma",
                "pcm_fftMag_spectralFlux_sma",
                "pcm_fftMag_spectralCentroid_sma",
                "pcm_fftMag_spectralEntropy_sma",
                "pcm_fftMag_spectralVariance_sma",
                "pcm_fftMag_spectralSkewness_sma",
                "pcm_fftMag_spectralKurtosis_sma",
                "pcm_fftMag_spectralSlope_sma",
                "pcm_fftMag_psySharpness_sma",
                "pcm_fftMag_spectralHarmonicity_sma",
            ]

        else:
            opensmile_feats_subset = None

        if opensmile_feats_subset is not None:
            # Select only the columns in the subset if they exist in the extracted DataFrame.
            existing_cols = [col for col in opensmile_feats_subset if col in acoustic_feats.columns]
            if existing_cols:
                opensmile_feats_subset = acoustic_feats[existing_cols]
                opensmile_feats_subset.reset_index(drop=True, inplace=True)
                return opensmile_feats_subset
            else:
                logger.warning("None of the requested subset features were found in the extracted acoustic features.")
                return None

    def postprocessor(self, acoustic_feats: pd.DataFrame) -> pd.DataFrame:
        """ Postprocesses the acoustic features. Optionally computes delta and delta-delta features. """
        post_feats = self.opensmile_subset_selector(acoustic_feats)
        if post_feats is None:
            post_feats = acoustic_feats

        try:
            tensor_array = from_numpy(post_feats.to_numpy())
            if self.config.compute_deltas:
                deltas = compute_deltas(tensor_array)
                df_deltas = pd.DataFrame(deltas)
                # Add columns name as delta-i
                df_deltas.rename(columns={i: f"delta-{i}" for i in range(deltas.shape[1])}, inplace=True)

                if self.config.compute_deltas_deltas:
                    deltas_deltas = compute_deltas(deltas)
                    # Add columns name as delta-delta-i
                    df_deltas_deltas = pd.DataFrame(deltas_deltas)
                    df_deltas_deltas.rename(columns={i: f"delta-delta{i}" for i in range(deltas_deltas.shape[1])}, inplace=True)

                    df_deltas = pd.concat([df_deltas, df_deltas_deltas], axis=1)
                    df_deltas.reset_index(drop=True, inplace=True)

                post_feats = pd.concat([post_feats, df_deltas], axis=1)
        except Exception as e:
            logger.error(f"Error during delta and delta-delta calculation: {e}")
            raise RuntimeError

        return post_feats
