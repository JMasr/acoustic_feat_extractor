import pickle
from pathlib import Path
from typing import Union

import pandas as pd
import torch
import torchaudio
from datasets import Dataset
from numpy import full, hstack, ndarray, vstack
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from speechbrain.inference import EncoderClassifier
from torch import Tensor

from gtm_feat.audio.base import BaseEmbeddingExtractor
from gtm_feat.bob.learn import em
from gtm_feat.config import MODELS_DIR, MODELS_PRETRAINED_DIR, logger


class IVectorExtractor(BaseEmbeddingExtractor):
    def __init__(self, config_object: Union[Path, dict]):
        super().__init__(config_object)
        self.load_model()

        self.transform = torchaudio.transforms.MFCC(
            sample_rate=self.config.get("resampling_rate", 16000),
            n_mfcc=self.config.get("n_mfcc", 13),
            log_mels=True,
            melkwargs={
                "n_fft": self.config.get("n_fft", 256),
                "hop_length": self.config.get("hop_length", 160),
                "n_mels": self.config.get("n_mels", 23),
                "center": False,
            },
        )

        knn_max_iter = self.config.get("knn_max_iter", 100)
        ubm_gaussians = self.config.get("ubm_gaussians", 16)
        ubm_max_fitting_steps = self.config.get("ubm_max_fitting_steps", 150)

        k_mean_trainer = em.KMeansMachine(ubm_gaussians, max_iter=knn_max_iter)
        self.ubm = em.GMMMachine(
            n_gaussians=ubm_gaussians,
            update_variances=True,
            k_means_trainer=k_mean_trainer,
            max_fitting_steps=ubm_max_fitting_steps,
            output_folder=self.config.get("model_local_path", MODELS_PRETRAINED_DIR),
        )

        ivector_dim = self.config.get("ivector_dim", 128)
        ivector_epochs = self.config.get("ivector_epochs", 10)

        self.ivector_machine = em.IVectorMachine(
            self.ubm,
            dim_t=ivector_dim * 2,
            max_iterations=ivector_epochs,
            output_folder=self.config.get("model_local_path", MODELS_PRETRAINED_DIR),
        )
        self.pca_machine = PCA(n_components=int(ivector_dim), whiten=True)

        self.lda_machine = LinearDiscriminantAnalysis()

        self.load_model()

    def load_model(self) -> bool:
        dict_models: list = ["ubm", "lda_machine", "pca_machine", "ivector_machine"]

        for model_name in dict_models:
            tag = "-" + self.config.get("tag", "custom")
            output_folder = self.config.get("origin_model", MODELS_PRETRAINED_DIR)
            output_model_path = output_folder / f"{model_name}{tag}.pt"

            try:
                if output_model_path.exists():
                    with open(output_model_path, "rb") as f:
                        model = pickle.load(f)
                        setattr(self, model_name, model)
                    logger.info(f"Model {model_name} loaded from {output_folder}")

                else:
                    logger.error(f"Model {model_name} not found in {output_folder}")
                    return False
            except Exception as e:
                logger.error(f"Error loading models: {e}")
                return False

        return True

    def save_model(self):
        """Save a model to disk."""

        tag = "-" + self.config.get("tag", "custom")
        path: Path = self.config.model_local_path
        path.mkdir(parents=True, exist_ok=True)

        dict_models = {
            "ubm": self.ubm,
            "lda_machine": self.lda_machine,
            "pca_machine": self.pca_machine,
            "ivector_machine": self.ivector_machine,
        }

        for name, model in dict_models.items():
            try:
                output_model_path = path / f"{name}{tag}.pt"
                with open(output_model_path, "wb") as f:
                    pickle.dump(model, f)
                logger.info(f"Models saved to: {path}")
            except Exception as e:
                logger.error(f"Model Path: {path}, Model Type: {name}")
                raise RuntimeError(f"Error saving models: {e}")

        return True

    def preprocessor(self, raw_audio_path: Path) -> Tensor:
        sample, sr = torchaudio.load(raw_audio_path, normalize=True)
        sample = torchaudio.functional.resample(
            sample, sr, self.config.get("resampling_rate", 1600)
        )

        mfcc = self.transform(sample)
        return mfcc.squeeze().T

    def feature_transform(self, pre_audio: Tensor) -> ndarray:
        gmm_stat_mfcc = self.ubm.acc_stats(pre_audio)

        i_vector = self.ivector_machine.transform([gmm_stat_mfcc])
        i_vector = self.pca_machine.transform(i_vector)
        return i_vector

    def postprocessor(self, acoustic_feats: ndarray) -> DataFrame:
        df_ivector = pd.DataFrame(acoustic_feats)
        df_ivector.rename(
            columns={i: f"ivector_{i}" for i in range(acoustic_feats.shape[1])},
            inplace=True,
        )
        return df_ivector

    def train(self, data_train: DataFrame):

        def extract_mfcc_features(df_train_data: DataFrame) -> Dataset:
            """
            Extract MFCC features for all audio files in the dataset.

            :param df_train_data: The dataset containing audio file metadata.
            :return: A dictionary mapping audio IDs to MFCC features.
            """

            def process_function(example):
                path = example["path"]

                if isinstance(path, Path):
                    path: str = path.name

                try:
                    sample, sr = torchaudio.load(path, normalize=True)
                    sample = torchaudio.functional.resample(
                        sample, sr, self.config.get("resampling_rate", 1600)
                    )

                    mfcc = self.transform(sample)
                    mfcc = mfcc.squeeze().T

                    label = full(mfcc.shape[0], example["label"])
                    return {"feat": mfcc, "label": label, "id": example["id"]}
                except Exception as e:
                    logger.warning(f"Audio Corrupted: {path}, Error: {e}")

            dataset_ = Dataset.from_pandas(df_train_data)
            return dataset_.map(process_function)

        if data_train is None:
            raise RuntimeError("You dont pass a dataset for train the UBM models.")

        logger.info("1- Extracting MFCC features")
        dataset: Dataset = extract_mfcc_features(data_train)
        feat_train = vstack(dataset["feat"]).squeeze()
        label_train = hstack(dataset["label"]).squeeze()

        logger.info("2- Training the UBM")
        self.ubm.fit(feat_train)

        logger.info("2.1 - Project the training features on the UBM trained")
        feat_statistics_train = self.ubm.transform(feat_train)

        logger.info("3- Training the i-vector machine")
        self.ivector_machine.fit(feat_statistics_train)

        # Project the GMM-stat of train feat on  i-vector space
        ivectors_train = self.ivector_machine.transform(feat_statistics_train)
        ivectors_train = vstack(ivectors_train)

        # Apply PCA over the LDA reduce vectors
        logger.info("4- Post-processing")
        ivectors_pca_train = self.pca_machine.fit_transform(ivectors_train)
        # Train an LDA model with the train i-vectors
        self.lda_machine.fit(ivectors_pca_train, label_train)

        logger.info("5- Saving the models")
        self.save_model()

        logger.info("Training completed. Models and outputs saved to:", MODELS_DIR)
        logger.info(
            " IVector Machine Parameters:"
            f"* UBM Gaussians: {self.ubm.n_gaussians},"
            f"* i-vector dimension: {self.ivector_machine.dim_t},"
            f"* E-M max iterations: {self.ivector_machine.max_iterations}"
        )


class XVectorExtractor(BaseEmbeddingExtractor):
    def __init__(self, config_object: Union[Path, dict]):
        super().__init__(config_object)
        self.x_vector_transform: EncoderClassifier = None
        self.load_model()

    def load_model(self) -> bool:
        try:
            self.x_vector_transform = EncoderClassifier.from_hparams(
                source=self.config.origin_model,
                savedir=self.config.model_local_path,
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def save_model(self):
        try:
            torch.save(self.x_vector_transform.state_dict(), self.config.model_local_path)
            logger.info("Model saved")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            raise IOError

    def preprocessor(self, raw_audio_path: Path) -> Tensor:
        speech_array, sr = torchaudio.load(raw_audio_path, normalize=True)
        speech_array = torchaudio.functional.resample(speech_array, sr, self.config.resampling_rate)

        return speech_array

    def feature_transform(self, pre_audio: torch.Tensor) -> torch.Tensor:
        x_vector = self.x_vector_transform.encode_batch(pre_audio).squeeze()
        return x_vector

    def postprocessor(self, acoustic_feats: torch.Tensor) -> pd.DataFrame:
        df_xvector = pd.DataFrame(acoustic_feats.numpy())
        df_xvector.rename(
            columns={i: f"xvector{i}" for i in range(len(acoustic_feats))},
            inplace=True,
        )
        return df_xvector

    def train(self, df_custom_data: pd.DataFrame):
        logger.warning("Trainin X-Vector isnt implemented yet.")
