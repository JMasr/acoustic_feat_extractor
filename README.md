# GTM - Acoustic Feature Extractor

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

<a target="_blank" href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-black?logo=PyTorch" />
</a>


## General Considerations

This project is designed to be a comprehensive acoustic feature extractor with a focus on flexibility and modularity.
Support for various feature extraction methods, including **short-term features** and **embeddings**, is provided.

The two data types supported for features are **torch.Tensor** and **numpy.ndarray**.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         gtm_feat and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── gtm_feat   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes gtm_feat a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Acoustic Features
Acoustic features help analyze and understand audio signals, particularly in speech and sound processing.
These features can be broadly categorized into two types:
- **Short-Term Features:** These are extracted from short segments of audio (typically in milliseconds) 
and capture local characteristics of the sound, such as spectral and temporal properties.
Examples include **Mel-Frequency Cepstral Coefficients (MFCCs)** and **Perceptual Linear Prediction (PLP)**,
which are widely used in speech and audio recognition tasks.

- **Embeddings:** These provide a more holistic, long-term representation of an audio signal,
typically encoding high-level characteristics over extended timeframes. They often utilize deep learning models,
such as **wav2vec** or **x-vector**, to create feature-rich representations useful for tasks like speaker identification,
emotion recognition, and sound classification.
- 
### Short-Term Features
**Configuration Schema:**

```
{
    "feature_type": "compare_2016_voicing",
    "top_db": 30,
    "pre_emphasis_coefficient": 0.97,
    "resampling_rate": 44100,
    "n_mels": 64,
    "n_mfcc": 32,
    "plp_order": 13,
    "conversion_approach": "Wang",
    "f_max": 22050,
    "f_min": 100,
    "window_size": 25.0,
    "hop_length": 10.0,
    "window_type": "hamming",
    "use_energy": false,
    "apply_mean_norm": false,
    "apply_vari_norm": false,
    "compute_deltas_feats": false,
    "compute_deltas_deltas_feats": false,
    "compute_opensmile_extra_features": false,
}
```  

#### Cepstral Features
- **`mfcc`** – **Mel-Frequency Cepstral Coefficients (MFCCs)**, widely used in speech recognition and speaker identification.
- **`imfcc`** – Inverted **MFCCs**, offering an alternative frequency representation for robust speech analysis.
- **`cqcc`** – **Constant-Q Cepstral Coefficients (CQCCs)**, designed to handle signals with varying frequency resolutions.
- **`gfcc`** – **Gammatone Frequency Cepstral Coefficients (GFCCs)**, inspired by auditory models for human hearing.
- **`lfcc`** – **Linear-Frequency Cepstral Coefficients (LFCCs)**, an alternative to MFCCs that uses a linear frequency scale.
- **`msrcc`** – **Modulation Spectrum Cepstral Coefficients (MSRCCs)**, capturing temporal modulations of speech for robust feature extraction.
- **`ngcc`** – **Normalized Group Delay Cepstral Coefficients (NGCCs)**, useful for speaker and speech recognition.
- **`pncc`** – **Power-Normalized Cepstral Coefficients (PNCCs)**, designed for robustness in noisy environments.
- **`psrcc`** – **Phase-based Spectral Root Cepstral Coefficients (PSRCCs)**, incorporating phase spectrum information for improved accuracy.

#### Linear Prediction Features
- **`lpc`** – **Linear Predictive Coding (LPC)**, used for estimating speech production parameters.
- **`lpcc`** – **Linear Predictive Cepstral Coefficients (LPCCs)**, derived from LPC for enhanced speech recognition.

#### Perceptual Features
- **`plp`** – **Perceptual Linear Prediction (PLP)**, mimics human auditory perception for speech analysis.
- **`rplp`** – **Relative Perceptual Linear Prediction (RPLP)**, a variant of PLP designed to improve feature robustness.

#### OpenSmile

 - **`compare_2016_voicing`** – Measures voice-related characteristics, such as pitch and periodicity, to analyze speech patterns.
 - **`compare_2016_energy`** – Captures variations in signal energy, helping assess loudness, intensity, and emphasis in speech.
 - **`compare_2016_llds`** – Extracts low-level descriptors (**LLDs**) related to fundamental frequency, energy, and spectral properties.
 - **`compare_2016_spectral`** – Represents spectral features that describe the frequency distribution of an audio signal, useful for timbre analysis.
 - **`compare_2016_mfcc`** – Computes **Mel-Frequency Cepstral Coefficients (MFCCs)**, a widely used feature set in speech and speaker recognition.
 - **`compare_2016_rasta`** – Extracts **RASTA (RelAtive Spectral Transform)** features, designed to improve robustness against channel distortions.
 - **`compare_2016_basic_spectral`** – Provides fundamental spectral features, such as spectral centroid, bandwidth, and flux, commonly used in sound classification.
