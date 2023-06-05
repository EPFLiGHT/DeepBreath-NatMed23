from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import librosa.display
import numpy as np
from numpy import ndarray
from scipy import signal


def read_audio(filename: str, offset: float = 0.0, duration: Optional[float] = None):
    samples, sr = librosa.load(filename, sr=None, offset=offset, duration=duration)
    return samples


def get_duration(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    duration = librosa.get_duration(y=samples, sr=config["sr"])
    return duration


# ---------- Pre-processing ---------- #


# Lowpass filter
def butter_lowpass(
    config: Dict[str, Union[int, float, List[str], str, bool]]
) -> Tuple[ndarray, ndarray]:
    nyq = 0.5 * config["sr"]
    low = config["freq_lowcut"] / nyq
    b, a = signal.butter(config["order_lowcut"], low, btype="low")
    return b, a


def lowpass_filter(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
) -> ndarray:
    b, a = butter_lowpass(config)
    samples_filtered = signal.lfilter(b, a, samples)
    return samples_filtered


# Highpass filter
def butter_highpass(
    config: Dict[str, Union[int, float, List[str], str, bool]]
) -> Tuple[ndarray, ndarray]:
    nyq = 0.5 * config["sr"]
    high = config["freq_highcut"] / nyq
    b, a = signal.butter(config["order_highcut"], high, btype="high")
    return b, a


def highpass_filter(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
) -> ndarray:
    b, a = butter_highpass(config)
    samples_filtered = signal.lfilter(b, a, samples)
    return samples_filtered


# Bandpass filter
def butter_bandpass(config: Dict[str, Union[int, float, List[str], str, bool]]):
    nyq = 0.5 * config["sr"]
    low = config["freq_lowcut"] / nyq
    high = config["freq_highcut"] / nyq
    b, a = signal.butter(
        min(config["order_lowcut"], config["order_highcut"]), [low, high], btype="band"
    )
    return b, a


def bandpass_filter(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    b, a = butter_bandpass(config)
    samples_filtered = signal.lfilter(b, a, samples)
    return samples_filtered


# ---------- Audio Features ---------- #


def compute_stft(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    stft = librosa.stft(
        samples,
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        center=config["center"],
    )
    S = np.abs(stft)

    return S


def compute_stft_windowed(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    S = compute_stft(samples, config)

    res = config["sr"] / (2 * S.shape[0])
    start_idx = int(config["fmin"] / res)
    end_idx = int(config["fmax"] / res)

    S = S[start_idx:end_idx]

    return S


def compute_mel_spectrogram(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    melspect = librosa.feature.melspectrogram(
        y=samples,
        sr=config["sr"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        center=config["center"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
    )
    return melspect


def log_spectrogram(spectrogram, ref=1.0):
    logspect = librosa.core.power_to_db(spectrogram, ref=ref)
    return logspect


def compute_logmel(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    melspect = compute_mel_spectrogram(samples, config)
    logmel = log_spectrogram(melspect)
    return logmel


def compute_mfcc(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    mfccs = librosa.feature.mfcc(
        y=samples,
        sr=config["sr"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        center=config["center"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
        n_mfcc=config["n_mfcc"],
    )
    return mfccs


def mfcc_delta(mfccs, order):
    delta = librosa.feature.delta(mfccs, order=order)
    return delta


def compute_delta1(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    mfccs = compute_mfcc(samples, config=config)
    delta1 = mfcc_delta(mfccs, 1)
    return delta1


def compute_delta2(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    mfccs = compute_mfcc(samples, config=config)
    delta2 = mfcc_delta(mfccs, 2)
    return delta2


def compute_zcr(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    return librosa.feature.zero_crossing_rate(
        y=samples, frame_length=config["n_fft"], hop_length=config["hop_length"]
    )


def compute_spectral_centroid(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    return librosa.feature.spectral_centroid(
        y=samples,
        sr=config["sr"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
    )


def compute_spectral_rolloff(
    samples: ndarray, config: Dict[str, Union[int, float, List[str], str, bool]]
):
    return librosa.feature.spectral_rolloff(
        y=samples,
        sr=config["sr"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        roll_percent=config["roll_percent"],
    )


# ---------- Feature Pipeline ---------- #

preprocessing_fcts = {
    "lowpass": lowpass_filter,
    "highpass": highpass_filter,
    "bandpass": bandpass_filter,
}
feature_fcts = {
    "stft": compute_stft,
    "stftw": compute_stft_windowed,
    "mel": compute_mel_spectrogram,
    "logmel": compute_logmel,
    "mfcc": compute_mfcc,
    "delta1": compute_delta1,
    "delta2": compute_delta2,
    "zcr": compute_zcr,
    "centroid": compute_spectral_centroid,
    "rolloff": compute_spectral_rolloff,
}


class AudioFeatures:
    def __init__(
        self,
        features: List[Any],
        config: Dict[str, Union[int, float, List[str], str, bool]],
    ) -> None:
        for pre in config["preprocessing"]:
            assert pre in preprocessing_fcts.keys()
        for feat in features:
            assert feat in feature_fcts.keys()

        self.preprocessing = config["preprocessing"]
        self.features = features
        self.config = config

    def transform(self, samples: ndarray, time_first: bool = False) -> ndarray:
        x = samples
        for pre in self.preprocessing:
            x = preprocessing_fcts[pre](x, config=self.config)

        output = x
        if self.features:
            output = [
                feature_fcts[feat](x, config=self.config) for feat in self.features
            ]  # added

        if time_first:
            output = [out.T for out in output]

        return output
