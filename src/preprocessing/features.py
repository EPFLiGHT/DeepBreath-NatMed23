import librosa
import librosa.display
import numpy as np
from scipy import signal

from config import pre_config, feat_config


def read_audio(filename, offset=0.0, duration=None, config=feat_config):
    samples, sr = librosa.load(filename, sr=None, offset=offset, duration=duration)
    assert sr == config["sr"]
    return samples


def get_duration(samples, config=feat_config):
    duration = librosa.get_duration(y=samples, sr=config["sr"])
    return duration


# ---------- Pre-processing ---------- #

# Lowpass filter
def butter_lowpass(config):
    nyq = 0.5 * config["sr"]
    low = config["freq_lowcut"] / nyq
    b, a = signal.butter(config["order_lowcut"], low, btype='low')
    return b, a


def lowpass_filter(samples, config=pre_config):
    b, a = butter_lowpass(config)
    samples_filtered = signal.lfilter(b, a, samples)
    return samples_filtered


# Highpass filter
def butter_highpass(config):
    nyq = 0.5 * config["sr"]
    high = config["freq_highcut"] / nyq
    b, a = signal.butter(config["order_highcut"], high, btype='high')
    return b, a


def highpass_filter(samples, config=pre_config):
    b, a = butter_highpass(config)
    samples_filtered = signal.lfilter(b, a, samples)
    return samples_filtered


# Bandpass filter
def butter_bandpass(config):
    nyq = 0.5 * config["sr"]
    low = config["freq_lowcut"] / nyq
    high = config["freq_highcut"] / nyq
    b, a = signal.butter(min(config["order_lowcut"], config["order_highcut"]), [low, high], btype='band')
    return b, a


def bandpass_filter(samples, config=pre_config):
    b, a = butter_bandpass(config)
    samples_filtered = signal.lfilter(b, a, samples)
    return samples_filtered


# ---------- Audio Features ---------- #

def compute_stft(samples, config=feat_config):
    stft = librosa.stft(
        samples,
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        center=config["center"]
    )
    S = np.abs(stft)

    return S


def compute_stft_windowed(samples, config=feat_config):
    S = compute_stft(samples, config)

    res = config["sr"] / (2 * S.shape[0])
    start_idx = int(config["fmin"] / res)
    end_idx = int(config["fmax"] / res)

    S = S[start_idx:end_idx]

    return S


def compute_mel_spectrogram(samples, config=feat_config):
    melspect = librosa.feature.melspectrogram(
        samples,
        sr=config["sr"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        center=config["center"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"]
    )
    return melspect


def log_spectrogram(spectrogram, ref=1.0):
    logspect = librosa.core.power_to_db(spectrogram, ref=ref)
    return logspect


def compute_logmel(samples, config=feat_config):
    melspect = compute_mel_spectrogram(samples, config)
    logmel = log_spectrogram(melspect)
    return logmel


def compute_mfcc(samples, config=feat_config):
    mfccs = librosa.feature.mfcc(
        samples,
        sr=config["sr"],
        n_fft=config["n_fft"],
        hop_length=config["hop_length"],
        center=config["center"],
        n_mels=config["n_mels"],
        fmin=config["fmin"],
        fmax=config["fmax"],
        n_mfcc=config["n_mfcc"]
    )
    return mfccs


def mfcc_delta(mfccs, order):
    delta = librosa.feature.delta(mfccs, order=order)
    return delta


def compute_delta1(samples, config=feat_config):
    mfccs = compute_mfcc(samples, config=config)
    delta1 = mfcc_delta(mfccs, 1)
    return delta1


def compute_delta2(samples, config=feat_config):
    mfccs = compute_mfcc(samples, config=config)
    delta2 = mfcc_delta(mfccs, 2)
    return delta2


def compute_zcr(samples, config=feat_config):
    return librosa.feature.zero_crossing_rate(samples, frame_length=config["n_fft"], hop_length=config["hop_length"])


def compute_spectral_centroid(samples, config=feat_config):
    return librosa.feature.spectral_centroid(samples, sr=config["sr"], n_fft=config["n_fft"],
                                             hop_length=config["hop_length"])


def compute_spectral_rolloff(samples, config=feat_config):
    return librosa.feature.spectral_rolloff(samples, sr=config["sr"], n_fft=config["n_fft"],
                                            hop_length=config["hop_length"], roll_percent=config["roll_percent"])


# ---------- Feature Pipeline ---------- #

preprocessing_fcts = {"lowpass": lowpass_filter, "highpass": highpass_filter, "bandpass": bandpass_filter}
feature_fcts = {"stft": compute_stft, "stftw": compute_stft_windowed, "mel": compute_mel_spectrogram,
                "logmel": compute_logmel, "mfcc": compute_mfcc, "delta1": compute_delta1, "delta2": compute_delta2,
                "zcr": compute_zcr, "centroid": compute_spectral_centroid, "rolloff": compute_spectral_rolloff}


class AudioFeatures:
    def __init__(self, features, preprocessing=[], pre_config=pre_config, feat_config=feat_config):
        for pre in preprocessing:
            assert pre in preprocessing_fcts.keys()
        for feat in features:
            assert feat in feature_fcts.keys()

        self.preprocessing = preprocessing
        self.features = features
        self.pre_config = pre_config
        self.feat_config = feat_config

    def transform(self, samples, time_first=False):
        x = samples
        for pre in self.preprocessing:
            x = preprocessing_fcts[pre](x, config=self.pre_config)

        output = x
        if self.features:
            output = [feature_fcts[feat](x, config=self.feat_config) for feat in self.features]  # added

        if time_first:
            output = [out.T for out in output]

        return output
