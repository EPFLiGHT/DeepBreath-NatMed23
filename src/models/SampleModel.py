from typing import Dict

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torchaudio.transforms import Spectrogram, MelSpectrogram, AmplitudeToDB, MFCC
from torchlibrosa.augmentation import SpecAugmentation

from .Cnn10Att import Cnn10Att
from utils.arguments import AudioArguments, ModelArguments


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AudioFrontend(nn.Module):
    def __init__(self, config: AudioArguments) -> None:
        super(AudioFrontend, self).__init__()

        self.config = config

        transform = config.transform

        if transform == "stftw":
            spectrogram_extractor = Spectrogram(
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                power=1.0,
                center=config.center,
            ).to(
                device
            )  # added
            freq_bins = 1 + config.n_fft / 2
            res = (0.5 * config.sr) / freq_bins
            start_idx = int(config.fmin / res)
            end_idx = int(config.fmax / res)
            self.extractor = lambda x: spectrogram_extractor(x)[:, start_idx:end_idx]
            self.n_feats = end_idx - start_idx
            print(self.n_feats, start_idx, end_idx)
        elif transform == "logmel":
            melspectrogram_extractor = MelSpectrogram(
                sample_rate=config.sr,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                f_min=config.fmin,
                f_max=config.fmax,
                n_mels=config.n_mels,
                center=config.center,
                norm="slaney",
                mel_scale="slaney",
            ).to(
                device
            )  # added
            db_scale = AmplitudeToDB(top_db=80.0).to(device)  # added
            self.extractor = lambda x: db_scale(melspectrogram_extractor(x))
            self.n_feats = config.n_mels
        elif transform == "mfcc":
            self.extractor = MFCC(
                sample_rate=config.sr,
                n_mfcc=config.n_mfcc,
                melkwargs=dict(
                    n_fft=config.n_fft,
                    hop_length=config.hop_length,
                    f_min=config.fmin,
                    f_max=config.fmax,
                    n_mels=config.n_mels,
                    norm="slaney",
                    mel_scale="slaney",
                ),
            )
            self.n_feats = config.n_mfcc
        else:
            print("Not supported")

    @property
    def n_feats(self):
        return self._n_feats

    @n_feats.setter
    def n_feats(self, val):
        self._n_feats = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch_size, data_length)"""

        out = self.extractor(x).unsqueeze(1)
        return out


def init_bn(bn: BatchNorm2d) -> None:
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


class SampleModel(nn.Module):
    def __init__(
        self, audio_config: AudioArguments, model_config: ModelArguments
    ) -> None:
        super(SampleModel, self).__init__()

        self.audio_frontend = AudioFrontend(config=audio_config)
        n_feats = self.audio_frontend.n_feats

        self.bn = nn.BatchNorm2d(n_feats)

        # time_drop_width = 10  # time_drop_width=64 / 100 * (64 / 1000) = 6.4% --> 157*0.064 = 10.048
        # freq_drop_width = n_feats // 8  # freq_drop_width=8 / 100 * (8 / 64) = 12.5%  --> SAME
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=audio_config.time_drop_width,
            time_stripes_num=2,
            freq_drop_width=audio_config.freq_drop_width,
            freq_stripes_num=2,
        )

        self.features = Cnn10Att(classes_num=model_config.classes_num)

        self.init_weight()

    def init_weight(self) -> None:
        init_bn(self.bn)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Audio Frontend: Convert Audio to Image
        if hasattr(self, "audio_frontend"):
            x = self.audio_frontend(x)

        # Added: transpose time and freq dimensions
        x = x.transpose(2, 3)

        # 2. Batch Normalization and Spectral Augmentation (Added Instance Normalization)
        x = x.transpose(1, 3)
        x = self.bn(x)
        x = x.transpose(1, 3)

        if self.training:
            x = self.spec_augmenter(x)

        return x

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Audio Preprocessing
        x = self.preprocess(x)

        # Prediction and Feature Extraction
        out = self.features(x)

        return out
