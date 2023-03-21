import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram, MelSpectrogram, AmplitudeToDB, MFCC
from torchlibrosa.augmentation import SpecAugmentation

from models.Cnn6 import Cnn6
from models.Cnn10 import Cnn10
from models.Cnn10Att import Cnn10Att


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AudioFrontend(nn.Module):
    def __init__(self, feat_config):

        super(AudioFrontend, self).__init__()

        self.feat_config = feat_config

        transform = feat_config["transform"]

        if transform == "stftw":
            spectrogram_extractor = Spectrogram(n_fft=feat_config["n_fft"], hop_length=feat_config["hop_length"],
                                                power=1.0, center=feat_config["center"]).to(device)  # added
            freq_bins = (1 + feat_config["n_fft"] / 2)
            res = (0.5 * feat_config["sr"]) / freq_bins
            start_idx = int(feat_config["fmin"] / res)
            end_idx = int(feat_config["fmax"] / res)
            self.extractor = lambda x: spectrogram_extractor(x)[:, start_idx:end_idx]
            self.n_feats = end_idx - start_idx
            print(self.n_feats, start_idx, end_idx)
        elif transform == "logmel":
            melspectrogram_extractor = MelSpectrogram(sample_rate=feat_config["sr"],
                                                      n_fft=feat_config["n_fft"],
                                                      hop_length=feat_config["hop_length"],
                                                      f_min=feat_config["fmin"],
                                                      f_max=feat_config["fmax"],
                                                      n_mels=feat_config["n_mels"],
                                                      center=feat_config["center"],
                                                      norm='slaney',
                                                      mel_scale='slaney').to(device)  # added
            db_scale = AmplitudeToDB(top_db=80.0).to(device)  # added
            self.extractor = lambda x: db_scale(melspectrogram_extractor(x))
            self.n_feats = feat_config["n_mels"]
        elif transform == "mfcc":
            self.extractor = MFCC(sample_rate=feat_config["sr"],
                                  n_mfcc=feat_config["n_mfcc"],
                                  melkwargs=dict(
                                      n_fft=feat_config["n_fft"],
                                      hop_length=feat_config["hop_length"],
                                      f_min=feat_config["fmin"],
                                      f_max=feat_config["fmax"],
                                      n_mels=feat_config["n_mels"],
                                      norm='slaney',
                                      mel_scale='slaney'))
            self.n_feats = feat_config["n_mfcc"]
        else:
            print("Not supported")

    @property
    def n_feats(self):
        return self._n_feats

    @n_feats.setter
    def n_feats(self, val):
        self._n_feats = val

    def forward(self, x):
        """
        Input: (batch_size, data_length)"""

        out = self.extractor(x).unsqueeze(1)
        return out


def feature_pooling(x):
    x = torch.mean(x, dim=3)  # Average over mel-bins
    (x1, _) = torch.max(x, dim=2)  # Max over time
    x2 = torch.mean(x, dim=2)  # Average over time
    x = x1 + x2

    return x


def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class Classifier(nn.Module):
    def __init__(self, n_feats, fc_features, classes_num, dropout):
        super(Classifier, self).__init__()

        self.dropout = dropout
        self.fc1 = nn.Linear(n_feats, fc_features, bias=True)
        self.fc2 = nn.Linear(fc_features, classes_num, bias=True)

        self.init_weight()

    def init_weight(self):
        init_layer(self.fc1)
        init_layer(self.fc2)

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)

        # x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.dropout, training=self.training)

        out = torch.sigmoid(self.fc2(x))

        return out


class SampleModel(nn.Module):
    def __init__(self, feature_model, n_feats=64, conv_channels=32, pos_emb_dim=0, fc_features=64,
                 classes_num=1, conv_dropout=0.2, fc_dropout=0.5, feat_config=None, in_lambda=0.1,
                 time_drop_width=10, freq_drop_width=4):

        super(SampleModel, self).__init__()

        if feat_config is not None:
            self.audio_frontend = AudioFrontend(feat_config)
            n_feats = self.audio_frontend.n_feats

        self.bn = nn.BatchNorm2d(n_feats)

        # time_drop_width = 10  # time_drop_width=64 / 100 * (64 / 1000) = 6.4% --> 157*0.064 = 10.048
        # freq_drop_width = n_feats // 8  # freq_drop_width=8 / 100 * (8 / 64) = 12.5%  --> SAME
        self.spec_augmenter = SpecAugmentation(time_drop_width=time_drop_width, time_stripes_num=2,
                                               freq_drop_width=freq_drop_width, freq_stripes_num=2)

        self.feature_model = feature_model

        if feature_model == "cnn6":
            self.features = Cnn6(conv_channels=conv_channels, dropout=conv_dropout)
        elif feature_model == "cnn10":
            self.features = Cnn10(conv_channels=conv_channels, dropout=conv_dropout)
        elif feature_model == "dense":
            self.features = Cnn10Att(classes_num=classes_num)

        self.classifier = Classifier(n_feats=8*conv_channels, fc_features=fc_features,
                                     classes_num=classes_num, dropout=fc_dropout)

        self.init_weight()

    def init_weight(self):
        init_bn(self.bn)

    def preprocess(self, x):
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

    def forward(self, x):

        x = self.preprocess(x)

        if self.feature_model == "dense":
            out = self.features(x)
            output_dict = {
                "diagnosis_output": out["clipwise_output"],
                "framewise_output": out["framewise_output"],
                "framewise_att": out["framewise_att"]
            }
        else:
            sample_features = self.features(x)
            sample_features = feature_pooling(sample_features)

            out = sample_features

            diagnosis_out = self.classifier(out)

            output_dict = {
                "diagnosis_output": diagnosis_out,
                "embedding": sample_features
            }

        return output_dict

