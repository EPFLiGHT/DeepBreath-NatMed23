# This code was adapted from https://github.com/ryanwongsa/kaggle-birdsong-recognition/blob/master/src/models/sed_models.py,
# which is licensed under the MIT license. The original author of this code is Ryan Wong (https://github.com/ryanwongsa).

from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d, BatchNorm2d
from torch.nn.modules.conv import Conv1d, Conv2d
from torch.nn.modules.linear import Linear


def init_layer(layer: Union[Linear, Conv1d, Conv2d]) -> None:
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn: Union[BatchNorm2d, BatchNorm1d]) -> None:
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def interpolate(x: torch.Tensor, ratio: int) -> torch.Tensor:
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(
    framewise_output: torch.Tensor, frames_num: int
) -> torch.Tensor:
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""

    return output


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(
        self,
        input: torch.Tensor,
        pool_size: Tuple[int, int] = (2, 2),
        pool_type: str = "avg",
    ) -> torch.Tensor:
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == "max":
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg":
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == "avg+max":
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception("Incorrect argument!")

        return x


class AttBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: str = "linear",
        temperature: float = 1.0,
    ) -> None:
        super().__init__()

        self.activation = activation
        self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_weights(self) -> None:
        init_layer(self.att)
        init_layer(self.cla)
        init_bn(self.bn_att)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class Cnn10Att(nn.Module):
    def __init__(self, classes_num: int) -> None:
        super().__init__()

        self.interpolate_ratio = 16  # Downsampled ratio: 16 32

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)

        self.fc1 = nn.Linear(1024, 1024, bias=True)
        self.att_block = AttBlock(
            1024, classes_num, activation="linear"
        )  # 'sigmoid' 'linear'

        self.init_weight()

    def init_weight(self) -> None:
        init_layer(self.fc1)

    def cnn_feature_extractor(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(1, 1), pool_type="avg")
        x = F.dropout(x, p=0.2, training=self.training)
        return x

    def forward(
        self, x: torch.Tensor
    ) -> Dict[str, torch.Tensor]:  # input, mixup_lambda=None
        """
        Input: (0: batch size, 1: channels, 2: time, 3: frequency)"""

        frames_num = x.shape[2]

        x = self.cnn_feature_extractor(x)

        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)

        x1 = F.max_pool1d(x, kernel_size=3, stride=1, padding=1)
        x2 = F.avg_pool1d(x, kernel_size=3, stride=1, padding=1)
        x = x1 + x2

        x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(1, 2)
        x = F.relu_(self.fc1(x))
        x = x.transpose(1, 2)
        x = F.dropout(x, p=0.5, training=self.training)

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)

        clipwise_output = torch.sigmoid(clipwise_output)  # added
        segmentwise_output = torch.sigmoid(segmentwise_output)  # added

        segmentwise_output = segmentwise_output.transpose(1, 2)
        norm_att = norm_att.transpose(1, 2)  # added

        # Get framewise output
        framewise_output = interpolate(segmentwise_output, self.interpolate_ratio)
        framewise_output = pad_framewise_output(
            framewise_output, frames_num
        ).squeeze()  # added squeeze

        framewise_att = interpolate(norm_att, self.interpolate_ratio)  # added
        framewise_att = pad_framewise_output(
            framewise_att, frames_num
        ).squeeze()  # added

        output_dict = {
            "diagnosis_output": clipwise_output,
            "framewise_output": framewise_output,
            "framewise_att": framewise_att,
        }

        return output_dict
