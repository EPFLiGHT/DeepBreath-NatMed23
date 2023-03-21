import torch.nn as nn
from torch.nn import functional as F


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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class Cnn10(nn.Module):
    def __init__(self, conv_channels=64, dropout=0.2):
        super(Cnn10, self).__init__()

        self.dropout = dropout
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=conv_channels)
        self.conv_block2 = ConvBlock(in_channels=conv_channels, out_channels=2*conv_channels)
        self.conv_block3 = ConvBlock(in_channels=2*conv_channels, out_channels=4*conv_channels)
        self.conv_block4 = ConvBlock(in_channels=4*conv_channels, out_channels=8*conv_channels)

    def forward(self, x):
        """
        Input: (batch_size, data_length)"""

        # x = self.spectrogram_extractor(input)   # (batch_size, 1, time_steps, freq_bins)
        # x = self.logmel_extractor(x)    # (0: batch_size, 1: 1, 2: time_steps, 3: mel_bins)

        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x
