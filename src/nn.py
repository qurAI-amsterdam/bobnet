# coding=utf-8
# Copyright: (c) 2024, Amsterdam University Medical Centers
# Apache License, Version 2.0, (see LICENSE or http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch
from torch import nn


class PyramidPooling2d(nn.Module):
    """
    Pyramid Pooling Module for 2D feature maps.

    :param pool_dims: list of pooling dimensions
    :type pool_dims: list
    """

    def __init__(self, pool_dims):
        super().__init__()
        self.pooling_pyramid = nn.ModuleList([nn.AdaptiveMaxPool2d((pdim, pdim)) for pdim in pool_dims])

    def forward(self, x):
        """
        Forward pass of the module.

        :param x: input tensor
        :type x: torch.Tensor
        """
        nsamples = x.shape[0]
        return torch.cat(tuple(pool(x).view(nsamples, -1) for pool in self.pooling_pyramid), 1)


def bnblock(inchans, outchans, **kwargs):
    """
    Convolutional block with batch normalization.

    :param inchans: number of input channels
    :type inchans: int
    :param outchans: number of output channels
    :type outchans: int
    :return: convolutional block
    :rtype: nn.Sequential
    """
    return nn.Sequential(nn.Conv2d(inchans, outchans, bias=False, **kwargs), nn.BatchNorm2d(outchans))


class BoBNet(nn.Module):
    """
    Bounding Box Network.

    :param out_classes: number of output classes
    :type out_classes: int
    """

    def __init__(self, out_classes=1):
        super(BoBNet, self).__init__()

        self.conv = bnblock

        # Encoder
        self.encodeLayers = nn.ModuleList()
        self.encodeLayers.append(self.conv(1, 16, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(nn.MaxPool2d(2, 2))
        self.encodeLayers.append(self.conv(16, 32, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(nn.MaxPool2d(2, 2))
        self.encodeLayers.append(self.conv(32, 64, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(self.conv(64, 64, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(nn.MaxPool2d(2, 2))
        self.encodeLayers.append(self.conv(64, 128, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(self.conv(128, 128, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(nn.MaxPool2d(2, 2))
        self.encodeLayers.append(self.conv(128, 128, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())
        self.encodeLayers.append(self.conv(128, 128, kernel_size=3, padding=1))
        self.encodeLayers.append(nn.ReLU())

        # Pooling
        pool_dims = [4, 2, 1]
        self.pool = PyramidPooling2d(pool_dims)

        # Classifier
        self.classLayers = nn.ModuleList()
        nodes = 128 * np.sum(np.square([4, 2, 1]))
        self.classLayers.append(nn.Linear(nodes, 128, bias=False))
        self.classLayers.append(nn.BatchNorm1d(128))
        self.classLayers.append(nn.ReLU())
        self.classLayers.append(nn.Dropout(0.5))
        self.classLayers.append(nn.Linear(128, 128, bias=False))
        self.classLayers.append(nn.BatchNorm1d(128))
        self.classLayers.append(nn.ReLU())
        self.classLayers.append(nn.Dropout(0.5))
        self.classLayers.append(nn.Linear(128, out_classes))

    def forward(self, x):
        """
        Forward pass of the network.

        :param x: input tensor
        :type x: torch.Tensor
        :return: output tensor
        :rtype: torch.Tensor
        """
        encoding = x
        for layer in self.encodeLayers:
            encoding = layer(encoding)
        pooling = self.pool(encoding)
        classification = pooling
        for layer in self.classLayers:
            classification = layer(classification)
        return classification
