import numpy as np
import torch
import torch.nn as nn
import itertools as it
from typing import Sequence

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
        self,
        in_size,
        out_classes: int,
        channels: Sequence[int],
        pool_every: int,
        hidden_dims: Sequence[int],
        conv_params: dict = {},
        activation_type: str = "relu",
        activation_params: dict = {},
        pooling_type: str = "max",
        pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # ====== YOUR CODE: ======
        channels = [in_channels] + self.channels

        for i in range(len(self.channels)):
            # compute class input size
            # adding convolutional layer
            layers.append(nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], **self.conv_params))

            # adding activation layer
            activation_layer = ACTIVATIONS[self.activation_type](**self.activation_params)
            layers.append(activation_layer)

            # adding pool layer
            if (i + 1) % self.pool_every == 0:
                # compute class input size
                pooling_layer = POOLINGS[self.pooling_type](**self.pooling_params)
                layers.append(pooling_layer)

            # in_channels = channels[i]

        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _n_features(self) -> int:
        """
        Calculates the number of extracted features going into the the classifier part.
        :return: Number of features.
        """
        rng_state = torch.get_rng_state()
        try:
            in_channels, in_h, in_w, = tuple(self.in_size)
            for layer in self.feature_extractor:
                if isinstance(layer, nn.Conv2d):
                    padding, dilation, kernel_size, stride = layer.padding, layer.dilation, layer.kernel_size, layer.stride
                    in_h = np.floor(((in_h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0]) + 1)
                    in_w = np.floor(((in_w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1]) + 1)
                elif isinstance(layer, POOLINGS['max']) or isinstance(layer, POOLINGS['avg']):
                    kernel_size, stride, padding = layer.kernel_size, layer.stride, layer.padding
                    dilation = layer.dilation if hasattr(layer, 'dilation') else 1
                    in_h = np.floor(((in_h + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
                    in_w = np.floor(((in_w + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1)
            return int(in_h * in_w * self.channels[-1])
        finally:
            torch.set_rng_state(rng_state)

    def _make_classifier(self):
        layers = []

        # Discover the number of features after the CNN part.
        n_features = self._n_features()
        # ====== YOUR CODE: ======
        dims = [n_features] + list(self.hidden_dims)

        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))

            activation_layer = ACTIVATIONS[self.activation_type](**self.activation_params)
            layers.append(activation_layer)

            # n_features = hidden_dim

        layers.append(nn.Linear(dims[-1], self.out_classes))
        # ========================

        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # ====== YOUR CODE: ======
        features = self.feature_extractor(x)
        features = features.reshape((features.size(0), -1))
        out = self.classifier(features)
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_sizes: Sequence[int],
        batchnorm: bool = False,
        dropout: float = 0.0,
        activation_type: str = "relu",
        activation_params: dict = {},
        **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None
        # ====== YOUR CODE: ======
        layers = []
        all_channels = [in_channels] + list(channels)
        N = len(channels)

        for i in range(N - 1):
            layer_padding = int((kernel_sizes[i]) / 2)  # the kernel size is always odd (given)
            layers.append(nn.Conv2d(all_channels[i],
                                    all_channels[i + 1],
                                    kernel_size=kernel_sizes[i],
                                    padding=layer_padding))
            if dropout > 0.0:
                layers.append(nn.Dropout2d(dropout))
            if batchnorm:
                layers.append(nn.BatchNorm2d(all_channels[i + 1]))
            layers.append(ACTIVATIONS[activation_type](*activation_params.values()))
        layer_padding = int((kernel_sizes[-1] - 1) / 2)
        layers.append(nn.Conv2d(all_channels[-2],
                                all_channels[-1],
                                kernel_size=kernel_sizes[-1],
                                padding=layer_padding))
        if all_channels[0] != all_channels[-1]:
            shortcut = nn.Conv2d(all_channels[0], all_channels[-1], kernel_size=1, bias=False)
        else:
            shortcut = nn.Identity()

        self.main_path = nn.Sequential(*layers)
        self.shortcut_path = nn.Sequential(shortcut)

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResidualBottleneckBlock(ResidualBlock):
    """
    A residual bottleneck block.
    """

    def __init__(
        self,
        in_out_channels: int,
        inner_channels: Sequence[int],
        inner_kernel_sizes: Sequence[int],
        **kwargs,
    ):
        """
        :param in_out_channels: Number of input and output channels of the block.
            The first conv in this block will project from this number, and the
            last conv will project back to this number of channel.
        :param inner_channels: List of number of output channels for each internal
            convolution in the block (i.e. not the outer projections)
            The length determines the number of convolutions.
        :param inner_kernel_sizes: List of kernel sizes (spatial) for the internal
            convolutions in the block. Length should be the same as inner_channels.
            Values should be odd numbers.
        :param kwargs: Any additional arguments supported by ResidualBlock.
        """
        # ====== YOUR CODE: ======
        super().__init__(in_channels=in_out_channels, kernel_sizes=[1] + inner_kernel_sizes + [1], channels=[inner_channels[0]] + inner_channels + [in_out_channels], **kwargs)
        # ========================


class ResNetClassifier(ConvClassifier):
    def __init__(
        self,
        in_size,
        out_classes,
        channels,
        pool_every,
        hidden_dims,
        batchnorm=False,
        dropout=0.0,
        **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # ====== YOUR CODE: ======
        channels_list = [in_channels] + list(self.channels)
        N = len(self.channels)
        P = self.pool_every

        for i in range(0, N,P):
            if i+P+1 > len(channels_list):
                channels = channels_list[i+1:]
            else:
                channels = channels_list[i+1: i+P+1]
            block = ResidualBlock(
                in_channels=channels_list[i],
                channels=channels,
                kernel_sizes=[3]*len(channels),
                batchnorm=self.batchnorm,
                dropout=self.dropout,
                activation_type=self.activation_type,
                activation_params=self.activation_params)
            layers.append(block)
            activation_layer = ACTIVATIONS[self.activation_type](**self.activation_params)
            layers.append(activation_layer)
            if i+P <= N:
                pooling = POOLINGS[self.pooling_type](**self.pooling_params)
                layers.append(pooling)
        # ========================
        seq = nn.Sequential(*layers)
        return seq

class YourCodeNet(ConvClassifier):
    def __init__(
        self, *args,
        **kwargs):
        """
        See ConvClassifier.__init__
        """
        super().__init__(*args, **kwargs)

    def _make_feature_extractor(self):
        print(self.in_size)
        self.batchnorm = True
        self.dropout = 0.1
        self.inner_bottleneck_channels = [64, 32, 64]
        self.inner_bottleneck_kernels = [3, 5, 3]
        in_channels, in_h, in_w, = tuple(self.in_size)
        print(self.dropout)
        self.conv_params = {"kernel_size": 3, "padding": 1}
        self.pooling_params = {"kernel_size": 3, "padding": 1}
        layers = []
        # ====== YOUR CODE: ======
        # Loop over groups of P output channels and create a block from them.
        self.class_in_h, self.class_in_w = in_h, in_w
        channels = self.channels
        pooling = POOLINGS[self.pooling_type]

        kernel_size = self.conv_params['kernel_size']
        pooling_kernel_size = self.pooling_params['kernel_size']

        for i in range(int(len(channels) / self.pool_every)):
            step = (i + 1) * self.pool_every
            curr_channels = channels[i * self.pool_every:step]
            layers.append(ResidualBlock(in_channels=in_channels,
                                        channels=curr_channels,
                                        kernel_sizes=[kernel_size] * len(curr_channels),
                                        batchnorm=self.batchnorm,
                                        dropout=self.dropout,
                                        activation_type=self.activation_type,
                                        activation_params=self.activation_params))

            layers.append(pooling(**self.pooling_params))
            in_channels = channels[step - 1]


        if len(channels) % self.pool_every > 0:
            curr_channels = channels[(len(channels) - len(channels) % self.pool_every):]
            layers.append(ResidualBlock(in_channels=in_channels,
                                        channels=curr_channels,
                                        kernel_sizes=[pooling_kernel_size] * len(curr_channels),
                                        batchnorm=self.batchnorm,
                                        dropout=self.dropout,
                                        activation_type=self.activation_type,
                                        activation_params=self.activation_params,
                                        ))
        # ========================
        bottleneck = ResidualBottleneckBlock(in_out_channels=channels[-1], inner_channels=self.inner_bottleneck_channels, inner_kernel_sizes=self.inner_bottleneck_kernels)
        layers.append(bottleneck)
        seq = nn.Sequential(*layers)
        return seq
    # ========================

    def _make_classifier(self):
        layers = []

        # Discover the number of features after the CNN part.
        n_features = self._n_features()
        # ====== YOUR CODE: ======
        dims = [n_features] + list(self.hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            activation_layer = ACTIVATIONS[self.activation_type](**self.activation_params)
            layers.append(activation_layer)
            if i+1%2 == 0:
                layers.append(nn.Dropout(self.dropout))

        layers.append(nn.Linear(dims[-1], self.out_classes))
        # ========================

        seq = nn.Sequential(*layers)
        return seq

