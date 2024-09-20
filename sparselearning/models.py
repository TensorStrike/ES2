import math
import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from sparselearning.dyrelu import DyReLUB

class SparseSpeedupBench(object):
    """Class to benchmark speedups for convolutional layers.

    Basic usage:
    1. Assing a single SparseSpeedupBench instance to class (and sub-classes with conv layers).
    2. Instead of forwarding input through normal convolutional layers, we pass them through the bench:
        self.bench = SparseSpeedupBench()
        self.conv_layer1 = nn.Conv2(3, 96, 3)

        if self.bench is not None:
            outputs = self.bench.forward(self.conv_layer1, inputs, layer_id='conv_layer1')
        else:
            outputs = self.conv_layer1(inputs)
    3. Speedups of the convolutional layer will be aggregated and print every 1000 mini-batches.
    """
    def __init__(self):
        self.layer_timings = {}
        self.layer_timings_channel_sparse = {}
        self.layer_timings_sparse = {}
        self.iter_idx = 0
        self.layer_0_idx = None
        self.total_timings = []
        self.total_timings_channel_sparse = []
        self.total_timings_sparse = []

    def get_density(self, x):
        return (x.data!=0.0).sum().item()/x.numel()

    def print_weights(self, w, layer):
        # w dims: out, in, k1, k2
        #outers = []
        #for outer in range(w.shape[0]):
        #    inners = []
        #    for inner in range(w.shape[1]):
        #        n = np.prod(w.shape[2:])
        #        density = (w[outer, inner, :, :] != 0.0).sum().item() / n
        #        #print(density, w[outer, inner])
        #        inners.append(density)
        #    outers.append([np.mean(inners), np.std(inner)])
        #print(outers)
        #print(w.shape, (w!=0.0).sum().item()/w.numel())
        pass

    def forward(self, layer, x, layer_id):
        if self.layer_0_idx is None: self.layer_0_idx = layer_id
        if layer_id == self.layer_0_idx: self.iter_idx += 1
        self.print_weights(layer.weight.data, layer)

        # calc input sparsity
        sparse_channels_in = ((x.data != 0.0).sum([2, 3]) == 0.0).sum().item()
        num_channels_in = x.shape[1]
        batch_size = x.shape[0]
        channel_sparsity_input = sparse_channels_in/float(num_channels_in*batch_size)
        input_sparsity = self.get_density(x)

        # bench dense layer
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        x = layer(x)
        end.record()
        start.synchronize()
        end.synchronize()
        time_taken_s = start.elapsed_time(end)/1000.0

        # calc weight sparsity
        num_channels = layer.weight.shape[1]
        sparse_channels = ((layer.weight.data != 0.0).sum([0, 2, 3]) == 0.0).sum().item()
        channel_sparsity_weight = sparse_channels/float(num_channels)
        weight_sparsity = self.get_density(layer.weight)

        # store sparse and dense timings
        if layer_id not in self.layer_timings:
            self.layer_timings[layer_id] = []
            self.layer_timings_channel_sparse[layer_id] = []
            self.layer_timings_sparse[layer_id] = []
        self.layer_timings[layer_id].append(time_taken_s)
        self.layer_timings_channel_sparse[layer_id].append(time_taken_s*(1.0-channel_sparsity_weight)*(1.0-channel_sparsity_input))
        self.layer_timings_sparse[layer_id].append(time_taken_s*input_sparsity*weight_sparsity)

        if self.iter_idx % 1000 == 0:
            self.print_layer_timings()
            self.iter_idx += 1

        return x

    def print_layer_timings(self):
        total_time_dense = 0.0
        total_time_sparse = 0.0
        total_time_channel_sparse = 0.0
        print('\n')
        for layer_id in self.layer_timings:
            t_dense = np.mean(self.layer_timings[layer_id])
            t_channel_sparse = np.mean(self.layer_timings_channel_sparse[layer_id])
            t_sparse = np.mean(self.layer_timings_sparse[layer_id])
            total_time_dense += t_dense
            total_time_sparse += t_sparse
            total_time_channel_sparse += t_channel_sparse

            print('Layer {0}: Dense {1:.6f} Channel Sparse {2:.6f} vs Full Sparse {3:.6f}'.format(layer_id, t_dense, t_channel_sparse, t_sparse))
        self.total_timings.append(total_time_dense)
        self.total_timings_sparse.append(total_time_sparse)
        self.total_timings_channel_sparse.append(total_time_channel_sparse)

        print('Speedups for this segment:')
        print('Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_time_dense, total_time_channel_sparse, total_time_dense/total_time_channel_sparse))
        print('Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_time_dense, total_time_sparse, total_time_dense/total_time_sparse))
        print('\n')

        total_dense = np.sum(self.total_timings)
        total_sparse = np.sum(self.total_timings_sparse)
        total_channel_sparse = np.sum(self.total_timings_channel_sparse)
        print('Speedups for entire training:')
        print('Dense took {0:.4f}s. Channel Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_dense, total_channel_sparse, total_dense/total_channel_sparse))
        print('Dense took {0:.4f}s. Sparse took {1:.4f}s. Speedup of {2:.4f}x'.format(total_dense, total_sparse, total_dense/total_sparse))
        print('\n')

        # clear timings
        for layer_id in list(self.layer_timings.keys()):
            self.layer_timings.pop(layer_id)
            self.layer_timings_channel_sparse.pop(layer_id)
            self.layer_timings_sparse.pop(layer_id)



class AlexNet(nn.Module):
    """AlexNet with batch normalization and without pooling.

    This is an adapted version of AlexNet as taken from
    SNIP: Single-shot Network Pruning based on Connection Sensitivity,
    https://arxiv.org/abs/1810.02340

    There are two different version of AlexNet:
    AlexNet-s (small): Has hidden layers with size 1024
    AlexNet-b (big):   Has hidden layers with size 2048

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, config='s', num_classes=1000, save_features=False, bench_model=False):
        super(AlexNet, self).__init__()
        self.save_features = save_features
        self.feats = []
        self.densities = []
        self.bench = None if not bench_model else SparseSpeedupBench()

        factor = 1 if config=='s' else 2
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 1024*factor),
            nn.BatchNorm1d(1024*factor),
            nn.ReLU(inplace=True),
            nn.Linear(1024*factor, 1024*factor),
            nn.BatchNorm1d(1024*factor),
            nn.ReLU(inplace=True),
            nn.Linear(1024*factor, num_classes),
        )

    def forward(self, x):
        for layer_id, layer in enumerate(self.features):
            if self.bench is not None and isinstance(layer, nn.Conv2d):
                x = self.bench.forward(layer, x, layer_id)
            else:
                x = layer(x)

            if self.save_features:
                if isinstance(layer, nn.ReLU):
                    self.feats.append(x.clone().detach())
                if isinstance(layer, nn.Conv2d):
                    self.densities.append((layer.weight.data != 0.0).sum().item()/layer.weight.numel())

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

class LeNet_300_100(nn.Module):
    """Simple NN with hidden layers [300, 100]

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """
    def __init__(self, save_features=None, bench_model=False):
        super(LeNet_300_100, self).__init__()
        self.fc1 = nn.Linear(28*28, 300, bias=True)
        self.fc2 = nn.Linear(300, 100, bias=True)
        self.fc3 = nn.Linear(100, 10, bias=True)
        self.mask = None

    def forward(self, x):
        x0 = x.view(-1, 28*28)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return F.log_softmax(x3, dim=1)

class MLP_CIFAR10(nn.Module):
    def __init__(self, save_features=None, bench_model=False):
        super(MLP_CIFAR10, self).__init__()

        self.fc1 = nn.Linear(3*32*32, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x0 = F.relu(self.fc1(x.view(-1, 3*32*32)))
        x1 = F.relu(self.fc2(x0))
        return F.log_softmax(self.fc3(x1), dim=1)


class LeNet_5_Caffe(nn.Module):
    """LeNet-5 without padding in the first layer.
    This is based on Caffe's implementation of Lenet-5 and is slightly different
    from the vanilla LeNet-5. Note that the first layer does NOT have padding
    and therefore intermediate shapes do not match the official LeNet-5.

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, save_features=None, bench_model=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, padding=0, bias=True)
        self.conv2 = nn.Conv2d(20, 50, 5, bias=True)
        self.fc3 = nn.Linear(50 * 4 * 4, 500)
        self.fc4 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.fc3(x.view(-1, 50 * 4 * 4)))
        x = F.log_softmax(self.fc4(x), dim=1)

        return x


VGG_CONFIGS = {
    # M for MaxPool, Number for channels
    'like': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'D': [
        64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M',
        512, 512, 512, 'M'
    ],
    'C': [
        64, 64, 'M', 128, 128, 'M', 256, 256, (1, 256), 'M', 512, 512, (1, 512), 'M',
        512, 512, (1, 512), 'M' # tuples indicate (kernel size, output channels)
    ]
}


class VGG16(nn.Module):
    """
    This is a base class to generate three VGG variants used in SNIP paper:
        1. VGG-C (16 layers)
        2. VGG-D (16 layers)
        3. VGG-like

    Some of the differences:
        * Reduced size of FC layers to 512
        * Adjusted flattening to match CIFAR-10 shapes
        * Replaced dropout layers with BatchNorm

    Based on https://github.com/mi-lad/snip/blob/master/train.py
    by Milad Alizadeh.
    """

    def __init__(self, config, num_classes=10, save_features=False, bench_model=False):
        super().__init__()

        self.features = self.make_layers(VGG_CONFIGS[config], batch_norm=True)
        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = None if not bench_model else SparseSpeedupBench()

        if config == 'C' or config == 'D':
            self.classifier = nn.Sequential(
                nn.Linear((512 if config == 'D' else 2048), 512),  # 512 * 7 * 7 in the original VGG
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, 512),
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(512, 512),  # 512 * 7 * 7 in the original VGG
                nn.ReLU(True),
                nn.BatchNorm1d(512),  # instead of dropout
                nn.Linear(512, num_classes),
            )

    @staticmethod
    def make_layers(config, batch_norm=False):
        layers = []
        in_channels = 3
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                kernel_size = 3
                if isinstance(v, tuple):
                    kernel_size, v = v
                conv2d = nn.Conv2d(in_channels, v, kernel_size=kernel_size, padding=1)
                if batch_norm:
                    layers += [
                        conv2d,
                        nn.BatchNorm2d(v),
                        nn.ReLU(inplace=True)
                    ]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer_id, layer in enumerate(self.features):
            if self.bench is not None and isinstance(layer, nn.Conv2d):
                x = self.bench.forward(layer, x, layer_id)
            else:
                x = layer(x)

            if self.save_features:
                if isinstance(layer, nn.ReLU):
                    self.feats.append(x.clone().detach())
                    self.densities.append((x.data != 0.0).sum().item()/x.numel())

        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x



class WideResNet(nn.Module):
    """Wide Residual Network with varying depth and width.

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146
    """
    def __init__(self, depth, widen_factor, num_classes=10, dropRate=0.3, save_features=False, bench_model=False):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bench = None if not bench_model else SparseSpeedupBench()
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, save_features=save_features, bench=self.bench)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, save_features=save_features, bench=self.bench)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, save_features=save_features, bench=self.bench)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.feats = []
        self.densities = []
        self.save_features = save_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        if self.bench is not None:
            out = self.bench.forward(self.conv1, x, 'conv1')
        else:
            out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        if self.save_features:
            # this is a mess, but I do not have time to refactor it now
            self.feats += self.block1.feats
            self.densities += self.block1.densities
            del self.block1.feats[:]
            del self.block1.densities[:]
            self.feats += self.block2.feats
            self.densities += self.block2.densities
            del self.block2.feats[:]
            del self.block2.densities[:]
            self.feats += self.block3.feats
            self.densities += self.block3.densities
            del self.block3.feats[:]
            del self.block3.densities[:]

        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return F.log_softmax(out, dim=1)


# class BasicBlock(nn.Module):
#
#     def __init__(self, in_planes, out_planes, stride, dropRate=0.0, save_features=False, bench=None):
#         super(BasicBlock, self).__init__()
#         self.bn1 = nn.BatchNorm2d(in_planes)
#         # self.relu1 = nn.ReLU(inplace=True)
#         self.relu1 = DyReLUB(in_planes, reduction=4, k=2, conv_type='2d')
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                                padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)
#         # self.relu2 = nn.ReLU(inplace=True)
#         self.relu2 = DyReLUB(out_planes, reduction=4, k=2, conv_type='2d')
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
#                                padding=1, bias=False)
#         self.droprate = dropRate
#         self.equalInOut = (in_planes == out_planes)
#         self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
#                                padding=0, bias=False) or None
#         self.feats = []
#         self.densities = []
#         self.save_features = save_features
#         self.bench = bench
#         self.in_planes = in_planes
#
#     def forward(self, x):
#         if not self.equalInOut:
#             x = self.relu1(self.bn1(x))
#             if self.save_features:
#                 self.feats.append(x.clone().detach())
#                 self.densities.append((x.data != 0.0).sum().item()/x.numel())
#         else:
#             out = self.relu1(self.bn1(x))
#             if self.save_features:
#                 self.feats.append(out.clone().detach())
#                 self.densities.append((out.data != 0.0).sum().item()/out.numel())
#
#         if self.bench:
#             out0 = self.bench.forward(self.conv1, (out if self.equalInOut else x), str(self.in_planes) + '.conv1')
#         else:
#             out0 = self.conv1(out if self.equalInOut else x)
#
#         out = self.relu2(self.bn2(out0))
#         if self.save_features:
#             self.feats.append(out.clone().detach())
#             self.densities.append((out.data != 0.0).sum().item()/out.numel())
#         if self.droprate > 0:
#             out = F.dropout(out, p=self.droprate, training=self.training)
#         if self.bench:
#             out = self.bench.forward(self.conv2, out, str(self.in_planes) + '.conv2')
#         else:
#             out = self.conv2(out)
#
#         return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    """Wide Residual Network network block which holds basic blocks.

    For more info, see the paper: Wide Residual Networks by Sergey Zagoruyko, Nikos Komodakis
    https://arxiv.org/abs/1605.07146
    """
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, save_features=False, bench=None):
        super(NetworkBlock, self).__init__()
        self.feats = []
        self.densities = []
        self.save_features = save_features
        self.bench = bench
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, save_features=self.save_features, bench=self.bench))
        return nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.layer:
            x = layer(x)
            if self.save_features:
                self.feats += layer.feats
                self.densities += layer.densities
                del layer.feats[:]
                del layer.densities[:]
        return x

############################################################################################################
################################################ ResNet ####################################################
############################################################################################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = DyReLUB(planes, reduction=4, k=2, conv_type='2d')

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = DyReLUB(planes, reduction=4, k=2, conv_type='2d')


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck_NoPara(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck_NoPara, self).__init__()
        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.relu1 = DyReLUB(planes)
        self.relu2 = DyReLUB(planes)
        self.relu3 = DyReLUB(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(self.expansion * planes)
            )

        self.stride = stride
        self.in_planes = in_planes
        self.planes = planes

        # learnable scaling factors
        self.scale_1 = nn.Parameter(torch.ones(1))
        self.scale_2 = nn.Parameter(torch.ones(1))
        self.scale_3 = nn.Parameter(torch.ones(1))
        if stride != 1 or in_planes != self.expansion * planes:
            self.scale_4 = nn.Parameter(torch.ones(1))

        nn.init.uniform_(self.scale_1)
        nn.init.uniform_(self.scale_2)
        nn.init.uniform_(self.scale_3)
        if hasattr(self, 'scale_4'):
            nn.init.uniform_(self.scale_4)

    def forward(self, x, weight_params):
        conv1_weight, conv2_weight, conv3_weight = weight_params[:3]

        out = self.relu1(self.bn1(F.conv2d(x, self.scale_1 * conv1_weight, stride=1, padding=0)))
        out = self.relu2(self.bn2(F.conv2d(out, self.scale_2 * conv2_weight, stride=self.stride, padding=1)))
        out = self.bn3(F.conv2d(out, self.scale_3 * conv3_weight, stride=1, padding=0))

        if len(weight_params) > 3:
            shortcut_weight = weight_params[3]
            shortcut = F.conv2d(x, self.scale_4 * shortcut_weight, stride=self.stride)
            shortcut = self.shortcut(shortcut)
        else:
            shortcut = self.shortcut(x)

        out += shortcut
        out = self.relu3(out)
        return out
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes, ratio=2):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.ratio = ratio

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = DyReLUB(64, reduction=4, k=2, conv_type='2d')
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block[0].expansion, num_classes, bias=False)

    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1] * (num_blocks - 1)
    #     layers = []
    #     for i, stride in enumerate(strides):
    #         if i < self.ratio:      # normal weights
    #             layers.append(block(self.in_planes, planes, stride))
    #         else:       # shared weights
    #             layers.append(BasicBlock_NoPara(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.ModuleList(layers)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            if i < self.ratio:
                layers.append(block[0](self.in_planes, planes, stride))     # basic block
            else:
                layers.append(block[1](self.in_planes, planes, stride))     # bottleneck
            self.in_planes = planes * block[0].expansion
        return nn.ModuleList(layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))

        out = self.layer_forward(self.layer1, out)
        out = self.layer_forward(self.layer2, out)
        out = self.layer_forward(self.layer3, out)
        out = self.layer_forward(self.layer4, out)

        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def layer_forward(self, layer, x):
        for i, block in enumerate(layer):
            if i < self.ratio:
                x = block(x)
            else:
                # Extract only the convolutional weights
                weight_params = []
                for name, param in layer[self.ratio - 1].named_parameters():    # use the parameters from ratio-1
                    if 'conv' in name and 'weight' in name:
                        weight_params.append(param)
                x = block(x, weight_params)
        return x

    # def get_shared_para(self):
    #     '''
    #     counts all active and pruned parameters in shared blocks
    #     '''
    #     shared_params = 0
    #     for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
    #         if len(layer) > self.ratio:
    #             params_per_block = sum(p.numel() for p in layer[self.ratio-1].parameters() if len(p.shape) > 1)
    #             shared_params += params_per_block * (len(layer) - self.ratio)
    #     return shared_params

    def get_shared_para(self):
        '''
        Counts only the convolutional and linear weights in shared blocks.
        '''
        shared_params = 0
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if len(layer) > self.ratio:
                num_shared_blocks = len(layer) - self.ratio
                last_non_shared_block = layer[self.ratio - 1]
                # Count only the prunable weights
                params_per_block = sum(
                    p.numel() for name, p in last_non_shared_block.named_parameters()
                    if ('weight' in name and (
                                'conv' in name.lower() or 'linear' in name.lower()) and 'relu' not in name.lower())
                )
                shared_params += params_per_block * num_shared_blocks
        return shared_params


class BasicBlock_NoPara(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_NoPara, self).__init__()
        self.in_planes = in_planes
        self.planes = planes
        self.stride = stride

        self.bn1 = nn.BatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.relu1 = DyReLUB(planes)
        self.relu2 = DyReLUB(planes)

        # learnable scaling factors
        self.scale_1 = nn.Parameter(torch.ones(1))
        self.scale_2 = nn.Parameter(torch.ones(1))
        if stride != 1 or in_planes != self.expansion*planes:
            self.scale_3 = nn.Parameter(torch.ones(1))

        nn.init.uniform_(self.scale_1)
        nn.init.uniform_(self.scale_2)
        if hasattr(self, 'scale_3'):
            nn.init.uniform_(self.scale_3)

    def forward(self, x, weight_params):
        conv1_weight, conv2_weight = weight_params[:2]

        # shortcut_weight = weight_params[2] if len(weight_params) > 2 else None

        assert conv1_weight.shape == (
        self.planes, self.in_planes, 3, 3), f"Unexpected shape for conv1_weight: {conv1_weight.shape}"
        assert conv2_weight.shape == (
        self.planes, self.planes, 3, 3), f"Unexpected shape for conv2_weight: {conv2_weight.shape}"

        out = self.relu1(self.bn1(F.conv2d(x, self.scale_1 * conv1_weight, stride=self.stride, padding=1)))
        out = self.bn2(F.conv2d(out, self.scale_2 * conv2_weight, stride=1, padding=1))

        # if shortcut_weight is not None:
        #     assert shortcut_weight.shape == (
        #     self.planes, self.in_planes, 1, 1), f"Unexpected shape for shortcut_weight: {shortcut_weight.shape}"
        #     shortcut = F.conv2d(x, self.scale_3 * shortcut_weight, stride=self.stride)
        #     shortcut = self.shortcut(shortcut)
        # else:
        #     shortcut = self.shortcut(x)

        if len(weight_params) > 2:
            shortcut_weight = weight_params[2]
            assert shortcut_weight.shape == (self.planes * self.expansion, self.in_planes, 1,
                                             1), f"Unexpected shape for shortcut_weight: {shortcut_weight.shape}"
            shortcut = F.conv2d(x, self.scale_3 * shortcut_weight, stride=self.stride)
            shortcut = self.shortcut(shortcut)
        else:
            shortcut = self.shortcut(x)

        out += shortcut
        out = self.relu2(out)
        return out


# def ResNet18(c=1000, ratio=2):
#     return ResNet(BasicBlock, [2,2,2,2],c, ratio=ratio)

# def ResNet34(c=10, ratio=2):
#     return ResNet(BasicBlock, [3,4,6,3],c,ratio=ratio)
#
# def ResNet50(c=10, ratio=2):
#     return ResNet(Bottleneck, [3,4,6,3],c, ratio=ratio)

def ResNet34(c, ratio=2):
    return ResNet([BasicBlock, BasicBlock_NoPara], [3, 4, 6, 3], c, ratio=ratio)

def ResNet50(c, ratio=2):
    return ResNet([Bottleneck, Bottleneck_NoPara], [3, 4, 6, 3], c, ratio=ratio)
