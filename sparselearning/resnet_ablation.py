import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock_ReLU(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_ReLU, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BasicBlock_NoPara_ReLU(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_NoPara_ReLU, self).__init__()
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

        assert conv1_weight.shape == (self.planes, self.in_planes, 3, 3), f"Unexpected shape for conv1_weight: {conv1_weight.shape}"
        assert conv2_weight.shape == (self.planes, self.planes, 3, 3), f"Unexpected shape for conv2_weight: {conv2_weight.shape}"

        out = F.relu(self.bn1(F.conv2d(x, self.scale_1 * conv1_weight, stride=self.stride, padding=1)))
        out = self.bn2(F.conv2d(out, self.scale_2 * conv2_weight, stride=1, padding=1))

        if len(weight_params) > 2:
            shortcut_weight = weight_params[2]
            assert shortcut_weight.shape == (self.planes * self.expansion, self.in_planes, 1, 1), f"Unexpected shape for shortcut_weight: {shortcut_weight.shape}"
            shortcut = F.conv2d(x, self.scale_3 * shortcut_weight, stride=self.stride)
            shortcut = self.shortcut(shortcut)
        else:
            shortcut = self.shortcut(x)

        out += shortcut
        out = F.relu(out)
        return out

class ResNet_ReLU(nn.Module):
    def __init__(self, block, num_blocks, num_classes, ratio=2):
        super(ResNet_ReLU, self).__init__()
        self.in_planes = 64
        self.ratio = ratio

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block[0].expansion, num_classes, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, stride in enumerate(strides):
            if i < self.ratio:
                layers.append(block[0](self.in_planes, planes, stride))
            else:
                layers.append(block[1](self.in_planes, planes, stride))
            self.in_planes = planes * block[0].expansion
        return nn.ModuleList(layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

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
                weight_params = []
                for name, param in layer[self.ratio - 1].named_parameters():
                    if 'conv' in name and 'weight' in name:
                        weight_params.append(param)
                x = block(x, weight_params)
        return x

    def get_shared_para(self):
        shared_params = 0
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            if len(layer) > self.ratio:
                num_shared_blocks = len(layer) - self.ratio
                last_non_shared_block = layer[self.ratio - 1]
                params_per_block = sum(
                    p.numel() for name, p in last_non_shared_block.named_parameters()
                    if ('weight' in name and ('conv' in name.lower() or 'linear' in name.lower()) and 'relu' not in name.lower())
                )
                shared_params += params_per_block * num_shared_blocks
        return shared_params

def ResNet34_ReLU(c, ratio=2):
    return ResNet_ReLU([BasicBlock_ReLU, BasicBlock_NoPara_ReLU], [3, 4, 6, 3], c, ratio=ratio)