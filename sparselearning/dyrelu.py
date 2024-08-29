import torch
import torch.nn as nn

class DyReLU(nn.Module):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLU, self).__init__()
        self.channels = channels
        self.k = k
        self.conv_type = conv_type
        assert self.conv_type in ['1d', '2d']

        self.fc1 = nn.Linear(channels, channels // reduction)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, 2*k)
        self.sigmoid = nn.Sigmoid()

        self.register_buffer('lambdas', torch.Tensor([1.]*k + [0.5]*k).float())
        self.register_buffer('init_v', torch.Tensor([1.] + [0.]*(2*k - 1)).float())

    def get_relu_coefs(self, x):
        theta = torch.mean(x, axis=-1)
        if self.conv_type == '2d':
            theta = torch.mean(theta, axis=-1)
        theta = self.fc1(theta)
        theta = self.relu(theta)
        theta = self.fc2(theta)
        theta = 2 * self.sigmoid(theta) - 1
        return theta

    def forward(self, x):
        raise NotImplementedError


class DyReLUA(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUA, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k)

    def forward(self, x):
        assert x.shape[1] == self.channels
        theta = self.get_relu_coefs(x)

        relu_coefs = theta.view(-1, 2*self.k) * self.lambdas + self.init_v
        # BxCxL -> LxCxBx1
        x_perm = x.transpose(0, -1).unsqueeze(-1)
        output = x_perm * relu_coefs[:, :self.k] + relu_coefs[:, self.k:]
        # LxCxBx2 -> BxCxL
        result = torch.max(output, dim=-1)[0].transpose(0, -1)

        return result


class DyReLUB(DyReLU):
    def __init__(self, channels, reduction=4, k=2, conv_type='2d'):
        super(DyReLUB, self).__init__(channels, reduction, k, conv_type)
        self.fc2 = nn.Linear(channels // reduction, 2*k*channels)
        self.beta = 1.0  # for phasing drelu to relu
        self.inference_slopes = nn.Parameter(torch.Tensor(channels, 2), requires_grad=False)
        self.inference_mode = False

    def forward(self, x):
        if self.inference_mode:
            return torch.where(x < 0, x * self.inference_slopes[:, 0].view(1, -1, 1, 1),
                               x * self.inference_slopes[:, 1].view(1, -1, 1, 1))
        else:
            assert x.shape[1] == self.channels
            theta = self.get_relu_coefs(x)

            relu_coefs = theta.view(-1, self.channels, 2*self.k) * self.lambdas + self.init_v

            relu_original = torch.zeros_like(relu_coefs)
            relu_original[:, :, 0] = 1.0  # Set the positive slope to 1
            relu_coefs = relu_coefs * self.beta + relu_original * (1 - self.beta)

            if self.conv_type == '1d':
                # BxCxL -> LxBxCx1
                x_perm = x.permute(2, 0, 1).unsqueeze(-1)
                output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
                # LxBxCx2 -> BxCxL
                result = torch.max(output, dim=-1)[0].permute(1, 2, 0)

            elif self.conv_type == '2d':
                # BxCxHxW -> HxWxBxCx1
                x_perm = x.permute(2, 3, 0, 1).unsqueeze(-1)
                output = x_perm * relu_coefs[:, :, :self.k] + relu_coefs[:, :, self.k:]
                # HxWxBxCx2 -> BxCxHxW
                result = torch.max(output, dim=-1)[0].permute(2, 3, 0, 1)

            return result

    def store_inference_slopes(self):
        with torch.no_grad():
            # Compute and store the slopes
            dummy_input = torch.randn(1, self.channels, 1, 1).to(self.inference_slopes.device)
            slopes = self.get_relu_coefs(dummy_input).view(-1, self.channels, 2)
            self.inference_slopes.copy_(slopes[0])

    def set_inference_mode(self, mode=True):
        self.inference_mode = mode