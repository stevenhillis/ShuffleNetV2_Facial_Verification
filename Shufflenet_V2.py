import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init

def conv_3x3_bn_relu(input_channels, output_channels, stride):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, stride, 1, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True)
    )

def conv_1x1_bn_relu(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(inplace=True)
    )

def shuffle(input, groups):
    batch_size, input_channels, input_height, input_width = input.data.size()

    channels_per_group = input_channels // groups

    input = input.view(batch_size, groups, channels_per_group, input_height, input_width)
    input = torch.transpose(input, 1, 2).contiguous()
    input = input.view(batch_size, -1, input_height, input_width)

    return input

class ShufflenetResidual(nn.Module):
    def __init__(self, input_channels, output_channels, stride, is_downsampling):
        super(ShufflenetResidual, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        self.is_downsampling = is_downsampling

        half_channels = self.output_channels // 2

        if self.is_downsampling:
            self.first_half = nn.Sequential(
                # 3x3 DW conv, stride=2, w/ BN
                nn.Conv2d(self.input_channels, self.input_channels, 3, stride, 1, groups=self.input_channels, bias=False),
                nn.BatchNorm2d(self.input_channels),
                # 1x1 conv w/ BN+RELU
                nn.Conv2d(self.input_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True),
            )
            self.second_half = nn.Sequential(
                # 1x1 conv w/ BN+RELU
                nn.Conv2d(self.input_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True),
                # 3x3 DW conv, stride=2, w/ BN
                nn.Conv2d(half_channels, half_channels, 3, stride, 1, groups=half_channels, bias=False),
                nn.BatchNorm2d(half_channels),
                # 1x1 conv w/ BN+RELU
                nn.Conv2d(half_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.second_half = nn.Sequential(
                # 1x1 conv w/ BN+RELU
                nn.Conv2d(half_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True),

                # 3x3 DW conv w/ BN
                nn.Conv2d(half_channels, half_channels, 3, stride, 1, groups=half_channels, bias=False),
                nn.BatchNorm2d(half_channels),

                # 1x1 conv w/ BN+RELU
                nn.Conv2d(half_channels, half_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(half_channels),
                nn.ReLU(inplace=True),
            )
    @staticmethod
    def _concat(first_half, second_half):
        return torch.cat((first_half, second_half), 1)

    def forward(self, input):
        if self.is_downsampling:
            output = self._concat(self.first_half(input), self.second_half(input))
        else:
            input1 = input[:, :(input.shape[1]//2), :, :]
            input2 = input[:, (input.shape[1]//2):, :, :]
            output = self._concat(input1, self.second_half(input2))
        return shuffle(output, 2)

class ShufflenetV2(nn.Module):
    def __init__(self, num_classes, input_size=32, width_mult=1.):
        super(ShufflenetV2, self).__init__()

        # self.stage_repeats = [2, 4, 2]
        self.stage_repeats = [3, 7, 3]
        # self.stage_repeats = [2, 3, 5, 3]
        # self.stage_repeats = [3, 6, 3]
        # self.stage_repeats = [4, 8, 4]

        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
            # self.stage_out_channels = [-1, 24, 224, 488, 976, 1512,  2048]

        input_channel = self.stage_out_channels[1]
        # TODO: try remove or scale this down to 1x1
        self.conv1 = conv_3x3_bn_relu(3, input_channel, 2)
        # self.conv1 = conv_1x1_bn_relu(3, input_channel)
        # TODO: try remove this in the forward
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages = []
        for stage_index in range(len(self.stage_repeats)):
            num_repeats = self.stage_repeats[stage_index]
            output_channel = self.stage_out_channels[stage_index+2]
            for i in range(num_repeats):
                if i == 0:
                    self.stages.append(ShufflenetResidual(input_channel, output_channel, 2, True))
                else:
                    self.stages.append(ShufflenetResidual(input_channel, output_channel, 1, False))
                input_channel = output_channel

        self.stages = nn.Sequential(*self.stages)

        self.last_conv = conv_1x1_bn_relu(input_channel, self.stage_out_channels[-1])
        self.last_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], num_classes))
        self.verifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], 4300))

    def forward(self, input, is_verifying):
        input = self.conv1(input)
        # input = self.maxpool(input)
        input = self.stages(input)
        input = self.last_conv(input)
        input = self.last_pool(input)
        input = input.view(-1, self.stage_out_channels[-1])

        if not is_verifying:
            input = self.classifier(input)
        # else:
        #     input = self.verifier(input)

        return input
