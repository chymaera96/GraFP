import torch
import torch.nn as nn

class Residual(nn.Module):
    def __init__(self, inplanes, channels, temporal_conv, strides, downsample = None):
        super(Residual, self).__init__()

        if temporal_conv:
            kernels = [[3,1],[1,3],[1,1]]
        else:
            kernels = [[1,1],[1,3],[1,1]]

        self.conv1 = nn.Sequential(
                        nn.Conv2d(inplanes, channels[1], kernel_size = kernels[0], stride = [1, strides[0]], padding = [int(kernels[0][0] / 2) ,0]),
                        nn.GroupNorm(channels[1], channels[1]),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(channels[1], channels[1], kernel_size = kernels[1], stride = [1, strides[1]], padding = [0, 1]),
                        nn.GroupNorm(channels[1], channels[1]),
                        nn.ReLU())
        
        self.conv3 = nn.Sequential(
                        nn.Conv2d(channels[1], channels[2], kernel_size = kernels[2], stride = [1,strides[2]], padding = 0),
                        nn.GroupNorm(channels[2], channels[2]))
        
        self.downsample = downsample
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # print(f" Downsample = {self.downsample}")
        # print(f"Inside res unit: {x.shape}")
        residual = x
        out = self.conv1(x)
        # print(f"After conv1 {out.shape}")
        out = self.conv2(out)
        # print(f"After conv2 {out.shape}")
        out = self.conv3(out)
        # print(f"After conv3 {out.shape}")
        if self.downsample:
            residual = self.downsample(x)
        # print(f"Residual shape {residual.shape}")
        out += residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, cfg):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.layers = cfg['layers']
        self.channels = [[64,128,128],[128,128,256],[256,256,512],[512,512,1024]] # [dim_in, dim_inner, dim_out]
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = [1,7], stride = 2, padding = [0,3]), #correct
                        nn.GroupNorm(64,64),
                        nn.ReLU())
        
        self.layer2 = self._make_layer(block, self.channels[0], self.layers[0], temporal_conv=False)
        self.layer3 = self._make_layer(block, self.channels[1], self.layers[1], temporal_conv=False)
        self.layer4 = self._make_layer(block, self.channels[2], self.layers[2], temporal_conv=True)
        self.layer5 = self._make_layer(block, self.channels[3], self.layers[3], temporal_conv=True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1) 
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, channels, blocks, temporal_conv, type=None, strided=True):
        downsample = None
        layers = []
        inplanes = channels[0]

        strides = [1,1,1]
        if strided:
            strides[0] = 2

        for i in range(blocks):

            if strides[0] != 1 or inplanes != channels[-1]:

                downsample = nn.Sequential(
                    nn.Conv2d(inplanes, channels[-1], kernel_size=1, stride=[1,strides[0]]),
                    nn.BatchNorm2d(channels[-1]),
                )
            else:
                downsample=None

            layers.append(block(inplanes, channels, temporal_conv, strides, downsample))
            inplanes = channels[-1]
            strides = [1,1,1]

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")
        x = self.layer2(x)
        # print(f"After layer2: {x.shape}")
        x = self.layer3(x)
        # print(f"After layer3: {x.shape}")
        x = self.layer4(x)
        # print(f"After layer4: {x.shape}")
        x = self.layer5(x)
        # print(f"After layer5: {x.shape}")
        x = self.avgpool(x)
        # print(f"After avgpool: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"After view: {x.shape}")

        return x