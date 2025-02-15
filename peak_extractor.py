import numpy as np
import os
import librosa
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur


class GPUPeakExtractorv2(nn.Module):
    '''
        Convolutional embedding block to compute node embeddings from the spectrogram.
    '''
    def __init__(self, cfg):
        super(GPUPeakExtractorv2, self).__init__()

        self.blur_kernel = cfg['blur_kernel']
        self.n_filters = cfg['n_filters']
        self.stride =cfg['peak_stride']

        self.convs = nn.Sequential(
            nn.Conv2d(3, 
                      self.n_filters, 
                      kernel_size=self.blur_kernel, 
                      stride=(self.stride, 1), 
                      padding=(self.blur_kernel[0] // 2, self.blur_kernel[1] // 2)
                      ),
            nn.ReLU(),
        )

        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if self.n_gpus == 0:
            self.n_gpus = 1

        T_tensor = torch.linspace(0, 1, steps=cfg['n_frames'])
        T_tensor = T_tensor.unsqueeze(0).unsqueeze(1).repeat(cfg['bsz_train'] // self.n_gpus, cfg['n_mels'], 1)
        self.T_tensor = T_tensor

        F_tensor = torch.linspace(0, 1, steps=cfg['n_mels'])
        F_tensor = F_tensor.unsqueeze(0).unsqueeze(2).repeat(cfg['bsz_train'] // self.n_gpus, 1, cfg['n_frames'])
        self.F_tensor = F_tensor
        self.init_weights()

        

        # Initialize conv layer with kaiming initialization
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        

    def forward(self, spec_tensor):
        # Normalize the spectrogram
        min_vals = torch.amin(spec_tensor, dim=(1, 2), keepdim=True)
        max_vals = torch.amax(spec_tensor, dim=(1, 2), keepdim=True)
        spec_tensor = (spec_tensor - min_vals) / (max_vals - min_vals)
        peaks = spec_tensor.unsqueeze(1)

        # Put postional tensors to the same device as the peaks tensor
        self.T_tensor = self.T_tensor.to(peaks.device)
        self.F_tensor = self.F_tensor.to(peaks.device)

        # Concatenate positional tensors (batch, 3, H, W)
        try:
            tensor = torch.cat((self.T_tensor.unsqueeze(1), self.F_tensor.unsqueeze(1), peaks), dim=1)
        except RuntimeError as e:
            # Validation case
            T_tensor = torch.linspace(0, 1, steps=spec_tensor.shape[2], device=spec_tensor.device)
            T_tensor = T_tensor.unsqueeze(0).unsqueeze(1).repeat(spec_tensor.shape[0], spec_tensor.shape[1], 1)
            F_tensor = torch.linspace(0, 1, steps=spec_tensor.shape[1], device=spec_tensor.device)
            F_tensor = F_tensor.unsqueeze(0).unsqueeze(2).repeat(spec_tensor.shape[0], 1, spec_tensor.shape[2])
            tensor = torch.cat((T_tensor.unsqueeze(1), F_tensor.unsqueeze(1), peaks), dim=1)

        feature = self.convs(tensor)

        B, C, H, W = feature.shape
        feature = feature.reshape(B, C, -1)
        return feature