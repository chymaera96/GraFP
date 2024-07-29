import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_audiomentations import Compose,AddBackgroundNoise, ApplyImpulseResponse
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking, AmplitudeToDB
import warnings


class GPUTransformNeuralfp(nn.Module):
    def __init__(self, cfg, ir_dir, noise_dir, train=True, cpu=False):
        super(GPUTransformNeuralfp, self).__init__()
        self.sample_rate = cfg['fs']
        self.ir_dir = ir_dir
        self.noise_dir = noise_dir
        self.overlap = cfg['overlap']
        self.arch = cfg['arch']
        self.n_frames = cfg['n_frames']
        self.train = train
        self.cpu = cpu
        self.cfg = cfg

        self.train_transform = Compose([
            ApplyImpulseResponse(ir_paths=self.ir_dir, p=cfg['ir_prob']),
            AddBackgroundNoise(background_paths=self.noise_dir, 
                               min_snr_in_db=cfg['tr_snr'][0],
                               max_snr_in_db=cfg['tr_snr'][1], 
                               p=cfg['noise_prob']),
            ])
        
        self.val_transform = Compose([
            ApplyImpulseResponse(ir_paths=self.ir_dir, p=1),
            AddBackgroundNoise(background_paths=self.noise_dir, 
                               min_snr_in_db=cfg['val_snr'][0], 
                               max_snr_in_db=cfg['val_snr'][1], 
                               p=1),

            ])
                
        self.logmelspec = nn.Sequential(
            MelSpectrogram(sample_rate=self.sample_rate, 
                           win_length=cfg['win_len'], 
                           hop_length=cfg['hop_len'], 
                           n_fft=cfg['n_fft'], 
                           n_mels=cfg['n_mels']),
            AmplitudeToDB()
        ) 

        self.spec_aug = nn.Sequential(
            TimeMasking(cfg['time_mask'], True),
            FrequencyMasking(cfg['freq_mask'], True)
        )
        
        self.melspec = MelSpectrogram(sample_rate=self.sample_rate, win_length=cfg['win_len'], hop_length=cfg['hop_len'], n_fft=cfg['n_fft'], n_mels=cfg['n_mels'])
    
    def forward(self, x_i, x_j):
        if self.cpu:
            try:
                x_j = self.train_transform(x_j.view(1,1,x_j.shape[-1]), sample_rate=self.sample_rate)
            except ValueError:
                print("Error loading noise file. Hack to solve issue...")
                # Increase length of x_j by 1 sample
                x_j = F.pad(x_j, (0,1))
                x_j = self.train_transform(x_j.view(1,1,x_j.shape[-1]), sample_rate=self.sample_rate)
            return x_i, x_j.flatten()[:int(self.sample_rate*self.cfg['dur'])]

        if self.train:
            X_i = self.logmelspec(x_i)
            if self.arch != 'grafp':
                X_i = self.spec_aug(X_i)
            # assert X_i.shape[1] == 2, f"X_i shape: {X_i.shape}"
            assert X_i.device == torch.device('cuda:0'), f"X_i device: {X_i.device}"
            X_j = self.logmelspec(x_j)
            if self.arch != 'grafp':
                X_j = self.spec_aug(X_j)

        else:
            # print(f"x_i shape in validation augment {x_i.shape}")
            X_i = self.logmelspec(x_i.squeeze(0)).transpose(1,0)
            X_i = X_i.unfold(0, size=self.n_frames, step=int(self.n_frames*(1-self.overlap)))
            # print(f"Final X_i shape {X_i.shape}")

            if x_j is None:
                # Dummy db does not need augmentation
                return X_i, X_i
            try:
                x_j = self.val_transform(x_j.view(1,1,x_j.shape[-1]), sample_rate=self.sample_rate)
            except ValueError:
                print("Error loading noise file. Retrying...")
                x_j = self.val_transform(x_j.view(1,1,x_j.shape[-1]), sample_rate=self.sample_rate)

            X_j = self.logmelspec(x_j.flatten()).transpose(1,0)
            # print(f"Intermediate X_j shape {X_j.shape}")
            X_j = X_j.unfold(0, size=self.n_frames, step=int(self.n_frames*(1-self.overlap)))
            # print(f"After unfold X_j shape {X_j.shape}")

        return X_i, X_j