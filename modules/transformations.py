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
        self.n_frames = cfg['n_frames']
        self.overlap = cfg['overlap']
        self.arch = cfg['arch']
        self.train = train
        self.cpu = cpu
        # self.gpu_transform = Compose([
        #     # ApplyImpulseResponse(ir_paths=self.ir_dir, p=0.5),
        #     AddBackgroundNoise(background_paths=self.noise_dir, 
        #                        min_snr_in_db=cfg['tr_snr'][0], 
        #                        max_snr_in_db=cfg['tr_snr'][1], 
        #                        p=1),            
        #     ])
        
        self.cpu_transform = Compose([
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
            MelSpectrogram(sample_rate=self.sample_rate, win_length=cfg['win_len'], hop_length=cfg['hop_len'], n_fft=cfg['n_fft'], n_mels=cfg['n_mels']),
            AmplitudeToDB()
        ) 
        self.spec_aug = nn.Sequential(
            TimeMasking(time_mask_param=cfg['time_mask']),
            FrequencyMasking(freq_mask_param=cfg['freq_mask'])
)

    def forward(self, x_i, x_j):

        if self.cpu:
            try:
                x_j = self.cpu_transform(x_j.view(1,1,x_j.shape[-1]), sample_rate=self.sample_rate)
            except ValueError:
                print("Error loading noise file. Hack to solve issue...")
                # Increase length of x_j by 1 sample
                x_j = F.pad(x_j, (0,1))
                x_j = self.cpu_transform(x_j.view(1,1,x_j.shape[-1]), sample_rate=self.sample_rate)
            return x_i, x_j.flatten()


        if self.train:
            X_i = self.logmelspec(x_i)
            X_i = self.spec_aug(X_i)
            X_i = F.pad(X_i, (self.n_frames - X_i.size(-1), 0))

            # x_j = self.gpu_transform(x_j, sample_rate=self.sample_rate)
            X_j = self.logmelspec(x_j)
            X_j = self.spec_aug(X_j)
            X_j = F.pad(X_j, (self.n_frames - X_j.size(-1), 0)) 


        
        else:
            X_i = self.logmelspec(x_i.squeeze(0)).permute(2,0,1)
            X_i = X_i.unfold(0, size=self.n_frames, step=int(self.n_frames*(1-self.overlap)))

            try:
                x_j = self.val_transform(x_j, sample_rate=self.sample_rate)
            except ValueError:
                print("Error loading noise file. Retrying...")
                x_j = self.val_transform(x_j, sample_rate=self.sample_rate)

            X_j = self.logmelspec(x_j.squeeze(0)).permute(2,0,1)
            X_j = X_j.unfold(0, size=self.n_frames, step=int(self.n_frames*(1-self.overlap)))

        if self.arch == 'sfnet' or self.arch == 'resnet':    # sfnet has transposed shape   
            return X_i.permute(0,1,3,2), X_j.permute(0,1,3,2)
        else:
            return X_i, X_j