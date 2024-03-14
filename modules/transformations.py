import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_audiomentations import Compose,AddBackgroundNoise, ApplyImpulseResponse
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking, AmplitudeToDB
import warnings

from peak_extractor import Analyzer, peaks2mask

class GPUTransformNeuralfp(nn.Module):
    
    def __init__(self, cfg, ir_dir, noise_dir, train=True):
        super(GPUTransformNeuralfp, self).__init__()
        self.sample_rate = cfg['fs']
        self.ir_dir = ir_dir
        self.noise_dir = noise_dir
        self.n_frames = cfg['n_frames']
        self.overlap = cfg['overlap']
        self.arch = cfg['arch']
        self.train = train

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
            MelSpectrogram(sample_rate=self.sample_rate, win_length=cfg['win_len'], hop_length=cfg['hop_len'], n_fft=cfg['n_fft'], n_mels=cfg['n_mels']),
            AmplitudeToDB()
        ) 

        self.melspec = MelSpectrogram(sample_rate=self.sample_rate, win_length=cfg['win_len'], hop_length=cfg['hop_len'], n_fft=cfg['n_fft'], n_mels=cfg['n_mels'])
    

        # self.spec_aug = nn.Sequential(
        #     TimeMasking(time_mask_param=cfg['time_mask']),
        #     FrequencyMasking(freq_mask_param=cfg['freq_mask'])
        # )


    def forward(self, x_i, x_j):

        analyzer = Analyzer(cfg=self.cfg)

        if self.train:
            try:
                x_j = self.train_transform(x_j.view(1,1,x_j.shape[-1]), sample_rate=self.sample_rate).flatten()
            except ValueError:
                print("Error loading noise file. Hack to solve issue...")
                # Increase length of x_j by 1 sample
                x_j = F.pad(x_j, (0,1))
                x_j = self.train_transform(x_j.view(1,1,x_j.shape[-1]), sample_rate=self.sample_rate).flatten()
            X_i = self.melspec(x_i)
            X_i = F.pad(X_i, (self.n_frames - X_i.size(-1), 0)).numpy()
            p_i, _ = analyzer.find_peaks(sgram=X_i)
            X_i = peaks2mask(p_i) * X_i 
            X_i = torch.from_numpy(X_i)

            X_j = self.melspec(x_j)
            X_j = F.pad(X_j, (self.n_frames - X_j.size(-1), 0)).numpy()
            p_j, _ = analyzer.find_peaks(sgram=X_j)
            X_j = peaks2mask(p_j) * X_j
            X_j = torch.from_numpy(X_j)
     
        else:
            X_i = self.melspec(x_i.squeeze(0)).squeeze(0).numpy()
            X_i = self.spec2patches(X_i, analyzer)

            try:
                x_j = self.val_transform(x_j, sample_rate=self.sample_rate)
            except ValueError:
                print("Error loading noise file. Retrying...")
                x_j = self.val_transform(x_j, sample_rate=self.sample_rate)

            X_j = self.melspec(x_j.squeeze(0)).squeeze(0).numpy()
            X_j = self.spec2patches(X_j, analyzer)

        return X_i, X_j
    
    def spec2patches(self, X, analyzer):
        """
        Spectrogram --> Peak computation --> Segmentation --> Masking --> Patches
        """
        p, _ = analyzer.find_peaks(sgram=X)
        X = torch.from_numpy(X).transpose(0,1)
        p = torch.from_numpy(p).transpose(0,1)
        X = X.unfold(0, size=self.n_frames, step=int(self.n_frames*(1-self.overlap)))
        p = p.unfold(0, size=self.n_frames, step=int(self.n_frames*(1-self.overlap)))
        for i in range(X.shape[0]):
            X[i] = peaks2mask(p[i]) * X[i]

        return X.unsqueeze(1)