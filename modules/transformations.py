import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_audiomentations import Compose,AddBackgroundNoise, ApplyImpulseResponse
from torchaudio.transforms import MelSpectrogram, TimeMasking, FrequencyMasking, AmplitudeToDB
import warnings

from peak_extractor import Analyzer, peaks2mask, GPUPeakExtractor

class TransformNeuralfp(nn.Module):
    
    def __init__(self, cfg, ir_dir, noise_dir, train=True):
        super(TransformNeuralfp, self).__init__()
        self.sample_rate = cfg['fs']
        self.ir_dir = ir_dir
        self.noise_dir = noise_dir
        self.n_peaks = cfg['n_peaks']
        self.overlap = cfg['overlap']
        self.arch = cfg['arch']
        self.n_frames = cfg['n_frames']
        self.train = train
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
            _, p_i = analyzer.find_peaks(sgram=X_i.numpy())
            if p_i is None: # spectrogram is identically zero
                return None, None
            if not p_i.shape[0] < self.n_peaks:
                return None, None
            p_i = torch.Tensor(p_i)
            # print(f"p_i shape: {p_i.shape}")
            p_i = torch.cat((p_i, torch.zeros(self.n_peaks - p_i.shape[0], 3))).permute(1,0)
        

            X_j = self.melspec(x_j)
            _, p_j = analyzer.find_peaks(sgram=X_j.numpy())
            if p_j is None: # spectrogram is identically zero
                return None, None
            if not p_j.shape[0] < self.n_peaks:
                return None, None
            p_j = torch.Tensor(p_j)
            # print(f"p_j shape: {p_j.shape}")
            p_j = torch.cat((p_j, torch.zeros(self.n_peaks - p_j.shape[0], 3))).permute(1,0)
     
        else:
            print(f"x_i shape in validation augment {x_i.shape}")
            X_i = self.melspec(x_i.squeeze(0)).squeeze(0)
            p_i = self.spec2points(X_i, analyzer).permute(0,2,1)

            try:
                x_j = self.val_transform(x_j.reshape(1,1,x_j.shape[-1]), sample_rate=self.sample_rate)
            except ValueError:
                print("Error loading noise file. Retrying...")
                x_j = self.val_transform(x_j, sample_rate=self.sample_rate)
                

            X_j = self.melspec(x_j.squeeze(0)).squeeze(0)
            p_j = self.spec2points(X_j, analyzer).permute(0,2,1)

        return p_i, p_j
    
    def spec2points(self, X, analyzer):
        """
        Spectrogram --> Segmentation --> Peaks --> batch of point clouds
        """
        X = X.transpose(0,1)
        X = X.unfold(0, size=self.n_frames, step=int(self.n_frames*(1-self.overlap)))
        p_list = []
        for i in range(X.shape[0]):
            _, p = analyzer.find_peaks(sgram=X[i].numpy())
            p_list.append(torch.Tensor(p))

        max_p = max(arr.shape[0] for arr in p_list)
        for i in range(len(p_list)):
            p_list[i] = torch.cat((p_list[i], torch.zeros(max_p - p_list[i].shape[0], 3)))

        p = torch.stack(p_list)

        return p
    

class GPUTransformNeuralfp(nn.Module):
    def __init__(self, cfg, ir_dir, noise_dir, train=True, cpu=False):
        super(GPUTransformNeuralfp, self).__init__()
        self.sample_rate = cfg['fs']
        self.ir_dir = ir_dir
        self.noise_dir = noise_dir
        self.n_peaks = cfg['n_peaks']
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
            # assert X_i.shape[1] == 2, f"X_i shape: {X_i.shape}"
            assert X_i.device == torch.device('cuda:0'), f"X_i device: {X_i.device}"
            X_j = self.logmelspec(x_j)

        else:
            # print(f"x_i shape in validation augment {x_i.shape}")
            X_i = self.logmelspec(x_i.squeeze(0)).transpose(1,0)
            X_i = X_i.unfold(0, size=self.n_frames, step=int(self.n_frames*(1-self.overlap)))
            # print(f"Final X_i shape {X_i.shape}")

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