import os
import json
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torchaudio
import numpy as np
import librosa
import torch.nn as nn
import warnings

from util import load_index, get_frames, qtile_normalize


class NeuralfpDataset(Dataset):
    def __init__(self, cfg, path, transform=None, train=False):
        self.path = path
        self.transform = transform
        self.train = train
        self.norm = cfg['norm']
        self.offset = cfg['offset']
        self.sample_rate = cfg['fs']
        self.dur = cfg['dur']
        self.n_frames = cfg['n_frames']
        self.size = cfg['train_sz'] if train else cfg['val_sz']
        self.filenames = load_index(path, max_len=self.size)
        print(f"Loaded {len(self.filenames)} files from {path}")
        self.ignore_idx = []
  
        
    def __getitem__(self, idx):
        if idx in self.ignore_idx:
            return self[idx + 1]
        
        datapath = self.filenames[str(idx)]
        try:
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            audio, sr = torchaudio.load(datapath)

        except Exception:
            print("Error loading:" + self.filenames[str(idx)])
            self.ignore_idx.append(idx)
            return self[idx+1]

        audio_mono = audio.mean(dim=0)
        if self.norm is not None:
            audio_mono = qtile_normalize(audio_mono, q=self.norm)
        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
        audio_resampled = resampler(audio_mono)    

        clip_frames = int(self.sample_rate*self.dur)
        
        if len(audio_resampled) <= clip_frames:
            self.ignore_idx.append(idx)
            return self[idx + 1]
        

        
        #   For training pipeline, output a random frame of the audio
        if self.train:
            a_i = audio_resampled
            a_j = a_i.clone()
            if self.transform is not None:
                a_i, a_j = self.transform(a_i, a_j)

            if a_i is None or a_j is None:
                return self[idx + 1]
            
            offset_mod = int(self.sample_rate*(self.offset) + clip_frames)
            if len(audio_resampled) < offset_mod:
                print(len(audio_resampled), offset_mod)
            r = np.random.randint(0,len(audio_resampled)-offset_mod)
            ri = np.random.randint(0,offset_mod - clip_frames)
            rj = np.random.randint(0,offset_mod - clip_frames)
            clip_i = a_i[r:r+offset_mod]
            clip_j = a_j[r:r+offset_mod]
            x_i = clip_i[ri:ri+clip_frames]
            x_j = clip_j[rj:rj+clip_frames]

            return torch.unsqueeze(x_i, 0), torch.unsqueeze(x_j, 0)
        
        #   For validation / test, output consecutive (overlapping) frames
        else:
            return torch.unsqueeze(audio_resampled, 0)
            # return audio_resampled
    
    def __len__(self):
        return len(self.filenames)