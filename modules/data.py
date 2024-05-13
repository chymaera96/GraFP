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

from util import load_index, get_frames, qtile_normalize, qtile_norm


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
        self.silence = cfg['silence']
        self.error_threshold = cfg['error_threshold']

        if train:
            self.filenames = load_index(cfg, path, mode="train")
        else:
            self.filenames = load_index(cfg, path, mode="valid")

        print(f"Loaded {len(self.filenames)} files from {path}")
        self.ignore_idx = []
        self.error_counts = {}
        
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
            self.error_counts[idx] = self.error_counts.get(idx, 0) + 1
            if self.error_counts[idx] > self.error_threshold:
                self.ignore_idx.append(idx)
            return self[idx+1]

        audio_mono = audio.mean(dim=0)
        
        resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
        audio_resampled = resampler(audio_mono)    

        clip_frames = int(self.sample_rate*self.dur)
        
        if len(audio_resampled) <= clip_frames:
            # self.ignore_idx.append(idx)
            return self[idx + 1]
    
        
        #   For training pipeline, output a random frame of the audio
        if self.train:
            a_i = audio_resampled
            a_j = a_i.clone()
            
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

            if x_i.abs().max() < self.silence or x_j.abs().max() < self.silence:
                print("Silence detected. Skipping...")
                return self[idx + 1]
            
            if self.norm is not None:
                norm_val = qtile_norm(audio_resampled, q=self.norm)
                x_i = x_i / norm_val
                x_j = x_j / norm_val

            if self.transform is not None:
                x_i, x_j = self.transform(x_i, x_j)

            if x_i is None or x_j is None:
                return self[idx + 1]

            # Pad or truncate to sample_rate * dur
            if len(x_i) < clip_frames:
                x_i = F.pad(x_i, (0, clip_frames - len(x_i)))
            else:
                x_i = x_i[:clip_frames]

            if len(x_j) < clip_frames:
                x_j = F.pad(x_j, (0, clip_frames - len(x_j)))
            else:    
                x_j = x_j[:clip_frames]


            return x_i, x_j
        
        #   For validation / test, output consecutive (overlapping) frames
        else:
            return audio_resampled
            # return audio_resampled
    
    def __len__(self):
        return len(self.filenames)