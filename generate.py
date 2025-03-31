# Ablation: generating audio fingerprints for a given audio dataset

import os
import json
import numpy as np
import torch
import torch.nn as nn
import argparse

from util import \
create_fp_dir, load_config, \
query_len_from_seconds, seconds_from_query_len, \
load_augmentation_index
from modules.data import NeuralfpDataset
from encoder.graph_encoder import GraphEncoder
from simclr.simclr import SimCLR   
from modules.transformations import GPUTransformNeuralfp

parser = argparse.ArgumentParser(description='GraFPrint Embedding generation')
parser.add_argument('--config', default='config/grafp.yaml', type=str,
                    help='Path to config file')
parser.add_argument('--ckp', default=None, type=str,
                    help='Path to checkpoint file')
parser.add_argument('--test_dir', default=None, type=str,
                    help='Path to audio directory or index file')
parser.add_argument('--output_dir', default='output', type=str,
                    help='Path to output directory')



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_db(dataloader, model, augment, output_dir, concat=True, max_size=128):
    fp = []
    print("Computing fingerprints...")
    for idx, audio in enumerate(dataloader):
        audio = audio.to(device)
        x_i, _ = augment(audio, None)
        # Determining mini-batches for large audio files
        x_list = torch.split(x_i, max_size, dim=0)  

        for x_i in x_list:
            with torch.no_grad():
                _, _, z_i, _ = model(x_i.to(device), x_i.to(device))  

            fp.append(z_i.detach().cpu().numpy())

        if idx % 10 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {z_i.shape}")

    # Concatenate fingerprints to remove song-level separation
    if concat:
        fp = np.concatenate(fp, axis=0)
    else:
        fp = np.array(fp)
    np.save(os.path.join(output_dir, "fingerprints.npy"), fp)
    

def main():

    args = parser.parse_args()
    cfg = load_config(args.config)
    ckp = args.ckp

    print("Loading model...")
    model = SimCLR(cfg, encoder=GraphEncoder(cfg=cfg, in_channels=cfg['n_filters'], k=3))
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        # model = DataParallel(model).to(device)
        model = model.to(device)
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(device)

    dataset = NeuralfpDataset(cfg, path=args.test_dir, train=False)
    db_loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=4, 
                                            pin_memory=True, 
                                            drop_last=False)
    
    # dummy augment; will not apply any transformations
    augment = GPUTransformNeuralfp(cfg=cfg, ir_dir=None, 
                                        noise_dir=None, 
                                        train=False).to(device)
    
    print("Loading checkpoint...")
    if os.path.isfile(ckp):
        print("=> loading checkpoint '{}'".format(ckp))
        checkpoint = torch.load(ckp)
        # Check for DataParallel
        if 'module' in list(checkpoint['state_dict'].keys())[0] and torch.cuda.device_count() == 1:
            checkpoint['state_dict'] = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(checkpoint['state_dict'])
    else:
        print("=> no checkpoint found at '{}'".format(ckp))
        return
    
    create_db(db_loader, model, augment, output_dir=args.output_dir, concat=True)

if __name__ == '__main__':
    main()