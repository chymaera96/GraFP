import os
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import faiss
import json
import shutil
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DataParallel
import torchaudio
torchaudio.set_audio_backend("soundfile")



from util import \
create_fp_dir, load_config, \
query_len_from_seconds, seconds_from_query_len, \
load_augmentation_index
from modules.data import NeuralfpDataset
from encoder.graph_encoder import GraphEncoder
from simclr.simclr import SimCLR   
from modules.transformations import GPUTransformNeuralfp
from eval import get_index, load_memmap_data, eval_faiss



# Directories
root = os.path.dirname(__file__)
model_folder = os.path.join(root,"checkpoint")

parser = argparse.ArgumentParser(description='Neuralfp Testing')
parser.add_argument('--config', default='config/grafp.yaml', type=str,
                    help='Path to config file')
parser.add_argument('--test_config', default='config/test_config.yaml', type=str)
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing testing. ')
parser.add_argument('--test_dir', default='data/fma_medium.json', type=str,
                    help='path to test data')
parser.add_argument('--noise_idx', default=None, type=str)
parser.add_argument('--noise_split', default='all', type=str,
                    help='Noise index file split to use for testing (all, test)')
parser.add_argument('--fp_dir', default='fingerprints', type=str)
parser.add_argument('--query_lens', default=None, type=str)
parser.add_argument('--encoder', default='grafp', type=str)
parser.add_argument('--n_dummy_db', default=None, type=int)
parser.add_argument('--n_query_db', default=350, type=int)
parser.add_argument('--small_test', action='store_true', default=False)
parser.add_argument('--text', default='test', type=str)
parser.add_argument('--test_snr', default=None, type=int)
parser.add_argument('--recompute', action='store_true', default=False)
parser.add_argument('--k', default=3, type=int)
parser.add_argument('--test_ids', default='1000', type=str)

device = torch.device('cuda' if torch.cuda.is_available else 'cpu')



def create_table(hit_rates, overlap, dur, test_seq_len=[1,3,5,9,11,19], text="test"):
    table = f'''<table>
    <tr>
    <th colspan="5"{text}</th>
    <th>Query Length</th>
    <th>Top-1 Exact</th>
    <th>Top-1 Near</th>
    <th>Top-3 Exact</th>
    <th>Top-10 Exact</th>
    </tr>
    '''
    for idx, q_len in enumerate(test_seq_len):
        table += f'''
        <tr>
        <td>{seconds_from_query_len(q_len, overlap, dur)}</td>
        <td>{hit_rates[0][idx]}</td>
        <td>{hit_rates[1][idx]}</td>
        <td>{hit_rates[2][idx]}</td>
        <td>{hit_rates[3][idx]}</td>
        </tr>
        '''
    table += '</table>'
    return table

def create_fp_db(dataloader, augment, model, output_root_dir, verbose=True):
    fp_q = []
    fp_db = []
    print("=> Creating query and db fingerprints...")
    for idx, audio in enumerate(dataloader):
        audio = audio.to(device)
        x_i, x_j = augment(audio, audio)
        with torch.no_grad():
            _, _, z_i, z_j= model(x_i.to(device),x_j.to(device))        

        # print(f'Shape of z_i {z_i.shape} inside the create_fp_db function')
        fp_db.append(z_i.detach().cpu().numpy())
        fp_q.append(z_j.detach().cpu().numpy())

        if verbose and idx % 10 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {z_i.shape}")
        # fp = torch.cat(fp)
    
    fp_db = np.concatenate(fp_db)
    fp_q = np.concatenate(fp_q)
    arr_shape = (len(fp_db), z_i.shape[-1])


    arr_q = np.memmap(f'{output_root_dir}/query.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    
    arr_q[:] = fp_q[:]
    arr_q.flush(); del(arr_q)   #Close memmap

    np.save(f'{output_root_dir}/query_shape.npy', arr_shape)

    arr_db = np.memmap(f'{output_root_dir}/db.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr_db[:] = fp_db[:]
    arr_db.flush(); del(arr_db)   #Close memmap

    np.save(f'{output_root_dir}/db_shape.npy', arr_shape)

def create_dummy_db(dataloader, augment, model, output_root_dir, fname='dummy_db', verbose=True):
    fp = []
    print("=> Creating dummy fingerprints...")
    for idx, audio in enumerate(dataloader):
        audio = audio.to(device)
        x_i, _ = augment(audio, None)
        # x_i = torch.unsqueeze(db[0],1)
        with torch.no_grad():
            _, _, z_i, _= model(x_i.to(device),x_i.to(device))  

        # print(f'Shape of z_i {z_i.shape} inside the create_dummy_db function')
        fp.append(z_i.detach().cpu().numpy())
        
        if verbose and idx % 100 == 0:
            print(f"Step [{idx}/{len(dataloader)}]\t shape: {z_i.shape}")
        # fp = torch.cat(fp)
    
    fp = np.concatenate(fp)
    arr_shape = (len(fp), z_i.shape[-1])

    arr = np.memmap(f'{output_root_dir}/{fname}.mm',
                    dtype='float32',
                    mode='w+',
                    shape=arr_shape)
    arr[:] = fp[:]
    arr.flush(); del(arr)   #Close memmap

    np.save(f'{output_root_dir}/{fname}_shape.npy', arr_shape)


def main():

    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.test_snr is not None:
        cfg['val_snr'] = [int(args.test_snr), int(args.test_snr)]
    test_cfg = load_config(args.test_config)
    ir_dir = cfg['ir_dir']
    noise_dir = cfg['noise_dir']
    # args.recompute = False
    # assert args.recompute is False
    assert args.small_test is False
    # Hyperparameters
    random_seed = 42
    shuffle_dataset =True

    ################## kNN experimental setup ##################
    if list(test_cfg.keys())[0] == 'tck':
        test_cfg = { f'tck{args.k}' : ['best']}
    ###########################################################

    print("Creating new model...")
    if args.encoder == 'resnet':
        # TODO: Add support for resnet encoder (deprecated)
        raise NotImplementedError
    elif args.encoder == 'grafp':
        model = SimCLR(cfg, encoder=GraphEncoder(cfg=cfg, in_channels=cfg['n_filters'], k=args.k))
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            # model = DataParallel(model).to(device)
            model = model.to(device)
            model = torch.nn.DataParallel(model)
        else:
            model = model.to(device)

    print("Creating dataloaders ...")

    # Augmentation for testing with specific noise subsets
    if args.noise_idx is not None:
        noise_test_idx = load_augmentation_index(noise_dir, json_path=args.noise_idx, splits=0.8)[args.noise_split]
    else:
        noise_test_idx = load_augmentation_index(noise_dir, splits=0.8)["test"]
    ir_test_idx = load_augmentation_index(ir_dir, splits=0.8)["test"]
    test_augment = GPUTransformNeuralfp(cfg=cfg, ir_dir=ir_test_idx, 
                                        noise_dir=noise_test_idx, 
                                        train=False).to(device)

    dataset = NeuralfpDataset(cfg, path=args.test_dir, train=False)


    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split1 = args.n_dummy_db
    split2 = args.n_query_db
    if split1 is None:
        split1 = len(dataset) - split2
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    dummy_indices, query_db_indices = indices[:split1], indices[split1: split1 + split2]

    print(f"Creating dummy db with {len(dummy_indices)} samples and query db with {len(query_db_indices)} samples")
    dummy_db_sampler = SubsetRandomSampler(dummy_indices)
    query_db_sampler = SubsetRandomSampler(query_db_indices)

    dummy_db_loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=4, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=dummy_db_sampler)
    
    query_db_loader = torch.utils.data.DataLoader(dataset, batch_size=1, 
                                            shuffle=False,
                                            num_workers=4, 
                                            pin_memory=True, 
                                            drop_last=False,
                                            sampler=query_db_sampler)


    if args.small_test:
        index_type = 'l2'
    else:
        index_type = 'ivfpq'

    if args.query_lens is not None:
        args.query_lens = [int(q) for q in args.query_lens.split(',')]
        test_seq_len = [query_len_from_seconds(q, cfg['overlap'], dur=cfg['dur'])
                        for q in args.query_lens]
        

    for ckp_name, epochs in test_cfg.items():
        if not type(epochs) == list:
            epochs = [epochs]   # Hack to handle case where only best ckp is to be tested
        writer = SummaryWriter(f'runs/{ckp_name}')

        for epoch in epochs:
            ckp = os.path.join(model_folder, f'model_{ckp_name}_{str(epoch)}.pth')
            if os.path.isfile(ckp):
                print("=> loading checkpoint '{}'".format(ckp))
                checkpoint = torch.load(ckp)
                # Check for DataParallel
                if 'module' in list(checkpoint['state_dict'].keys())[0] and torch.cuda.device_count() == 1:
                    checkpoint['state_dict'] = {key.replace('module.', ''): value for key, value in checkpoint['state_dict'].items()}
                model.load_state_dict(checkpoint['state_dict'])
            else:
                print("=> no checkpoint found at '{}'".format(ckp))
                continue
            
            fp_dir = create_fp_dir(resume=ckp, train=False)
            if args.recompute or os.path.isfile(f'{fp_dir}/db.mm') is False:
                print("=> Computing dummy fingerprints...")
                create_dummy_db(dummy_db_loader, augment=test_augment,
                                model=model, output_root_dir=fp_dir, verbose=False)
            else:
                print("=> Skipping dummy db creation...")

            create_fp_db(query_db_loader, augment=test_augment, 
                         model=model, output_root_dir=fp_dir, verbose=False)
            
            
            text = f'{args.text}_{str(epoch)}'
            label = epoch if type(epoch) == int else 0


            if args.query_lens is not None:
                hit_rates = eval_faiss(emb_dir=fp_dir,
                                    test_ids=args.test_ids, 
                                    test_seq_len=test_seq_len, 
                                    index_type=index_type,
                                    nogpu=True) 

                writer.add_text("table", 
                                create_table(hit_rates, 
                                            cfg['overlap'], cfg['dur'],
                                            test_seq_len, text=text), 
                                label)
  
            else:
                hit_rates = eval_faiss(emb_dir=fp_dir, 
                                    test_ids=args.test_ids, 
                                    index_type=index_type,
                                    nogpu=True)
                
                writer.add_text("table", 
                                create_table(hit_rates, 
                                            cfg['overlap'], cfg['dur'], text=text), 
                                label)


            print("-------Test hit-rates-------")
            # Create table
            print(f'Top-1 exact hit rate = {hit_rates[0]}')
            print(f'Top-1 near hit rate = {hit_rates[1]}')



if __name__ == '__main__':
    main()