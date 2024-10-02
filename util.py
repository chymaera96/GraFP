import os
import torch
import numpy as np
import json
import glob
import soundfile as sf
import shutil
import yaml
from prettytable import PrettyTable

class DummyScaler:
    def scale(self, loss):
        return loss

    def step(self, optimizer):
        return optimizer.step()

    def update(self):
        pass

def load_index(cfg, data_dir, ext=['wav','mp3'], shuffle_dataset=True, mode="train"):

    if data_dir.endswith('.json'):
        print(f"=>Loading indices from index file {data_dir}")
        with open(data_dir, 'r') as fp:
            dataset = json.load(fp)
        return dataset
    
    print(f"=>Loading indices from {data_dir}")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory {data_dir} not found")
    
    json_path = os.path.join(cfg['data_dir'], data_dir.split('/')[-1] + ".json")
    if os.path.exists(json_path):
        print(f"Loading indices from {json_path}")
        with open(json_path, 'r') as fp:
            dataset = json.load(fp)
        return dataset
    
    fpaths = glob.glob(os.path.join(data_dir,'**/*.*'), recursive=True)
    fpaths = [p for p in fpaths if p.split('.')[-1] in ext]
    dataset_size = len(fpaths)
    indices = list(range(dataset_size))
    if shuffle_dataset :
        np.random.seed(42)
        np.random.shuffle(indices)
    if mode == "train":
        size = cfg['train_sz']
    else:
        size = cfg['val_sz']
    dataset = {str(i):fpaths[ix] for i,ix in enumerate(indices[:size])}

    with open(json_path, 'w') as fp:
        json.dump(dataset, fp)

    return dataset

def load_augmentation_index(data_dir, splits, json_path=None, ext=['wav','mp3'], shuffle_dataset=True):
    dataset = {'train' : [], 'test' : [], 'validate': []}
    if json_path is None:
        json_path = os.path.join(data_dir, data_dir.split('/')[-1] + ".json")
    if not os.path.exists(json_path):
        fpaths = glob.glob(os.path.join(data_dir,'**/*.*'), recursive=True)
        fpaths = [p for p in fpaths if p.split('.')[-1] in ext]
        dataset_size = len(fpaths)   
        indices = list(range(dataset_size))
        if shuffle_dataset :
            np.random.seed(42)
            np.random.shuffle(indices)
        if type(splits) == list or type(splits) == np.ndarray:
            splits = [int(splits[ix]*dataset_size) for ix in range(len(splits))]
            train_idxs, valid_idxs, test_idxs = indices[:splits[0]], indices[splits[0]: splits[0] + splits[1]], indices[splits[1]:]
            dataset['validate'] = [fpaths[ix] for ix in valid_idxs]
        else:
            splits = int(splits*dataset_size)
            train_idxs, test_idxs = indices[:splits], indices[splits:]
        
        dataset['train'] = [fpaths[ix] for ix in train_idxs]
        dataset['test'] = [fpaths[ix] for ix in test_idxs]

        with open(json_path, 'w') as fp:
            json.dump(dataset, fp)
    
    else:
        with open(json_path, 'r') as fp:
            dataset = json.load(fp)

    return dataset


def get_frames(y, frame_length, hop_length):
    # frames = librosa.util.frame(y.numpy(), frame_length, hop_length, axis=0)
    frames = y.unfold(0, size=frame_length, step=hop_length)
    return frames

def qtile_normalize(y, q, eps=1e-8):
    return y / (eps + torch.quantile(y.abs(),q=q))

def qtile_norm(y, q, eps=1e-8):
    return eps + torch.quantile(y.abs(),q=q)


def query_len_from_seconds(seconds, overlap, dur):
    hop = dur*(1-overlap)
    return int((seconds-dur)/hop + 1)

def seconds_from_query_len(query_len, overlap, dur):
    hop = dur*(1-overlap)
    return int((query_len-1)*hop + dur)

def load_ckp(checkpoint_fpath, model, optimizer, scheduler):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    return model, optimizer, scheduler, checkpoint['epoch'], checkpoint['loss'], checkpoint['valid_acc']

def save_ckp(state,model_name,model_folder,text):
    if not os.path.exists(model_folder): 
        print("Creating checkpoint directory...")
        os.mkdir(model_folder)
    torch.save(state, "{}/model_{}_{}.pth".format(model_folder, model_name, text))

def load_config(config_path):
    with open(config_path, 'r') as fp:
        config = yaml.safe_load(fp)
    return config

def override(config_val, arg):
    return arg if arg is not None else config_val 


def create_fp_dir(resume=None, ckp=None, epoch=1, train=True, large=False):

    if train:
        parent_dir = 'logs/store/valid'
    else:
        if large:
            parent_dir = '/data/scratch/acw723/logs/store/test'
        else:
            parent_dir = 'logs/store/test'

    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    if resume is not None:
        ckp_name = resume.split('/')[-1].split('.pt')[0]
    else:
        ckp_name = f'model_{ckp}_epoch_{epoch}'
    fp_dir = os.path.join(parent_dir, ckp_name)
    if not os.path.exists(fp_dir):
        os.mkdir(fp_dir)
    return fp_dir

def update_index(data_dir, idx_path, ext=['wav','mp3']):
    # Update paths with new parent directory
    new_index = {}
    with open(idx_path, 'r') as fp:
        index = json.load(fp)
    dir_name = idx_path.split('/')[-1].split('.')[0]
    for key, value in index.items():
        rel_path = value.split(dir_name)[-1]
        new_index[key] = os.path.join(data_dir, dir_name, rel_path)

    return new_index

def count_parameters(model, encoder):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    # Write table in text file
    with open(f'model_summary_{encoder}.txt', 'w') as f:
        f.write(str(table))
    return total_params

def calculate_output_sparsity(output):
    total_elements = torch.numel(output)
    zero_elements = torch.sum((output == 0).int()).item()

    sparsity = zero_elements / total_elements * 100
    return sparsity
    
