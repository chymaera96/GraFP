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


# def load_index(cfg, data_dir, ext=['wav','mp3'], mode="train"):

#     if data_dir.endswith('.json'):
#         print(f"=>Loading indices from index file {data_dir}")
#         with open(data_dir, 'r') as fp:
#             dataset = json.load(fp)
#         return dataset
    
#     print(f"=>Loading indices from {data_dir}")
#     train_json_path = os.path.join('data', data_dir.split('/')[-1] + "_train.json")
#     valid_json_path = os.path.join('data', data_dir.split('/')[-1] + "_valid.json")

#     if mode == "train" and os.path.exists(train_json_path):
#         print(f"Train index exists. Loading indices from {train_json_path}")
#         with open(train_json_path, 'r') as fp:
#             train = json.load(fp)
    
#     elif mode != "train" and os.path.exists(valid_json_path):
#         print(f"Valid index exists. Loading indices from {valid_json_path}")
#         with open(valid_json_path, 'r') as fp:
#             valid = json.load(fp)

#     else:
#         print(f"Creating new index files {train_json_path} and {valid_json_path}")
#         train_len = cfg['train_sz']
#         valid_len = cfg['val_sz']
#         idx = 0
#         train = {}
#         for fpath in glob.iglob(os.path.join(data_dir,'**/*.*'), recursive=True):
#             if len(train) >= train_len:
#                 break
#             if fpath.split('.')[-1] in ext: 
#                 train[str(idx)] = fpath
#                 idx += 1

#         with open(train_json_path, 'w') as fp:
#             json.dump(train, fp)

#         idx = 0
#         valid = {}
#         existing = [f.split('/')[-1] for f in list(train.values())]
#         for fpath in glob.iglob(os.path.join(data_dir,'**/*.*'), recursive=True):
#             if len(valid) >= valid_len:
#                 break
#             if fpath.split('.')[-1] in ext and fpath.split('/')[-1] not in existing:
#                 valid[str(idx)] = fpath
#                 idx += 1

#         with open(valid_json_path, 'w') as fp:
#             json.dump(valid, fp)

#     if mode == "train":
#         dataset = train
#     else:
#         dataset = valid

#     return dataset

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
    # # Check if dataparallel is used
    # if 'module' in list(checkpoint['state_dict'].keys())[0]:
    #     print("Loading model with dataparallel...")
    #     checkpoint = {key.replace('module.', ''): value for key, value in checkpoint.items()}
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

def create_fp_dir(resume=None, ckp=None, epoch=1, train=True):
    if train:
        parent_dir = 'logs/emb/valid'
    else:
        parent_dir = 'logs/emb/test'

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



def create_train_set(data_dir, dest, size=10000):
    if not os.path.exists(dest):
        os.mkdir(dest)
        print(data_dir)
        print(dest)
    for ix,fname in enumerate(os.listdir(data_dir)):
        fpath = os.path.join(data_dir, fname)
        if ix <= size and fpath.endswith('mp3'):
            shutil.move(fpath,dest)
            print(ix)
        if len(os.listdir(dest)) >= size:
            return dest
    
    return dest

def create_downstream_set(data_dir, size=5000):
    src = os.path.join(data_dir, f'fma_downstream')
    dest = data_dir
    # if not os.path.exists(dest):
    #     os.mkdir(dest)   
    # if len(os.listdir(dest)) >= size:
    #     return dest
    for ix,fname in enumerate(os.listdir(src)):
        fpath = os.path.join(src, fname)
        if not fpath.endswith('mp3'):
            continue
        # if ix < size:
        if len(os.listdir(src)) > 500:
            shutil.move(fpath,dest)

    return dest

def preprocess_aug_set_sr(data_dir, sr=22050):
    for fpath in glob.iglob(os.path.join(data_dir,'**/*.wav'), recursive=True):
        y, sr = sf.read(fpath)
        print(sr)
        break
        # sf.write(fpath, data=y, samplerate=sr)
    return

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

    # Get paths of files not in the index
def get_test_index(data_dir):
    train_idx = load_index(data_dir)
    all_file_list = glob.glob(os.path.join(data_dir,'**/*.mp3'), recursive=True)
    print(f'Number of files in {data_dir}: {len(all_file_list)}')
    # test_idx = {str(i):f for i,f in enumerate(all_file_list) if f not in train_idx.values()}
    idx = 0
    test_idx = {}
    for i, fpath in enumerate(all_file_list):
        if i % 200 == 0:
            print(f"Processed {i}/{len(all_file_list)} files")
        if fpath not in train_idx.values():
            test_idx[str(idx)] = fpath
            idx += 1

    return test_idx
    
def main():

    path = '/import/c4dm-datasets-ext/fma/fma/data/fma_small'
    cfg = load_config('config/grafp.yaml')
    # train_json_path = os.path.join('data', path.split('/')[-1] + "_train.json")
    # train_idx = load_index(cfg, data_dir=train_json_path, mode='train')
    # print("Train index length: ", len(train_idx))
    # all_file_list = os.listdir(path)
    
    # print(f'Number of files in {path}: {len(all_file_list)}')
    # test_idx_path = os.path.join('data', path.split('/')[-1] + "_test.json")
    # test_idx = {}
    # print("Creating test index...")
    # ix = 0
    # for i, fpath in enumerate(all_file_list):
    #     fpath = os.path.join(path, fpath)
    #     if i % 200 == 0:
    #         print(f"Processed {i}/{len(all_file_list)} files")
    #     if fpath not in train_idx.values() and fpath.endswith('mp3'):
    #         test_idx[str(ix)] = fpath
    #         ix += 1

    # with open(test_idx_path, 'w') as fp:
    #     json.dump(test_idx, fp)
    # print("Test index length: ", len(test_idx))

    # Create a index file containing all the files in the directory
    all_file_list = os.listdir(path)
    print(f'Number of files in {path}: {len(all_file_list)}')
    all_idx_path = os.path.join('data', path.split('/')[-1] + "_all.json")
    all_idx = {}
    print("Creating all index...")
    ix = 0
    for i, fpath in enumerate(all_file_list):
        fpath = os.path.join(path, fpath)
        if i % 200 == 0:
            print(f"Processed {i}/{len(all_file_list)} files")
        if fpath.endswith('mp3'):
            all_idx[str(ix)] = fpath
            ix += 1

    with open(all_idx_path, 'w') as fp:
        json.dump(all_idx, fp)
    print("All index length: ", len(all_idx))
    
if __name__ == '__main__':
    main()
