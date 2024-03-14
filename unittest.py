import unittest
import torch
from torch.utils.data import DataLoader, Dataset

from modules.data import NeuralfpDataset
from modules.transformations import GPUTransformNeuralfp
from util import load_config, load_augmentation_index


class TestDataLoader(unittest.TestCase):
    def test_data_loader_shape(self):
        cfg = load_config("grafp.yaml")
        train_dir = cfg['train_dir']

        noise_train_idx = load_augmentation_index(cfg['noise_dir'], splits=0.8)["train"]
        ir_train_idx = load_augmentation_index(cfg['ir_dir'], splits=0.8)["train"]

        train_augment = GPUTransformNeuralfp(cfg=cfg, ir_dir=ir_train_idx, noise_dir=noise_train_idx, train=True)
        dataset = NeuralfpDataset(cfg=cfg, path=train_dir, train=True, transform=train_augment)

        batch_size = 8
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for batch in data_loader:
            x_i, x_j = batch
            self.assertTrue(torch.eq(x_i.shape, x_j.shape))

if __name__ == '__main__':
    unittest.main()
