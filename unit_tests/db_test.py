import torch
import torch.nn as nn
import torch.nn.functional as F
from peak_extractor import GPUPeakExtractorv2
import unittest
import argparse
import numpy as np
import os

from eval import load_memmap_data

argparser = argparse.ArgumentParser()
argparser.add_argument('--emb_dir', type=str, default='logs/emb/test')

# Check if memmap data is loaded correctly
class DatabaseTest(unittest.TestCase):
    def test_memmap(self):
        flag = False
        args = argparser.parse_args()
        for fname in os.listdir(args.emb_dir):
            fpath = os.path.join(args.emb_dir, fname)
            db, db_shape = load_memmap_data(fpath, 'dummy_db')
            print(f'Shape of {fname} is {db_shape}')
        
        flag = True
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()