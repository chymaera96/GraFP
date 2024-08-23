import torch
import torch.nn as nn
import torch.nn.functional as F
from peak_extractor import GPUPeakExtractorv2
import unittest
import argparse

from encoder.graph_encoder import GraphEncoder
from encoder.ast_encoder import ASTEncoder
from simclr.simclr import SimCLR
import util

argparser = argparse.ArgumentParser()
argparser.add_argument('--arch', type=str, default='grafp')
argparser.add_argument('--ckp', type=str, default=None)


# check if model weights are loaded correctly

class TestSimCLR(unittest.TestCase):
    def test_model_weights(self):
        flag = False
        args = argparser.parse_args()
        if args.arch == 'grafp':
            cfg = util.load_config('configs/grafp.yaml')
            model = SimCLR(cfg, encoder=GraphEncoder(cfg=cfg, in_channels=8, k=3))
        else:
            cfg = util.load_config('configs/ast.yaml')
            model = SimCLR(cfg, encoder=ASTEncoder())

        model = model.to('cuda')
        model.load_state_dict(torch.load(args.ckp))
        flag = True
        self.assertTrue(flag)


if __name__ == '__main__':
    unittest.main()
        



