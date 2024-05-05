import torch
import torch.nn as nn
import torch.nn.functional as F
from peak_extractor import GPUPeakExtractorv2


class SimCLR(nn.Module):
    def __init__(self, cfg, encoder):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        # self.projector = nn.Sequential(nn.Linear(v,u),
        #                                nn.ELU(),
        #                                nn.Linear(u,1)
        #                                )
        d = cfg['d']
        h = cfg['h']
        u = cfg['u']

        self.peak_extractor = GPUPeakExtractorv2(cfg)

        self.projector = nn.Sequential(nn.Linear(h, d*u),
                                       nn.ELU(),
                                       nn.Linear(d*u, d)
                               )

    def forward(self, x_i, x_j):
        
        x_i = self.peak_extractor(x_i)
        print('[3/4] peaks extracted')
        l1_i = self.peak_extractor.l1
        h_i = self.encoder(x_i)
        print('[4/4] encoder applied')
        # print(f'Shape of h_i {h_i.shape} inside the SimCLR forward function')
        z_i = self.projector(h_i)
        # print(f'Shape of z_i {z_i.shape} inside the SimCLR forward function')
        z_i = F.normalize(z_i, p=2)

        x_j = self.peak_extractor(x_j)
        l1_j = self.peak_extractor.l1
        h_j = self.encoder(x_j)
        z_j = self.projector(h_j)
        z_j = F.normalize(z_j, p=2)


        return l1_i, l1_j, z_i, z_j
