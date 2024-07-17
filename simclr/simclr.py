import torch
import torch.nn as nn
import torch.nn.functional as F
from peak_extractor import GPUPeakExtractorv2


class SimCLR(nn.Module):
    def __init__(self, cfg, encoder):
        super(SimCLR, self).__init__()
        self.encoder = encoder
        self.cfg = cfg
        # self.projector = nn.Sequential(nn.Linear(v,u),
        #                                nn.ELU(),
        #                                nn.Linear(u,1)
        #                                )
        d = cfg['d']
        h = cfg['h']
        u = cfg['u']
        if cfg['arch'] == 'grafp':
            self.peak_extractor = GPUPeakExtractorv2(cfg)
        else:
            self.peak_extractor = None

        self.projector = nn.Sequential(nn.Linear(h, d*u),
                                       nn.ELU(),
                                       nn.Linear(d*u, d)
                               )

    def forward(self, x_i, x_j):
        
        l1_i = torch.tensor(0.0)
        if self.cfg['arch'] == 'grafp':
            x_i = self.peak_extractor(x_i)
            l1_i = self.peak_extractor.l1
        # print(f'Shape of x_i {x_i.shape} inside the SimCLR forward function')
        # print(f'Shape of x_j {x_j.shape} inside the SimCLR forward function')
        h_i = self.encoder(x_i)
        # print(f'Shape of h_i {h_i.shape} inside the SimCLR forward function')
        z_i = self.projector(h_i)
        # print(f'Shape of z_i {z_i.shape} inside the SimCLR forward function')
        z_i = F.normalize(z_i, p=2)

        l1_i = torch.tensor(0.0)
        if self.cfg['arch'] == 'grafp':
            x_j = self.peak_extractor(x_j)
            l1_j = self.peak_extractor.l1
        h_j = self.encoder(x_j)
        z_j = self.projector(h_j)
        z_j = F.normalize(z_j, p=2)


        return l1_i, l1_j, z_i, z_j