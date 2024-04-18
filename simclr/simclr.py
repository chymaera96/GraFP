import torch
import torch.nn as nn
import torch.nn.functional as F


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

        self.projector = nn.Sequential(nn.Linear(h, d*u),
                                       nn.ELU(),
                                       nn.Linear(d*u, d)
                               )

        # self.projector = nn.Sequential(nn.Conv1d(h, d * u, kernel_size=(1,), groups=d),
        #                                 nn.ELU(),
        #                                 nn.Conv1d(d * u, d, kernel_size=(1,), groups=d)
        #                                 )

    def forward(self, x_i, x_j):
        
        print(f'Shape of x_i {x_i.shape} inside the SimCLR forward function')
        h_i = self.encoder(x_i)
        # print(f'Shape of h_i {h_i.shape} inside the SimCLR forward function')
        z_i = self.projector(h_i)
        # print(f'Shape of z_i {z_i.shape} inside the SimCLR forward function')
        z_i = F.normalize(z_i, p=2)

        h_j = self.encoder(x_j)
        z_j = self.projector(h_j)
        z_j = F.normalize(z_j, p=2)


        return h_i, h_j, z_i, z_j
