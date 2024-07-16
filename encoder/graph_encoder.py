import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from timm.models.layers import DropPath
from encoder.gcn_lib.torch_vertex import Grapher
from encoder.gcn_lib.torch_nn import act_layer, norm_layer, MLP, BasicConv
from timm.models.layers import to_2tuple,trunc_normal_
from timm.models.layers import DropPath
import timm
import torchvision 





class Stem(nn.Module):
    
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x
    
class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ChannelConv(nn.Module):

    def __init__(self,in_dim,out_dim):

        super().__init__()
        
        self.conv = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_dim))
                    
    
    def forward(self,x):
        x = self.conv(x)
        return x 
    
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu',drop_path=0.0):
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = hidden_features if hidden_features is not None else in_features
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.act = act_layer(act)
        self.fc1 = Seq(nn.Conv2d(in_features,hidden_features,1,stride=1,bias=False,padding=0),
                          nn.BatchNorm2d(hidden_features))
        self.fc2 = Seq(nn.Conv2d(hidden_features,out_features,1,stride=1,bias=False,padding=0),
                          nn.BatchNorm2d(out_features))
        
                          
       

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
       
        return x

class GraphEncoder(nn.Module):

    def __init__(self, cfg, k=3,conv='mr',act='relu',norm='batch',bias=True,dropout=0.0,dilation=True,epsilon=0.2,drop_path=0.1,size ='t',
               emb_dims=1024,in_channels=3):
        
        super().__init__()
        
        """
        Args:   
        Inputs:
            k: K for KNN
            conv: Type of graph conv. Mr-conv/edgeconv/GIN/GCN
            act: activation function
            norm: normalization type
            bias: True or False
            dropout: dropout value to be used
            dilation: Graph dilation 
            epsilon: epsilon
            size: size of the model . Either small or medium
            emb_dims: output emb_dim
            in_channels: channels of the input
            num_points: num points in the pointcloud
        Outputs:
            None  
        """
        
        # Different versions of the encoder 
        if size == 't':
            self.blocks = [2,2,6,2]
            self.channels = [64, 128, 256, 512]
            self.emb_dims = 1024
        elif size == 's':
            self.blocks = [2, 2, 6, 2]
            self.channels = [80, 160, 400, 640]
            self.emb_dims = 1024
        elif size == 'm':
            self.blocks = [2,2,16,2]
            self.channels = [96, 192, 384, 768]
            self.emb_dims = 1024
        else:
            self.blocks = [2,2,18,2]
            self.channels = [128, 256, 512, 1024]
        self.k = int(k)  # number of edges per node  
        self.act = act 
        self.norm = norm
        self.bias = bias
        self.drop_path = drop_path

        self.emb_dims = emb_dims
        self.epsilon = epsilon
        self.dilation = dilation
        self.dropout = dropout
        stochastic = False
        self.num_blocks = sum(self.blocks)
        self.conv = 'mr'
        N = cfg['n_mels'] * cfg['n_frames'] // cfg['peak_stride']


        num_k  = [int(x.item()) for x in torch.linspace(k,k,self.num_blocks)]
        max_dilation = 128//max(num_k) # max_dilation value 

        # Stem conv for extracting non linear representation from the points
        self.stem = nn.Sequential(nn.Conv2d(in_channels,self.channels[0], kernel_size=1, bias=False),
                                   nn.BatchNorm2d(self.channels[0]),
                                   nn.LeakyReLU(negative_slope=0.2))

        ############################### New Stem #############################
        # self.stem_1 = Stem(in_dim=in_channels, out_dim=self.channels[0],act='relu')
        ######################################################################          
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.num_blocks)]

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(self.blocks)):

            if i > 0:
                self.backbone.append(Downsample(self.channels[i-1], self.channels[i]))
                N = N // 4
            for j in range(self.blocks[i]):

                self.backbone += [
                        Seq(Grapher(self.channels[i], num_k[idx], min(idx // 4 + 1, max_dilation), self.conv, self.act, self.norm,
                                        self.bias, stochastic, epsilon, 1, n=N, drop_path=dpr[idx],
                                        relative_pos=True),
                            FFN(in_features=self.channels[i],hidden_features= self.channels[i] * 4,out_features=self.channels[i], act=act, drop_path=dpr[idx])
                            )]
        self.backbone = Seq(*self.backbone)

        # Linear projection for common subspace in contrastive learning

        self.proj = nn.Conv2d(self.channels[-1], 1024, 1, bias=True)
    
    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True
    
    def forward(self,x):
        
        """
        Args:   
        Inputs:
            x: Input with shape (B,C,num_points)
            
        Outputs:
            x: Output embedding with shape (B,emb_dim) # Batch,1024
        """
        
        x = x.unsqueeze(-1)
        
        B, C,N,_ = x.shape
        x = self.stem(x)

        for i in range(len(self.backbone)):
            
            x = self.backbone[i](x)
        
        x = self.proj(x)
        x = torch.mean(x,dim=2).squeeze(-1).squeeze(-1)
        
        
        return x 
    
    # def __call__(self, x):
    #     return self.forward(x)


if __name__ == '__main__':

    encoder = GraphEncoder()
    dummy_tensor = torch.rand(8,3,512)
    out = encoder.forward(dummy_tensor)
    

    
    



            

        




        
