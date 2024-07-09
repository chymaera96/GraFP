import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

d = 128
h = 1024
u = 32
v = int(h/d)
chang_fp = [d,d,2*d,2*d,4*d,4*d,h,h]

class NAFPEncoder(nn.Module):
	def __init__(self, in_channels=1, stride=2, kernel_size=3):
		super(NAFPEncoder, self).__init__()
		self.in_channels = in_channels
		self.stride = stride
		self.kernel_size = kernel_size
		self.conv_layers = self.create_conv_layers(chang_fp)

	def forward(self, x):
		x = self.conv_layers(x)
		x = x.reshape(x.shape[0],-1)
		return x


	def create_conv_layers(self, architecture):
		layers = []
		in_channels = self.in_channels
		kernel_size = self.kernel_size
		stride = self.stride
		shape = [1,256,32]
		for channels in architecture:

			layers.append(nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=(1,kernel_size), stride=(1,stride), padding=(0,1)))
			shape[0] = channels
			shape[2] = int(np.ceil(shape[2]/2))
			layers.append(nn.LayerNorm(shape))
			layers.append(nn.ReLU())
			layers.append(nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=(kernel_size,1), stride=(stride,1), padding=(1,0)))
			shape[1] = int(np.ceil(shape[1]/2))
			layers.append(nn.LayerNorm(shape))
			layers.append(nn.ReLU())

			in_channels = channels

		return nn.Sequential(*layers)