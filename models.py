#DCGAN MODELS
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm

import os
from os.path import join

class GEN(nn.Module):

	def __init__(self, imSize, nz=100, fSize=2):
		super(dcGEN, self).__init__()

		self.nz = nz
		self.prior = prior

		inSize = imSize // (2 ** 4)
		self.inSize = inSize

		self.useCUDA = torch.cuda.is_available()

		self.gen1 = nn.Linear(nz, (fSize * 8) * inSize * inSize)
		self.gen2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
		self.gen2b = nn.BatchNorm2d(fSize * 4)
		self.gen3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
		self.gen3b = nn.BatchNorm2d(fSize * 2)
		self.gen4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
		self.gen4b = nn.BatchNorm2d(fSize)
		self.gen5 = nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)

	def sample_z(self, no_samples):
		return torch.randn(no_samples, self.nz)

	def gen(self, z):

		lrelu = torch.nn.LeakyReLU(0.2)

		x = lrelu(self.gen1(z))
		x = x.view(z.size(0), -1, self.inSize, self.inSize)
		x = lrelu(self.gen2b(self.gen2(x)))
		x = lrelu(self.gen3b(self.gen3(x)))
		x = lrelu(self.gen4b(self.gen4(x)))
		x = F.tanh(self.gen5(x))

		return x

	def forward(self, z):
		return self.gen(z)

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'gen_params'))

	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'gen_params')))


class DIS(nn.Module):
	def __init__(self, imSize, fSize=2):
		super(dcDIS, self).__init__()

		self.fSize = fSize
		self.imSize = imSize

		inSize = imSize / ( 2 ** 4)

		self.dis1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)  #6 cause 2 images are concatenated in the feature channel
		self.dis1b = nn.BatchNorm2d(fSize)
		self.dis2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
		self.dis2b = nn.BatchNorm2d(fSize * 2)
		self.dis3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
		self.dis3b = nn.BatchNorm2d(fSize * 4)
		self.dis4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
		self.dis4b = nn.BatchNorm2d(fSize * 8)
		self.dis5 = nn.Linear((fSize * 8) * inSize * inSize, 1)
	
		self.useCUDA = torch.cuda.is_available()


	def dis(self, x):

		lrelu = torch.nn.LeakyReLU(0.2)

		d = lrelu(self.dis1b(self.dis1(x)))
		d = lrelu(self.dis2b(self.dis2(d)))
		d = lrelu(self.dis3b(self.dis3(d)))
		d = lrelu(self.dis4b(self.dis4(d)))
		d = d.view(x.size(0), -1)
		d = F.sigmoid(self.dis5(d)) #no sigmoid cause is a WGAN

		return d

	def forward(self, x):
		return self.dis(x)

	def save_params(self, exDir):
		print 'saving params...'
		torch.save(self.state_dict(), join(exDir, 'dis_params'))


	def load_params(self, exDir):
		print 'loading params...'
		self.load_state_dict(torch.load(join(exDir, 'dis_params')))

	def ones(self, N):
		ones = Variable(torch.Tensor(N,1).fill_(1))
		if self.useCUDA:
			return ones.cuda()
		else:
			return ones

	def zeros(self, N):
		zeros = Variable(torch.Tensor(N,1).fill_(0))
		if self.useCUDA:
			return zeros.cuda()
		else:
			return zeros

	def corrupt(x, level=0.003):
		noise = sigma * Variable(torch.rand(x.size())).type_as(x)
		return x + noise

 