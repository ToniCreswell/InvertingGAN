import sys
sys.path.append('../')

from dataload import CELEBA
from utils import make_new_folder, plot_norm_losses, save_input_args, \
sample_z, class_loss_fn, plot_losses, corrupt, prep_data # one_hot
from models import GEN, DIS


import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy as bce

from torchvision import transforms
from torchvision.utils import make_grid, save_image

import numpy as np

import os
from os.path import join

import argparse

from PIL import Image

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from time import time

EPSILON = 1e-6

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', default='../../../../data/', type=str)
	parser.add_argument('--batchSize', default=128, type=int)
	parser.add_argument('--maxEpochs', default=200, type=int)
	parser.add_argument('--nz', default=100, type=int)
	parser.add_argument('--lr', default=2e-4, type=float)
	parser.add_argument('--fSize', default=64, type=int)  #multiple of filters to use
	parser.add_argument('--exDir', required=True, type=str)
	parser.add_argument('--gpuNo', default=0, type=int)

	return parser.parse_args()


def find_z(gen, x, nz, lr, exDir, maxEpochs=100):

	#generator in eval mode
	gen.eval()

	if gen.useCUDA:
		gen.cuda()

	Zinit = Variable(torch.randn(1,opts.nz).cuda(), requires_grad=True)

	#optimizer
	optZ = torch.optim.RMSprop([Zinit], lr=lr)

	losses = {'rec': []}
	for e in range(maxEpochs):

		xHAT = gen.forward(Zinit)
		# recLoss = F.mse_loss(x, xHAT)
		recLoss = (x - xHAT).pow(2).mean()

		optZ.zero_grad()
		recLoss.backward()
		optZ.step()

		losses['rec'].append(recLoss.data[0])
		print '[%d] loss: %0.5f' % (e, recLoss.data[0])

		#plot training losses
		if e>0:
			plot_losses(losses, exDir, e+1)

	#visualise the final output
	xHAT = gen.forward(Zinit)
	save_image(xHAT.data, join(exDir, 'rec'+str(e)+'.png'))
	save_image(x.data, join(exDir, 'original'+str(e)+'.png'))

	return Zinit















if __name__=='__main__':
	opts = get_args()

	#Create new subfolder for saving results and training params
	invDir = join(opts.exDir, 'inversionExperiments')
	try:
		os.mkdir(invDir)
	except:
		print 'already exists'

	exDir = make_new_folder(invDir)
	print 'Outputs will be saved to:',exDir
	save_input_args(exDir, opts)

	####### Test data set #######
	print 'Prepare data loaders...'
	transform = transforms.Compose([transforms.ToTensor(), \
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	testDataset = CELEBA(root=opts.root, train=False, transform=transform, Ntest=100)  #most models trained with Ntest=1000, but using 100 to prevent memory errors
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=1, shuffle=False)
	print 'Data loaders ready.'

	###### Create model and load parameters #####
	IM_SIZE = 64

	gen = GEN(imSize=IM_SIZE, nz=opts.nz, fSize=opts.fSize)
	gen.load_params(opts.exDir)
	print 'params loaded'


	#Find each z individually for each x
	for i, data in enumerate(testLoader):
		x, y = prep_data(data, useCUDA=gen.useCUDA)
		z = find_z(gen=gen, x=x, nz=opts.nz, lr=opts.lr, exDir=exDir, maxEpochs=opts.maxEpochs)

		break






