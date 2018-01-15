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


def find_z(gen, dataLoader, nz, lr, exDir, batchSize, maxEpochs=100):

	#generator in eval mode
	gen.eval()

	if gen.useCUDA:
		gen.cuda()

	#save subset of target images:
	xTarget = iter(dataLoader).next()
	save_image(xTarget[0], join(exDir, 'target.png'))


	#start with an initially random z
	#N.B. dataloader must not be shuffeling x
	Z = Variable(torch.randn(len(dataLoader), nz).cuda(), requires_grad=True)

	#optimizer
	optZ = torch.optim.RMSprop(params = [Z], lr=lr, momentum=0)

	losses = {'rec': []}
	for e in range(maxEpochs):
		epochLoss=0
		for i, data in enumerate(dataLoader):

			x, y = prep_data(data, useCUDA = gen.useCUDA)
			z = Z[i * batchSize : (i + 1) * batchSize]
			xHAT = gen.forward(z)

			loss = F.mse_loss(x, xHAT)

			optZ.zero_grad()
			loss.backwards()
			optZ.backwards()

			epochLoss+=loss

		losses.append(loss/(i+1))

		[]

		#plot training losses
		if e>0:
			plot_losses(losses)

		#visualise the training progress
		xHAT = gen,forward(z)
		save_image(xHAT.data, join(exDir, 'rec_'+str(e)+'.png'))

	return z















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
	testDataset = CELEBA(root=opts.root, train=False, transform=transform)
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)
	print 'Data loaders ready.'

	###### Create model and load parameters #####
	IM_SIZE = 64

	gen = GEN(imSize=IM_SIZE, nz=opts.nz, fSize=opts.fSize)
	gen.load_params(opts.exDir)

	z = find_z(gen=gen, dataLoader=testLoader, nz=opts.nz, lr=opts.lr, exDir=exDir, batchSize = opts.batchSize, maxEpochs=opts.maxEpochs)






