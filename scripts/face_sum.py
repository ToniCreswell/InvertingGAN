import sys
sys.path.append('../')

from dataload import CELEBA_ALL_LABELS as CELEBA
from models import GEN

from utils import make_new_folder, plot_norm_losses, save_input_args, \
sample_z, class_loss_fn, plot_losses, corrupt, prep_data

import torch
from torch import optim
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.functional import binary_cross_entropy as bce

from torchvision import transforms
from torchvision.utils import make_grid, save_image

from invert import find_z, get_args

import os
from os.path import join

import torch

import numpy as np

import matplotlib 
matplotlib.use('Agg')
from matplotlib import pyplot as plt


if __name__=='__main__':
	opts = get_args()
	opts.data = 'CELEBA'
	opts.imSize = 64
	# opts.batchSize = 100

	#Create new subfolder for saving results and training params
	exDir = join(opts.exDir, 'face_sum_experiments')
	try:
		os.mkdir(exDir)
	except:
		print 'already exists'

	print 'Outputs will be saved to:',exDir
	save_input_args(exDir, opts)

	# Load data (glasses and male labels)
	IM_SIZE = 64
	print 'Prepare data loader...'
	transform = transforms.Compose(
		[transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	testDataset = CELEBA(root=opts.root, train=False, labels=['Male', 'Eyeglasses'], transform=transform, Ntest=1000)  #most models trained with Ntest=1000, but using 100 to prevent memory errors
	testLoader = torch.utils.data.DataLoader(testDataset, batch_size=opts.batchSize, shuffle=False)
	print 'Data loader ready.'

	#Load model
	gen = GEN(imSize=IM_SIZE, nz=opts.nz, fSize=opts.fSize)
	if gen.useCUDA:
		torch.cuda.set_device(opts.gpuNo)
		gen.cuda()
	gen.load_params(opts.exDir, gpuNo=opts.gpuNo)
	print 'params loaded'

	# Get the data:
	data = iter(testLoader).next()
	x, y = prep_data(data, useCUDA=gen.useCUDA)
	
	# Get men with glasses
	idx_men_w_glasses = np.argwhere(torch.prod(y.data==torch.Tensor([1,1]).cuda(), dim=1))[0]
	img_men_w_glasses = x[idx_men_w_glasses[:10]]
	save_image(img_men_w_glasses.data, join(exDir,'img_men_w_glasses_original.png'), nrow=10, normalize=True)


	# Get men without glasses
	idx_men_wout_glasses = np.argwhere(torch.prod(y.data==torch.Tensor([1,0]).cuda(), dim=1))[0]
	img_men_wout_glasses = x[idx_men_wout_glasses[:10]]
	save_image(img_men_wout_glasses.data, join(exDir,'img_men_wout_glasses_original.png'), nrow=10, normalize=True)


	# Get womean without glasses
	idx_women_wout_glasses = np.argwhere(torch.prod(y.data==torch.Tensor([0,0]).cuda(), dim=1))[0]
	img_women_wout_glasses = x[idx_women_wout_glasses[:10]]
	save_image(img_women_wout_glasses.data, join(exDir,'img_women_wout_glasses_original.png'), nrow=10, normalize=True)

	# Put in one batch to make inversion faster:
	x_in = torch.cat([img_men_w_glasses, img_men_wout_glasses, img_women_wout_glasses], dim=0)
	print(np.shape(x_in))

	try:
		z_men_w_glasses = np.load(join(exDir, 'z_men_w_glasses.npy'))
		z_men_w_glasses = torch.Tensor(z_men_w_glasses).cuda()
		z_men_wout_glasses = np.load(join(exDir, 'z_men_wout_glasses.npy'))
		z_men_wout_glasses = torch.Tensor(z_men_wout_glasses).cuda()
		z_women_wout_glasses = np.load(join(exDir, 'z_women_wout_glasses.npy'))
		z_women_wout_glasses = torch.Tensor(z_women_wout_glasses).cuda()
	except:
		z_out = find_z(gen, x_in, nz=opts.nz, lr=opts.lr, exDir=exDir, maxEpochs=opts.maxEpochs)

		z_men_w_glasses = z_out[:10]
		z_men_wout_glasses = z_out[10:20]
		z_women_wout_glasses = z_out[10:]

		np.save(join(exDir, 'z_men_w_glasses.npy'), z_men_w_glasses.detach().cpu().numpy())
		np.save(join(exDir, 'z_men_wout_glasses.npy'), z_men_wout_glasses.detach().cpu().numpy())
		np.save(join(exDir, 'z_women_wout_glasses.npy'), z_women_wout_glasses.detach().cpu().numpy())

	z_mean_man_w_glasses = torch.mean(z_men_w_glasses, dim=0, keepdim=True)
	z_mean_man_wout_glasses = torch.mean(z_men_wout_glasses, dim=0, keepdim=True)
	z_mean_woman_wout_glasses = torch.mean(z_women_wout_glasses, dim=0, keepdim=True)

	img_mean_man_w_glasses = gen.forward(z_mean_man_w_glasses)
	img_mean_man_wout_glasses = gen.forward(z_mean_man_wout_glasses)
	img_mean_woman_wout_glasses = gen.forward(z_mean_woman_wout_glasses)

	save_image(img_mean_man_w_glasses, join(exDir,'img_mean_man_w_glasses.png', nrow=1, normalize=True))
	save_image(img_mean_man_wout_glasses, join(exDir,'img_mean_man_wout_glasses.png', nrow=1, normalize=True))
	save_image(img_mean_woman_wout_glasses, join(exDir,'img_mean_woman_wout_glasses.png', nrow=1, normalize=True))









