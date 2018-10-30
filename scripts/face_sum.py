import sys
sys.path.append('../')

from dataload import CELEBA_ALL_LABELS as CELEBA

from invert import find_z, get_args





if __name__=='__main__':
	opts = get_args()
	opts.data = 'CELEBA'
	opts.imSize = 64

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
	print 'Prepare data loaders...'
	transform = transforms.Compose(
		[transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	testDataset = CELEBA(root=opts.root, train=False, labels=['Male', 'Eyeglasses'] transform=transform, Ntest=1000)  #most models trained with Ntest=1000, but using 100 to prevent memory errors
	gen = GEN(imSize=IM_SIZE, nz=opts.nz, fSize=opts.fSize)

	# Load model


	# Get men with glasses


	# Get men without glasses


	# Get womean without glasses
