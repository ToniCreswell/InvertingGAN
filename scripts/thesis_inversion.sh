#!/bin/bash

#To be run from monal inside /vol/biomedic2/ac2211/Projects/Project12_Visualisation/Code/scripts
GPUNO=1
#SHOES
python invert.py --alpha=0.01 --batchSize=500 --data='SHOES' --exDir=../../ExperimentsSHOES/Ex_3 --fSize=128 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
python invert.py --alpha=0.01 --batchSize=500 --data='SHOES' --exDir=../../ExperimentsSHOES/Ex_4 --fSize=128 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
python invert.py --alpha=0.01 --batchSize=500 --data='SHOES' --exDir=../../ExperimentsSHOES/Ex_5 --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
python invert.py --alpha=0.01 --batchSize=500 --data='SHOES' --exDir=../../ExperimentsSHOES_bicv/Ex_19\(GAN64+noise\) --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
# python invert.py --alpha=0.01 --batchSize=500 --data='SHOES' --exDir=../../ExperimentsSHOES/Ex_7 --fSize=64 --gpuNo=1 --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
#Still need the BICV experiment too!

#OMNIGLOT
python invert.py --alpha=0.0 --batchSize=500 --data='OMNI' --exDir=../../ExperimentsOMNI_bicv/Ex_1\(GAN\) --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
python invert.py --alpha=0.0 --batchSize=500 --data='OMNI' --exDir=../../ExperimentsOMNI_bicv/Ex_3\(WGAN\) --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=200 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
python invert.py --alpha=0.0 --batchSize=500 --data='OMNI' --exDir=../../ExperimentsOMNI/Ex30 --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000

#CELEBA
python invert.py --alpha=0.01 --batchSize=500 --data='CELEBA' --exDir=../../Experiments_bicv/Ex_19\(GAN+noise\) --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
python invert.py --alpha=0.01 --batchSize=500 --data='CELEBA' --exDir=../../Experiments_bicv/Ex_20\(GAN\) --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
python invert.py --alpha=0.01 --batchSize=500 --data='CELEBA' --exDir=../../Experiments_bicv/Ex_23\(WGAN\) --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
