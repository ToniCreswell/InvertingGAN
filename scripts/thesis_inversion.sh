#!/bin/bash

#To be run from monal inside /vol/biomedic2/ac2211/Projects/Project12_Visualisation/Code/scripts
GPUNO=1
#SHOES
python invert.py --alpha=0.01 --batchSize=500 --data='SHOES' --exDir=../../ExperimentsSHOES/Ex_3 --fSize=64 --gpuNo=$GPUNO --imSize=128 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000 #TODO
python invert.py --alpha=0.01 --batchSize=500 --data='SHOES' --exDir=../../ExperimentsSHOES/Ex_4 --fSize=64 --gpuNo=$GPUNO --imSize=128 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000 #TODO
# python invert.py --alpha=0.01 --batchSize=500 --data='SHOES' --exDir=../../ExperimentsSHOES/Ex_5 --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
# python invert.py --alpha=0.01 --batchSize=500 --data='SHOES' --exDir=../../ExperimentsSHOES/Ex_7 --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
# python invert.py --alpha=0.01 --batchSize=500 --data='SHOES' --exDir=../../ExperimentsSHOES_bicv/Ex_19_GAN64+noise --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
python invert.py --alpha=0.01 --batchSize=500 --data='SHOES' --exDir=../../ExperimentsSHOES_bicv/Ex_21_GAN64 --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000 #TODO
#Still need the BICV experiment too!

#OMNIGLOT
# python invert.py --alpha=0.0 --batchSize=500 --data='OMNI' --exDir=../../ExperimentsOMNI_bicv/Ex_1_GAN --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
# python invert.py --alpha=0.0 --batchSize=500 --data='OMNI' --exDir=../../ExperimentsOMNI_bicv/Ex_3_WGAN --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=200 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
python invert.py --alpha=0.0 --batchSize=500 --data='OMNI' --exDir=../../ExperimentsOMNI/Ex_30 --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000  #TODO

#CELEBA
# python invert.py --alpha=0.01 --batchSize=500 --data='CELEBA' --exDir=../../Experiments_bicv/Ex_19_GAN+noise --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
# python invert.py --alpha=0.01 --batchSize=500 --data='CELEBA' --exDir=../../Experiments_bicv/Ex_20_GAN --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
# python invert.py --alpha=0.01 --batchSize=500 --data='CELEBA' --exDir=../../Experiments_bicv/Ex_23_WGAN --fSize=64 --gpuNo=$GPUNO --imSize=64 --lr=0.01 --nz=100 --root=/vol/biomedic2/ac2211/data/ --maxEpochs=5000
