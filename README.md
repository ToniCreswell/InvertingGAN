# Inverting the Generator of a Generative Adversarial Network

## To use code:
1. Download the celebA dataset from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
OR
  Download the Shoes dataset from http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images-square.zip
 OR
 	Download the Omniglot dataset from https://github.com/brendenlake/omniglot/tree/master/python
2. Install dependencies listed in req.txt
3. You will also need pyTorch which may be downloaded from http://pytorch.org
4. Run [this](https://github.com/ToniCreswell/attribute-cVAEGAN/blob/master/notebooks/DataToTensorCelebA_smileLabel.ipynb) Jupyter notebook to get the data tensors for CelebA
 OR
	Run the jupyter notebook to get the data tensors xShoes.npy and yShoes.npy and move them in to folder InData/SHOES/

5. The code may be run from cmd line with various options detailed in the code


## Example results:

![alt text](https://github.com/ToniCreswell/InvertingGAN/blob/master/imgs/shoes.png)
