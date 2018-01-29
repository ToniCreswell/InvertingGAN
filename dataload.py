import numpy as np

import os
from os.path import join

from torch.utils import data

from torchvision import transforms, datasets

from PIL import Image


class CELEBA(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset where directory
            ``celebA`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """


    def __init__(self, root, train=True, transform=None, label='Smiling', Ntest=1000):
        attributes = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.filename='celebA'
        self.transform=transform
        self.idx = attributes.index(label)
        print self.idx

        # now load the picked numpy arrays
        if self.train:
            self.train_data = np.load(join(self.root, self.filename, 'xTrain.npy'), mmap_mode='r')[Ntest:]
            self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
            train_labels = np.load(join(self.root, self.filename, 'yAllTrain.npy'))[Ntest:, self.idx]
            self.train_labels = (train_labels.astype(int)+1) /2
            print np.shape(self.train_labels), np.shape(self.train_data)
            print np.unique(self.train_labels)

        else:
            self.test_data = np.load(join(self.root, self.filename, 'xTrain.npy'), mmap_mode='r')[:Ntest]
            self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
            test_labels = np.load(join(self.root, self.filename, 'yAllTrain.npy'))[:Ntest, self.idx]
            self.test_labels = (test_labels.astype(int)+1) /2


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
            

        # doing this so that it is consistent with all other datasets

        if self.transform is not None:
            img = self.transform(img)

        target = target.astype(int)


        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_dir_exist(self):
        inDir=join(self.root, self.filename)
        assert os.path.isdir(inDir)
        assert os.path.exists(join(inDir, 'xTrain.npy'))
        assert os.path.exists(join(inDir, 'yAllTrain.npy'))


class SHOES(data.Dataset):
    """
    Args:
        root (string): Root directory of dataset where directory
            ``SHOES`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        grain: 'None', 'fine', 'subfine'
            None: Boots','Sandals', 'Shoes', 'Slippers'
            fine: c. 20 sub categories
            subfine: c.3000 subsub categories
    """


    def __init__(self, root, train=True, grain=None, transform=None, Ntest=1000):
        

        self.root = os.path.expanduser(root)
        self.train = train  # training set or test set
        self.filename='SHOES'
        self.transform=transform
        self.labels = ['Boots','Sandals', 'Shoes', 'Slippers']
        self.grain = grain

        ####### Shuffel data same way each time #######
        #load data
        xData = np.load(join(self.root, self.filename, 'xShoes.npy'), mmap_mode='r')
        if grain is None:  #get right grandularity of labels
            yData = np.load(join(self.root, self.filename, 'yShoes.npy'), mmap_mode='r')
        else:
            yData = np.load(join(self.root, self.filename, grain+'Shoes.npy'), mmap_mode='r')
        
        #shuffel data
        np.random.seed(1993)
        rndIdx = np.random.permutation(np.shape(xData)[0])
        xData = xData[rndIdx]
        yData = yData[rndIdx]

        # now load the picked numpy arrays
        if self.train:
            self.train_data = xData[Ntest:]
            self.train_labels = yData[Ntest:]
            self.train_labels = self.train_labels.astype(int)
            print np.shape(self.train_labels), np.shape(self.train_data)
            print np.unique(self.train_labels)

        else: #test
            self.test_data = xData[:Ntest]
            self.test_labels = yData[:Ntest]
            self.test_labels = self.test_labels.astype(int)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]
            

        # doing this so that it is consistent with all other datasets

        if self.transform is not None:
            img = self.transform(img)

        target = target.astype(int)


        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)



