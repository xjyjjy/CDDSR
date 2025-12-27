import os
import json
from PIL import Image

import pickle
import imageio
import random
import numpy as np
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets.data import Dataset as Dataset_1
from torchvision import transforms

from utils import extract_number
from datasets import register

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none', rank=True):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        filenames = sorted(filenames, key=extract_number)
        if first_k is not None:
            filenames = filenames[:first_k]

        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with open(bin_file, 'wb') as f:
                        pickle.dump(imageio.imread(file), f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                self.files.append(transforms.ToTensor()(
                    Image.open(file).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            return transforms.ToTensor()(Image.open(x).convert('RGB'))

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
            x = np.ascontiguousarray(x.transpose(2, 0, 1))
            x = torch.from_numpy(x).float() / 255
            return x

        elif self.cache == 'in_memory':
            # print(type(x), x.max(), x.min())
            return x

@register('numpy-folder')
class NumpyFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                start=None, end=None,repeat=1, cache='none', rank=True):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        if rank :
            filenames = sorted(filenames, key=extract_number)
        else:
            filenames =  sorted(filenames)
        if first_k is not None and not("ware" in root_path):
            filenames = filenames[:first_k]
        if start is not None:
            filenames = filenames[start:end]
        self.files = []
        for filename in filenames:
            file = os.path.join(root_path, filename)

            if cache == 'none':
                self.files.append(file)

            elif cache == 'bin':
                bin_root = os.path.join(os.path.dirname(root_path),
                    '_bin_' + os.path.basename(root_path))
                if not os.path.exists(bin_root):
                    os.mkdir(bin_root)
                    print('mkdir', bin_root)
                bin_file = os.path.join(
                    bin_root, filename.split('.')[0] + '.pkl')
                if not os.path.exists(bin_file):
                    with np.load(file) as npzfile:
                        array = npzfile['arr_0']
                    with open(bin_file, 'wb') as f:
                        pickle.dump(array, f)
                    print('dump', bin_file)
                self.files.append(bin_file)

            elif cache == 'in_memory':
                with np.load(file) as npzfile:
                    array = npzfile['arr_0']
                # if array.ndim == 2:
                #     array = np.expand_dims(array, 0)
                # print((array.shape))
                # self.files.append(transforms.ToTensor()(array))
                self.files.append(torch.from_numpy(array.astype('float64')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        x = self.files[idx % len(self.files)]

        if self.cache == 'none':
            with np.load(x) as npzfile:
                array = npzfile['arr_0']
            return transforms.ToTensor()(array)

        elif self.cache == 'bin':
            with open(x, 'rb') as f:
                x = pickle.load(f)
                x = torch.from_numpy(x.astype('float64'))
            return x

        elif self.cache == 'in_memory':
            return x





@register('five-image-folders')
class FiveImageFolders(Dataset):
    def __init__(self, root_path_HR, root_path_LR, root_path_Depth, root_path_Label, root_path_Kernel,**kwargs):
        self.dataset_1 = ImageFolder(root_path_HR, **kwargs)
        self.dataset_2 = ImageFolder(root_path_LR, **kwargs)
        self.dataset_3 = NumpyFolder(root_path_Depth, **kwargs)
        self.dataset_4 = NumpyFolder(root_path_Label, **kwargs)
        self.dataset_5 = NumpyFolder(root_path_Kernel, **kwargs)
        #self.kernel_warehuose = kernel_warehuose
        #self.root_path_KernelWarehouse = NumpyFolder(root_path_KernelWarehouse, **kwargs)

    def __len__(self):
        return len(self.dataset_1)
        # return int((len(self.dataset_1) + len(self.dataset_2) + len(self.dataset_3) 
        #          + len(self.dataset_4) + len(self.dataset_5))/5)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx], self.dataset_3[idx], self.dataset_4[idx],self.dataset_5[idx]#, self.kernel_warehuose
