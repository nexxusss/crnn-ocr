from pyparsing import Any
import torch
from torch.utils.data import Dataset
from torch.utils.data import sampler
from torchvision.transforms import transforms


from PIL import Image
import numpy as np

import random
from io import BytesIO
import sys
import lmdb

class lmdbDataset(Dataset):
    def __init__(self, root=None, transform=None, target_transform=None):
        self.root = root
        self.transfrom = transform
        self.target_transform = target_transform

        self.env = lmdb.open(
            root=self.root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        
        self.txn = self.env.begin(write=False)

        if not self.env:
            print(f"failure to create the lmdb from {root}")
            sys.exit(0)

        n_sample = int(self.txn.get('num-samples'))
        self.n_samples = n_sample

    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index) -> Any:
        if index > len(self):
            raise IndexError(f'Index {index} is out of range')
        
        index += 1

        img_key = f'image-{index:09d}'
        img_buf = self.txn.get(img_key)

        buf = BytesIO()
        buf.write(img_buf)
        buf.seek(0)

        try:
            img = Image.open(buf).convert('L')
        except IOError:
            print(f'Image {index}, with key {img_key} is potentially corrupted')
            return self[index + 1]

        # applying transform on inputs
        if self.transfrom is not None:
            img = self.transfrom(img)

        #fetching label keys
        label_key = f'label-{index:09d}'
        label = str(self.txn.get(label_key), 'utf-8')
        
        #applying transforms on labels
        if self.target_transform is not None:
            label = self.target_transform(label)
        
        return img, label

class NormResize():
    """
    Class to encapsulate the logic for normalizing and resizing the images
    """   

    def __init__(self, size, interpolation=Image.BILINEAR) -> None:

        # using transforms.Compose to combine all transformations needed
        self.transforms = transforms.Compose([
            transforms.Resize(size, interpolation=interpolation),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __call__(self, image) -> Any:
        return self.transforms(image)
    
class SequentialSampler():

    def __init__(self, image_source, batch_size):
        self.num_samples = len(image_source)
        self.batch_size = batch_size

    def __iter__(self):

        # generate a random permutation of indices, which is more efficient than manually shuffling
        indices = torch.randperm(len(self)).tolist()

        # pad to ensure the last batch is not truncated
        indices += indices[:(self.num_samples % self.batch_size)]
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


        
        