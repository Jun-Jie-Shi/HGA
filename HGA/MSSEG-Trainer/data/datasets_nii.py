import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .data_utils import pkload
from .rand import Uniform
from .transforms import (CenterCrop, Compose, Flip, GaussianBlur, Identity,
                         Noise, Normalize, NumpyType, Pad, RandCrop, CenterCrop3D,
                         RandCrop3D, RandomFlip, RandomIntensityChange,
                         RandomRotion, RandSelect, Rot90)
import glob
import random


# mask_array = np.array([[False, False, False, False, True], [False, False, False, True, False], [False, False, True, False, False], [False, True, False, False, False], [True, False, False, False, False],
#         [False, False, False, True, True], [False, False, True, False, True], [False, True, False, False, True], [True, False, False, False, True], [False, False, True, True, False], [False, True, False, True, False], [True, False, False, True, False], [False, True, True, False, False], [True, False, True, False, False], [True, True, False, False, False],
#         [False, False, True, True, True], [False, True, False, True, True], [True, False, False, True, True], [False, True, True, False, True], [True, False, True, False, True], [True, True, False, False, True], [False, True, True, True, False], [True, False, True, True, False], [True, True, False, True, False], [True, True, True, False, False],
#         [False, True, True, True, True], [True, False, True, True, True], [True, True, False, True, True], [True, True, True, False, True], [True, True, True, True, False],
#         [True, True, True, True, True]])

mask_array = np.array([[False, False, False, False, True], [False, False, False, True, False], [False, False, True, False, False], [False, True, False, False, False], [True, False, False, False, False],
        [False, False, False, True, True], [False, False, True, False, True], [False, True, False, False, True], [True, False, False, False, True], [False, False, True, True, False], [False, True, False, True, False], [True, False, False, True, False], [False, True, True, False, False], [True, False, True, False, False], [True, True, False, False, False],
        [False, False, True, True, True], [False, True, False, True, True], [True, False, False, True, True], [False, True, True, False, True], [True, False, True, False, True], [True, True, False, False, True], [False, True, True, True, False], [True, False, True, True, False], [True, True, False, True, False], [True, True, True, False, False],
        [False, True, True, True, True], [True, False, True, True, True], [True, True, False, True, True], [True, True, True, False, True], [True, True, True, True, False],
        [True, True, True, True, True]])

# mask_valid_array = np.array([[False, False, True, False],
#             [False, True, True, False],
#             [True, True, False, True],
#             [True, True, True, True]])

class MSSEG_loadall_train_nii_idt_wimaskvalue(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=2, mask_type='idt', train_file=None):
        self.excel_path = train_file
        excel_data = pd.read_csv(self.excel_path)
        data_name = excel_data['data_name']
        data_name_list = data_name.values.tolist()
        mask_id = excel_data['mask_id']
        mask_id_list = mask_id.values.tolist()
        samples_mask = excel_data['mask']
        samples_mask_list = samples_mask.values.tolist()
        pos_mask_id = excel_data['pos_mask_ids']
        pos_mask_id_list = pos_mask_id.values.tolist()

        volpaths = []
        for dataname in data_name_list:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = data_name_list
        self.mask_ids = mask_id_list
        self.samples_masks = samples_mask_list
        self.pos_mask_ids = pos_mask_id_list
        self.num_cls = num_cls
        self.mask_type = mask_type
        if modal == 'all':
            self.modal_ind = np.array([0,1,2,3,4])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        if self.mask_type == 'idt':
            mask_idx = np.array([self.mask_ids[index]])
        elif self.mask_type == 'idt_drop':
            mask_idx = np.random.choice(eval(self.pos_mask_ids[index]),1)
        elif self.mask_type == 'pdt':
            mask_idx = np.random.choice(31, 1)

        return mask_idx

    def __len__(self):
        return len(self.volpaths)

class MSSEG_loadall_train_nii_idt(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=2, mask_type='idt', train_file='/home/sjj/MMMSeg/2D/data/myops_split/MyoPS2020_longtail_split_357.csv'):
        self.excel_path = train_file
        excel_data = pd.read_csv(self.excel_path)
        data_name = excel_data['data_name']
        data_name_list = data_name.values.tolist()
        mask_id = excel_data['mask_id']
        mask_id_list = mask_id.values.tolist()
        samples_mask = excel_data['mask']
        samples_mask_list = samples_mask.values.tolist()
        pos_mask_id = excel_data['pos_mask_ids']
        pos_mask_id_list = pos_mask_id.values.tolist()

        volpaths = []
        for dataname in data_name_list:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = data_name_list
        self.mask_ids = mask_id_list
        self.samples_masks = samples_mask_list
        self.pos_mask_ids = pos_mask_id_list
        self.num_cls = num_cls
        self.mask_type = mask_type

        if modal == 'all':
            self.modal_ind = np.array([0,1,2,3,4])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        if self.mask_type == 'idt':
            mask_idx = np.array([self.mask_ids[index]])
        elif self.mask_type == 'pdt':
            mask_idx = np.random.choice(31, 1)
            # mask_idx = np.array([self.mask_ids[index]])
            # mask_idx = np.array([14])
        x = np.expand_dims(np.load(volpath),axis=2)
        segpath = volpath.replace('vol', 'seg')
        y = np.expand_dims(np.load(segpath),axis=2)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]
        # xo = np.zeros_like(x)
        # modal_exist = eval(self.samples_masks[index])
        # xo[:, modal_exist, :, :, :] = x[:, modal_exist, :, :, :]
        # xo=x
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        # modal_weight = sum(samples_num)/(samples_num[mask_idx.item()]*len(samples_num))
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)

class MSSEG_loadall_metaval_nii_idt(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=2, mask_value=30, train_file=None):
        self.excel_path = train_file
        self.mask_value = mask_value
        excel_data_all = pd.read_csv(self.excel_path)

        excel_data = excel_data_all[excel_data_all['pos_mask_ids'].apply(lambda x: mask_value in eval(x))]
        data_name = excel_data['data_name']
        data_name_list = data_name.values.tolist()
        mask_id = excel_data['mask_id']
        mask_id_list = mask_id.values.tolist()
        samples_mask = excel_data['mask']
        samples_mask_list = samples_mask.values.tolist()
        pos_mask_id = excel_data['pos_mask_ids']
        pos_mask_id_list = pos_mask_id.values.tolist()

        volpaths = []
        for dataname in data_name_list:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = data_name_list
        self.mask_ids = mask_id_list
        self.samples_masks = samples_mask_list
        self.pos_mask_ids = pos_mask_id_list
        self.num_cls = num_cls
        # self.mask_type = mask_type
        if modal == 'all':
            self.modal_ind = np.array([0,1,2,3,4])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        # if self.mask_type == 'nosplit_longtail':
        #     mask_idx = np.array([self.mask_ids[index]])
        # elif self.mask_type == 'split_longtail':
        #     mask_idx = np.random.choice(eval(self.pos_mask_ids[index]),1)
        # elif self.mask_type == 'random_all':
        #     mask_idx = np.random.choice(7, 1)
        #     # mask_idx = np.array([self.mask_ids[index]])
        #     # mask_idx = np.array([14])
        mask_idx = np.array([self.mask_value])
        # x = np.load(volpath)
        x = np.expand_dims(np.load(volpath),axis=2)
        segpath = volpath.replace('vol', 'seg')
        y = np.expand_dims(np.load(segpath),axis=2)
        # y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]
        # xo = np.zeros_like(x)
        # modal_exist = eval(self.samples_masks[index])
        # xo[:, modal_exist, :, :, :] = x[:, modal_exist, :, :, :]
        # xo=x
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        # modal_weight = sum(samples_num)/(samples_num[mask_idx.item()]*len(samples_num))
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)

class MSSEG_loadall_test_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', test_file='test.txt'):
        data_file_path = os.path.join(root, test_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()
        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        if modal == 'all':
            self.modal_ind = np.array([0,1,2,3,4])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        # x = np.load(volpath)
        x = np.expand_dims(np.load(volpath),axis=2)
        segpath = volpath.replace('vol', 'seg')
        # y = np.load(segpath).astype(np.uint8)
        y = np.expand_dims(np.load(segpath),axis=2).astype(np.uint8)
        x, y = x[None, ...], y[None, ...]
        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        y = np.ascontiguousarray(y)

        x = x[:, self.modal_ind, :, :, :]
        x = torch.squeeze(torch.from_numpy(x), dim=0)
        y = torch.squeeze(torch.from_numpy(y), dim=0)

        return x, y, name

    def __len__(self):
        return len(self.volpaths)

class MSSEG_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=2, train_file='val.txt'):
        data_file_path = os.path.join(root, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'all':
            self.modal_ind = np.array([0,1,2,3,4])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.expand_dims(np.load(volpath),axis=2)
        segpath = volpath.replace('vol', 'seg')
        y = np.expand_dims(np.load(segpath),axis=2).astype(np.uint8)
        
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        # mask_idx = np.random.choice(4, 1)
        # mask = torch.squeeze(torch.from_numpy(mask_valid_array[mask_idx]), dim=0)
        return x, yo, name

    def __len__(self):
        return len(self.volpaths)
    
class MSSEG_loadall_val_nii_idt(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=2, train_file='val.txt'):
        data_file_path = os.path.join(root, train_file)
        with open(data_file_path, 'r') as f:
            datalist = [i.strip() for i in f.readlines()]
        datalist.sort()

        volpaths = []
        for dataname in datalist:
            volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

        self.volpaths = volpaths
        self.transforms = eval(transforms or 'Identity()')
        self.names = datalist
        self.num_cls = num_cls
        if modal == 'all':
            self.modal_ind = np.array([0,1,2,3,4])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.expand_dims(np.load(volpath),axis=2)
        segpath = volpath.replace('vol', 'seg')
        y = np.expand_dims(np.load(segpath),axis=2).astype(np.uint8)
        
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        _, H, W, Z = np.shape(y)
        y = np.reshape(y, (-1))
        one_hot_targets = np.eye(self.num_cls)[y]
        yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
        yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        # mask_idx = np.random.choice(4, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[30]), dim=0)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)
