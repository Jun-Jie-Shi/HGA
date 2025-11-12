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

HGG = []
LGG = []
for i in range(0, 260):
    HGG.append(str(i).zfill(3))
for i in range(336, 370):
    HGG.append(str(i).zfill(3))
for i in range(260, 336):
    LGG.append(str(i).zfill(3))

# mask_array = np.array([[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
#          [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
#          [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
#          [True, True, True, True]])

mask_array = np.array([[False, False, True], [False, True, False], [True, False, False],
        [False, True, True], [True, True, False], [True, False, True],
        [True, True, True]])

mask_valid_array = np.array([[False, False, True, False],
            [False, True, True, False],
            [True, True, False, True],
            [True, True, True, True]])
# samples_num = [1,5,17,2,
#                 4,48,10,8,1,5,
#                 48,7,4,25,34]

# class Brats_loadall_train_nii_random(Dataset):
#     def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
#         data_file_path = os.path.join(root, train_file)
#         with open(data_file_path, 'r') as f:
#             datalist = [i.strip() for i in f.readlines()]
#         datalist.sort()

#         volpaths = []
#         for dataname in datalist:
#             volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))

#         self.volpaths = volpaths
#         self.transforms = eval(transforms or 'Identity()')
#         self.names = datalist
#         self.num_cls = num_cls
#         if modal == 'flair':
#             self.modal_ind = np.array([0])
#         elif modal == 't1ce':
#             self.modal_ind = np.array([1])
#         elif modal == 't1':
#             self.modal_ind = np.array([2])
#         elif modal == 't2':
#             self.modal_ind = np.array([3])
#         elif modal == 'all':
#             self.modal_ind = np.array([0,1,2,3])

#     def __getitem__(self, index):

#         volpath = self.volpaths[index]
#         name = self.names[index]
        
#         x = np.load(volpath)
#         segpath = volpath.replace('vol', 'seg')
#         y = np.load(segpath)
#         x, y = x[None, ...], y[None, ...]

#         x,y = self.transforms([x, y])

#         x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
#         _, H, W, Z = np.shape(y)
#         y = np.reshape(y, (-1))
#         one_hot_targets = np.eye(self.num_cls)[y]
#         yo = np.reshape(one_hot_targets, (1, H, W, Z, -1))
#         yo = np.ascontiguousarray(yo.transpose(0, 4, 1, 2, 3))

#         x = x[:, self.modal_ind, :, :, :]

#         x = torch.squeeze(torch.from_numpy(x), dim=0)
#         yo = torch.squeeze(torch.from_numpy(yo), dim=0)

#         mask_idx = np.random.choice(15, 1)
#         mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
#         return x, yo, mask, name

#     def __len__(self):
#         return len(self.volpaths)

class Myops_loadall_train_nii_longtail(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=6, mask_type='nosplit_longtail', train_file='/home/sjj/MMMSeg/2D/data/myops_split/MyoPS2020_longtail_split_357.csv'):
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
        if modal == 'bssfp':
            self.modal_ind = np.array([0])
        elif modal == 'lge':
            self.modal_ind = np.array([1])
        elif modal == 't2':
            self.modal_ind = np.array([2])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        if self.mask_type == 'nosplit_longtail':
            mask_idx = np.array([self.mask_ids[index]])
        elif self.mask_type == 'split_longtail':
            mask_idx = np.random.choice(eval(self.pos_mask_ids[index]),1)
        elif self.mask_type == 'random_all':
            mask_idx = np.random.choice(7, 1)
            # mask_idx = np.array([self.mask_ids[index]])
            # mask_idx = np.array([14])
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
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
    
class Myops_loadall_metaval_nii_longtail(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=6, mask_value=6, train_file=None):
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
        if modal == 'bssfp':
            self.modal_ind = np.array([0])
        elif modal == 'lge':
            self.modal_ind = np.array([1])
        elif modal == 't2':
            self.modal_ind = np.array([2])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2])

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
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
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

class Myops_loadall_test_nii(Dataset):
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
        if modal == 'bssfp':
            self.modal_ind = np.array([0])
        elif modal == 'lge':
            self.modal_ind = np.array([1])
        elif modal == 't2':
            self.modal_ind = np.array([2])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath).astype(np.uint8)
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

class Brats_loadall_train_nii_d2net(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=4, train_file='train.txt'):
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
        if modal == 'flair':
            self.modal_ind = np.array([0])
        elif modal == 't1ce':
            self.modal_ind = np.array([1])
        elif modal == 't1':
            self.modal_ind = np.array([2])
        elif modal == 't2':
            self.modal_ind = np.array([3])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2,3])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
        x, y = x[None, ...], y[None, ...]

        x,y = self.transforms([x, y])

        x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
        yo = np.ascontiguousarray(y)

        x = x[:, self.modal_ind, :, :, :]

        x = torch.squeeze(torch.from_numpy(x), dim=0)
        yo = torch.squeeze(torch.from_numpy(yo), dim=0)

        mask_idx = np.random.choice(15, 1)
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)

# class Brats_loadall_val_nii(Dataset):
#     def __init__(self, transforms='', root=None, settype='train', modal='all'):
#         data_file_path = os.path.join(root, 'val.txt')
#         with open(data_file_path, 'r') as f:
#             datalist = [i.strip() for i in f.readlines()]
#         datalist.sort()
#         volpaths = []
#         for dataname in datalist:
#             volpaths.append(os.path.join(root, 'vol', dataname+'_vol.npy'))
#         self.volpaths = volpaths
#         self.transforms = eval(transforms or 'Identity()')
#         self.names = datalist
#         if modal == 'flair':
#             self.modal_ind = np.array([0])
#         elif modal == 't1ce':
#             self.modal_ind = np.array([1])
#         elif modal == 't1':
#             self.modal_ind = np.array([2])
#         elif modal == 't2':
#             self.modal_ind = np.array([3])
#         elif modal == 'all':
#             self.modal_ind = np.array([0,1,2,3])

#     def __getitem__(self, index):

#         volpath = self.volpaths[index]
#         name = self.names[index]
#         x = np.load(volpath)
#         segpath = volpath.replace('vol', 'seg')
#         y = np.load(segpath).astype(np.uint8)
#         x, y = x[None, ...], y[None, ...]
#         x,y = self.transforms([x, y])

#         x = np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))# [Bsize,channels,Height,Width,Depth]
#         y = np.ascontiguousarray(y)
#         x = x[:, self.modal_ind, :, :, :]

#         x = torch.squeeze(torch.from_numpy(x), dim=0)
#         y = torch.squeeze(torch.from_numpy(y), dim=0)

#         mask = mask_array[index%15]
#         mask = torch.squeeze(torch.from_numpy(mask), dim=0)
#         return x, y, mask, name

#     def __len__(self):
#         return len(self.volpaths)
class Myops_loadall_val_nii(Dataset):
    def __init__(self, transforms='', root=None, modal='all', num_cls=6, train_file='val.txt'):
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
        if modal == 'bssfp':
            self.modal_ind = np.array([0])
        elif modal == 'lge':
            self.modal_ind = np.array([1])
        elif modal == 't2':
            self.modal_ind = np.array([2])
        elif modal == 'all':
            self.modal_ind = np.array([0,1,2])

    def __getitem__(self, index):

        volpath = self.volpaths[index]
        name = self.names[index]
        
        x = np.load(volpath)
        segpath = volpath.replace('vol', 'seg')
        y = np.load(segpath)
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

        mask_idx = np.random.choice(4, 1)
        mask = torch.squeeze(torch.from_numpy(mask_valid_array[mask_idx]), dim=0)
        return x, yo, mask, name

    def __len__(self):
        return len(self.volpaths)

class Brats2018(Dataset):

    def __init__(self, patients_dir, crop_size, modes, train=True, normalization = True):
        self.patients_dir = patients_dir
        self.modes = modes
        self.train = train
        self.crop_size = crop_size
        self.normalization = normalization

    def __len__(self):
        return len(self.patients_dir)

    def __getitem__(self, index):
        patient_dir = self.patients_dir[index]
        volumes = []
        modes = list(self.modes) + ['seg']
        for mode in modes:
            patient_id = os.path.split(patient_dir)[-1]
            volume_path = os.path.join(patient_dir, patient_id + "_" + mode + '.nii')
            volume = nib.load(volume_path).get_data()
            if not mode == "seg" and self.normalization:
                volume = self.normlize(volume)  # [0, 1.0]
            volumes.append(volume)                  # [h, w, d]
        seg_volume = volumes[-1]
        volumes = volumes[:-1]
        volume, seg_volume = self.aug_sample(volumes, seg_volume)
        ed_volume = (seg_volume == 2) # peritumoral edema ED
        net_volume = (seg_volume == 1) # enhancing tumor core NET
        et_volume = (seg_volume == 4) # enhancing tumor ET
        bg_volume = (seg_volume == 0)
        
        seg_volume = [ed_volume, net_volume, et_volume, bg_volume]
        seg_volume = np.concatenate(seg_volume, axis=0).astype("float32")

        return (torch.tensor(volume.copy(), dtype=torch.float),
                torch.tensor(seg_volume.copy(), dtype=torch.float))


    def aug_sample(self, volumes, mask):
        """
            Args:
                volumes: list of array, [h, w, d]
                mask: array [h, w, d], segmentation volume
            Ret: x, y: [channel, h, w, d]
        """
        x = np.stack(volumes, axis=0)       # [N, H, W, D]
        y = np.expand_dims(mask, axis=0)    # [channel, h, w, d]

        if self.train:
            # crop volume
            x, y = self.random_crop(x, y)
            if random.random() < 0.5:
                x = np.flip(x, axis=1)
                y = np.flip(y, axis=1)
            if random.random() < 0.5:
                x = np.flip(x, axis=2)
                y = np.flip(y, axis=2)
            if random.random() < 0.5:
                x = np.flip(x, axis=3)
                y = np.flip(y, axis=3)
        else:
            x, y = self.center_crop(x, y)

        return x, y

    def random_crop(self, x, y):
        """
        Args:
            x: 4d array, [channel, h, w, d]
        """
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = random.randint(0, height - crop_size[0] - 1)
        sy = random.randint(0, width - crop_size[1] - 1)
        sz = random.randint(0, depth - crop_size[2] - 1)
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def center_crop(self, x, y):
        crop_size = self.crop_size
        height, width, depth = x.shape[-3:]
        sx = (height - crop_size[0] - 1) // 2
        sy = (width - crop_size[1] - 1) // 2
        sz = (depth - crop_size[2] - 1) // 2
        crop_volume = x[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]
        crop_seg = y[:, sx:sx + crop_size[0], sy:sy + crop_size[1], sz:sz + crop_size[2]]

        return crop_volume, crop_seg

    def normlize(self, x):
        return (x - x.min()) / (x.max() - x.min())
    
    
    def normlize_brain(self, x, epsilon=1e-8):
        average        = x[np.nonzero(x)].mean()
        std            = x[np.nonzero(x)].std() + epsilon
        mask           = x>0
        sub_mean       = np.where(mask, x-average, x)
        x_normalized   = np.where(mask, sub_mean/std, x)
        return x_normalized

def split_dataset(data_root, test_p):
    patients_dir = glob.glob(os.path.join(data_root, "*GG", "Brats18*"))
    patients_dir.sort()
    N = int(len(patients_dir)*test_p)
    train_patients_list =  patients_dir[N:]
    val_patients_list   =  patients_dir[:N]

    return train_patients_list, val_patients_list
    
def make_data_loaders(config):
    train_list, val_list = split_dataset(config['path_to_data'], float(config['test_p']))
    crop_size = np.zeros((3))
    crop_size[0] = config['inputshape'][0]
    crop_size[1] = config['inputshape'][1]      
    crop_size[2] = config['inputshape'][2]
    crop_size    = crop_size.astype(np.uint16)
    crop_size    = (160, 192, 128)
    train_ds = Brats2018(train_list, crop_size=crop_size, modes=config['modalities'], train=True)
    val_ds = Brats2018(val_list,     crop_size=crop_size, modes=config['modalities'], train=False)
    loaders = {}
    loaders['train'] = DataLoader(train_ds, batch_size=int(config['batch_size_tr']),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=True)
    loaders['eval'] = DataLoader(val_ds, batch_size=int(config['batch_size_va']),
                                  num_workers=4,
                                  pin_memory=True,
                                  shuffle=False)
    return loaders