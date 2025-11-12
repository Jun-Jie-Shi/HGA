import os

import medpy.io as medio
import numpy as np

src_path = '/home/sjj/MMMSeg/MyoPS/MyoPS2020_Training_Data'
tar_path = '/home/sjj/MMMSeg/MyoPS/MyoPS2020_Training_none_npy'

name_list = os.listdir(src_path)


def normalize(vol):
    mask = vol.sum(0) > 0
    for k in range(3):
        x = vol[k, ...]
        y = x[mask]
        x = (x - y.mean()) / y.std()
        vol[k, ...] = x

    return vol

if not os.path.exists(os.path.join(tar_path, 'vol')):
    os.makedirs(os.path.join(tar_path, 'vol'))

if not os.path.exists(os.path.join(tar_path, 'seg')):
    os.makedirs(os.path.join(tar_path, 'seg'))

for file_name in name_list:
    print (file_name)
    # num = file_name.split('_')[2]
    # HLG = 'HG_' if int(num) <= 259 or int(num) >= 336 else 'LG_'
    c0, c0_header = medio.load(os.path.join(src_path, file_name, file_name+'_C0.nii.gz'))
    de, de_header = medio.load(os.path.join(src_path, file_name, file_name+'_DE.nii.gz'))
    # t1, t1_header = medio.load(os.path.join(src_path, file_name, file_name+'_t1.nii.gz'))
    t2, t2_header = medio.load(os.path.join(src_path, file_name, file_name+'_T2.nii.gz'))

    vol = np.stack((c0, de, t2), axis=0).astype(np.float32)
    # x_min, x_max, y_min, y_max, z_min, z_max = crop(vol)
    vol1 = normalize(vol)
    vol1 = vol1.transpose(1,2,3,0)
    print (vol1.shape)

    seg, seg_header = medio.load(os.path.join(src_path, file_name, file_name+'_gd.nii.gz'))
    seg = seg.astype(np.uint32)
    seg1 = seg
    seg1[seg1==200]=1
    seg1[seg1==500]=2
    seg1[seg1==600]=3
    seg1[seg1==1220]=4
    seg1[seg1==2221]=5
    # print(vol1.shape)
    # print(seg1.shape)
    # print(np.unique(seg1))

    for i in range(vol1.shape[2]):
        np.save(os.path.join(tar_path, 'vol', file_name+'_slice_{}'.format(i)+'_vol.npy'), vol1[:,:,i:i+1,:])
        np.save(os.path.join(tar_path, 'seg', file_name+'_slice_{}'.format(i)+'_seg.npy'), seg1[:,:,i:i+1])
