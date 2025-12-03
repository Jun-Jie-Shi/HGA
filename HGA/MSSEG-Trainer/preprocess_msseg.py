import nibabel as nib
import glob
import os
import numpy as np
from tqdm import tqdm
import sys

import csv
import random

# DIM2PAD = [256, 256, 256]
DIM2PAD = [224, 224, 224]

def z_score_normalize(image, low_perc=1, high_perc=99):
    """Main pre-processing function used for the challenge (seems to work the best).
    Remove outliers voxels first, then min-max scale.
    """
    non_zeros = image > 0
    low, high = np.percentile(image[non_zeros], [low_perc, high_perc])
    image = np.clip(image, low, high)
    std = np.std(image[non_zeros])
    std = 1 if std == 0 else std
    mean = np.mean(image[non_zeros])
    image = (image - mean) / std
    # image = (image - low) / (high - low)
    return image.astype(np.float32)


def pad_background(image, dim2pad=(224, 224, 224)):
    """
    to invert the operation, use :
    inverted_image = np.zeros_like(image)
    inverted_image[crop_index] = padded_image[padded_index]
    """

    # use np.nonzero to find the indices of all non-zero elements in the image
    nz = np.nonzero(image)

    # get the minimum and maximum indices along each axis
    min_indices = np.min(nz, axis=1)
    max_indices = np.max(nz, axis=1)

    # crop the image to only include non-zero values
    crop_index = tuple(slice(imin, imax + 1) for imin, imax in zip(min_indices, max_indices))
    cropped_img = image[crop_index]
    padded_image = np.zeros(dim2pad)

    # crop further if any axis is larger than dim2pad
    crop_index_new = crop_index
    if cropped_img.shape[0] > dim2pad[0]:
        cx, cx_pad = cropped_img.shape[0] // 2, dim2pad[0] // 2
        cropped_img = cropped_img[cx - cx_pad : cx + cx_pad, :, :]
        crop_index_new = (
            slice(crop_index[0].start + cx - cx_pad, crop_index[0].start + cx + cx_pad),
            crop_index[1],
            crop_index[2],
        )
    if cropped_img.shape[1] > dim2pad[1]:
        cy, cy_pad = cropped_img.shape[1] // 2, dim2pad[1] // 2
        cropped_img = cropped_img[:, cy - cy_pad : cy + cy_pad, :]
        crop_index_new = (
            crop_index_new[0],
            slice(crop_index[1].start + cy - cy_pad, crop_index[1].start + cy + cy_pad),
            crop_index_new[2],
        )
    if cropped_img.shape[2] > dim2pad[2]:
        cz, cz_pad = cropped_img.shape[2] // 2, dim2pad[2] // 2
        cropped_img = cropped_img[:, :, cz - cz_pad : cz + cz_pad]
        crop_index_new = (
            crop_index_new[0],
            crop_index_new[1],
            slice(crop_index[2].start + cz - cz_pad, crop_index[2].start + cz + cz_pad),
        )

    # calculate the amount of padding needed along each axis
    pad_widths = [(padded_image.shape[i] - cropped_img.shape[i]) // 2 for i in range(3)]

    # pad the image with zeros
    padded_index = tuple(slice(pad_widths[i], pad_widths[i] + cropped_img.shape[i]) for i in range(3))
    padded_image[padded_index] = cropped_img

    return padded_image, crop_index_new, padded_index

def pad_background_with_index(image, crop_index_new, padded_index, dim2pad=(224, 224, 224)):
    padded_image = np.zeros(dim2pad)
    crop_image = image[crop_index_new]
    padded_image[padded_index] = crop_image
    return padded_image

def extract2d_data2npz(list_file_flair, distinct_subject=False, save_path="/home/sjj/MMMSeg/MSSEG/MSSEG2016_Training_none_npy/"):
    os.makedirs(save_path, exist_ok=True)
    txt_path = os.path.join(save_path, 'test.txt')

    useless = 0
    useful = 0
    num_zero_list, num_non_zero_list = [], []
    for count_subject, path_flair in tqdm(enumerate(list_file_flair, 1)):
        path_parts = path_flair.strip('/').split('/')
        flair = nib.load(path_flair).get_fdata()
        t1 = nib.load(path_flair.replace("FLAIR", "T1")).get_fdata()
        t2 = nib.load(path_flair.replace("FLAIR", "T2")).get_fdata()
        dp = nib.load(path_flair.replace("FLAIR", "DP")).get_fdata()
        gado = nib.load(path_flair.replace("FLAIR", "GADO")).get_fdata()
        consensus = nib.load(path_flair.replace("Preprocessed_Data", "Masks").replace("FLAIR_preprocessed", "Consensus"))
        consensus = consensus.get_fdata().astype(np.uint32)

        padded_flair, crop_index, padded_index = pad_background(flair, dim2pad=DIM2PAD)
        padded_t1 = pad_background_with_index(t1, crop_index, padded_index, dim2pad=DIM2PAD)
        padded_dp = pad_background_with_index(dp, crop_index, padded_index, dim2pad=DIM2PAD)
        padded_t2 = pad_background_with_index(t2, crop_index, padded_index, dim2pad=DIM2PAD)
        padded_gado = pad_background_with_index(gado, crop_index, padded_index, dim2pad=DIM2PAD)
        padded_consensus = pad_background_with_index(consensus, crop_index, padded_index, dim2pad=DIM2PAD)

        flair = z_score_normalize(flair)
        t1 = z_score_normalize(t1)
        t2 = z_score_normalize(t2)
        dp = z_score_normalize(dp)
        gado = z_score_normalize(gado)

        padded_masks = [padded_consensus]
        for padded_mask in padded_masks:
            _, (num_zero, num_non_zero) = np.unique(padded_mask, return_counts=True)
            num_zero_list.append(num_zero)
            num_non_zero_list.append(num_non_zero)

        for i in range(padded_flair.shape[-1]):
            slices_t1 = padded_t1[..., i]  # shape (224, 224, 1)
            slices_flair = padded_flair[..., i]  # shape (224, 224, 1)
            slices_t2 = padded_t2[..., i]  # shape (224, 224, 1)
            slices_dp = padded_dp[..., i]  # shape (224, 224, 1)
            slices_gado = padded_gado[..., i]  # shape (224, 224, 1)
            slice_inputs = np.stack(
                [slices_flair, slices_t1, slices_t2, slices_dp, slices_gado],
                axis=-1,
            )
            slices_mask = padded_consensus[..., i]  # shape (224, 224)
            if np.count_nonzero(slices_mask) >= 150:
                useful += 1
                name_subject = f"{path_parts[-4]}_{path_parts[-3]}_slice_{i}"
                np.save(os.path.join(save_path, 'vol', name_subject + '_vol.npy'), slice_inputs.astype(np.float32))
                np.save(os.path.join(save_path, 'seg', name_subject + '_seg.npy'), slices_mask)
                print(f"{path_parts[-4]}_{path_parts[-3]}_slice_{i} is useful_{useful}")
                with open(txt_path,"a") as f:
                    i_ = name_subject + "\n"
                    f.write(i_)
            else:
                useless += 1
                print(f"{path_parts[-4]}_{path_parts[-3]}_slice_{i} is useless")

    print("useless: ", useless)



if __name__ == "__main__":
    list_file_flair = sorted(glob.glob(f"/home/sjj/MMMSeg/MSSEG/Training/Center*/*/Preprocessed_Data/*FLAIR*"))
    print(len(list_file_flair))

    extract2d_data2npz(
        list_file_flair,
        distinct_subject=True,
        save_path="/home/sjj/MMMSeg/MSSEG/MSSEG2016_Training_none_npy/",
    )
    # new_image_name_list = []
    # for new_image_name in os.listdir("/home/sjj/MMMSeg/MSSEG/MSSEG2016_Training_none_npy/vol/"):
    #     new_image_name_list.append(new_image_name.rsplit('_vol.npy', 1)[0])
    # new_image_name_list.sort(key=None, reverse=False)
    # with open("/home/sjj/MMMSeg/MSSEG/MSSEG2016_Training_none_npy/train.txt","w") as f:
    #     for i in new_image_name_list:
    #         i_ = i + "\n"
    #         f.write(i_)
