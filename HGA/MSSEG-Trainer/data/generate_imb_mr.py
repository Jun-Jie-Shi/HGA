import csv
import os
import numpy as np
import torch
import random
import math

# currentdirPath = os.path.dirname(os.path.abspath(__file__))
# relativePath = '../../datasets'
datarootPath = '/home/sjj/MMMSeg'
## Note: or directly set datarootPath as your data-saving path (absolute root)
train_path = os.path.join(datarootPath, 'MSSEG/MSSEG2016_Training_none_npy')
train_file = os.path.join(train_path, 'train.txt')
split_path = os.path.join(datarootPath, 'MSSEG/msseg_split')
os.makedirs(split_path, exist_ok=True)
csv_name = os.path.join(split_path, 'MSSEG2016_imb_split_mr97531.csv')
txt_path = os.path.join(train_path, 'val.txt')

p=[0.9, 0.7, 0.5, 0.3, 0.1]

## set_seed
seed = 1048
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

with open(train_file, 'r') as f:
    datalist = [i.strip() for i in f.readlines()]
datalist.sort()

img_max = len(datalist)
img_num_per_cls = [0 for i in range(31)]

def imb_mr_split(p, img_max):

    flair = np.random.rand(img_max)>p[0]
    t1 = np.random.rand(img_max)>p[1]
    t2 = np.random.rand(img_max)>p[2]
    dp = np.random.rand(img_max)>p[3]
    gado = np.random.rand(img_max)>p[4]


    return flair, t1, t2, dp, gado


flair, t1, t2, dp, gado = imb_mr_split(p, img_max)

index = 0
pos_index = []

## Counting and Saving Statistics of Imbalanced-MR BraTS Data

masks = [[False, False, False, False, True], [False, False, False, True, False], [False, False, True, False, False], [False, True, False, False, False], [True, False, False, False, False],
        [False, False, False, True, True], [False, False, True, False, True], [False, True, False, False, True], [True, False, False, False, True], [False, False, True, True, False], [False, True, False, True, False], [True, False, False, True, False], [False, True, True, False, False], [True, False, True, False, False], [True, True, False, False, False],
        [False, False, True, True, True], [False, True, False, True, True], [True, False, False, True, True], [False, True, True, False, True], [True, False, True, False, True], [True, True, False, False, True], [False, True, True, True, False], [True, False, True, True, False], [True, True, False, True, False], [True, True, True, False, False],
        [False, True, True, True, True], [True, False, True, True, True], [True, True, False, True, True], [True, True, True, False, True], [True, True, True, True, False],
        [True, True, True, True, True]]

file = open(csv_name, "a+")
csv_writer = csv.writer(file)
csv_writer.writerow(['data_name', 'mask_id', 'mask', 'pos_mask_ids']) ## possible mask id (after-moddrop)
for i in range(img_max):
    if (not flair[i] and not t1[i] and not t2[i] and not dp[i] and not gado[i]):
        with open(txt_path,"a") as f:
            i_ = datalist[i] + "\n"
            f.write(i_)
        continue
    if [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[0]:
        img_num_per_cls[0] +=1
        index = 0
        pos_index = [0]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[1]:
        img_num_per_cls[1] +=1
        index = 1
        pos_index = [1]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[2]:
        img_num_per_cls[2] +=1
        index = 2
        pos_index = [2]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[3]:
        img_num_per_cls[3] +=1
        index = 3
        pos_index = [3]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[4]:
        img_num_per_cls[4] +=1
        index = 4
        pos_index = [4]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[5]:
        img_num_per_cls[5] +=1
        index = 5
        pos_index = [0,1]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[6]:
        img_num_per_cls[6] +=1
        index = 6
        pos_index = [0,2]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[7]:
        img_num_per_cls[7] +=1
        index = 7
        pos_index = [0,3]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[8]:
        img_num_per_cls[8] +=1
        index = 8
        pos_index = [0,4]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[9]:
        img_num_per_cls[9] +=1
        index = 9
        pos_index = [1,2]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[10]:
        img_num_per_cls[10] +=1
        index = 10
        pos_index = [1,3]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[11]:
        img_num_per_cls[11] +=1
        index = 11
        pos_index = [1,4]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[12]:
        img_num_per_cls[12] +=1
        index = 12
        pos_index = [2,3]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[13]:
        img_num_per_cls[13] +=1
        index = 13
        pos_index = [2,4]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[14]:
        img_num_per_cls[14] +=1
        index = 14
        pos_index = [3,4]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[15]:
        img_num_per_cls[15] +=1
        index = 15
        pos_index = [0,1,2,5,6,9,15]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[16]:
        img_num_per_cls[16] +=1
        index = 16
        pos_index = [0,1,3,5,7,10,16]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[17]:
        img_num_per_cls[17] +=1
        index = 17
        pos_index = [0,1,4,5,8,11,17]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[18]:
        img_num_per_cls[18] +=1
        index = 18
        pos_index = [0,2,3,6,7,12,18]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[19]:
        img_num_per_cls[19] +=1
        index = 19
        pos_index = [0,2,4,6,8,13,19]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[20]:
        img_num_per_cls[20] +=1
        index = 20
        pos_index = [0,3,4,7,8,14,20]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[21]:
        img_num_per_cls[21] +=1
        index = 21
        pos_index = [1,2,3,9,10,12,21]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[22]:
        img_num_per_cls[22] +=1
        index = 22
        pos_index = [1,2,4,9,11,13,22]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[23]:
        img_num_per_cls[23] +=1
        index = 23
        pos_index = [1,3,4,10,11,14,23]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[24]:
        img_num_per_cls[24] +=1
        index = 24
        pos_index = [2,3,4,12,13,14,24]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[25]:
        img_num_per_cls[25] +=1
        index = 25
        pos_index = [0,1,2,3,5,6,7,9,10,12,15,16,18,21,25]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[26]:
        img_num_per_cls[26] +=1
        index = 26
        pos_index = [0,1,2,4,5,6,8,9,11,13,15,17,19,22,26]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[27]:
        img_num_per_cls[27] +=1
        index = 27
        pos_index = [0,1,3,4,5,7,8,10,11,14,16,17,20,23,27]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[28]:
        img_num_per_cls[28] +=1
        index = 28
        pos_index = [0,2,3,4,6,7,8,12,13,14,18,19,20,24,28]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[29]:
        img_num_per_cls[29] +=1
        index = 29
        pos_index = [1,2,3,4,9,10,11,12,13,14,21,22,23,24,29]
    elif [flair[i],t1[i],t2[i],dp[i],gado[i]] == masks[30]:
        img_num_per_cls[30] +=1
        index = 30
        pos_index = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
    
    csv_writer = csv.writer(file)
    csv_writer.writerow([datalist[i],index,[flair[i],t1[i],t2[i],dp[i],gado[i]],pos_index])
file.close()
print(img_num_per_cls)
print(sum(img_num_per_cls))

img_num_per_cls = np.array(img_num_per_cls)

flair_index = [4,8,11,13,14,17,19,20,22,23,24,26,27,28,29,30]
t1_index = [3,7,10,12,14,16,18,20,21,23,24,25,27,28,29,30]
t2_index = [2,6,9,12,13,15,18,19,21,22,24,25,26,28,29,30]
dp_index = [1,5,9,10,11,15,16,17,21,22,23,25,26,27,29,30]
gado_index = [0,5,6,7,8,15,16,17,18,19,20,25,26,27,28,30]

flair = img_num_per_cls[flair_index]
t1 = img_num_per_cls[t1_index]
t2 = img_num_per_cls[t2_index]
dp = img_num_per_cls[dp_index]
gado = img_num_per_cls[gado_index]
print([sum(flair),sum(t1),sum(t2),sum(dp),sum(gado)])

# [154, 37, 22, 7, 1, 331, 157, 66, 20, 32, 16, 14, 8, 2, 1, 363, 140, 46, 79, 16, 8, 16, 5, 1, 1, 170, 47, 15, 6, 3, 18]
# 1802
# [204, 555, 945, 1254, 1636]


