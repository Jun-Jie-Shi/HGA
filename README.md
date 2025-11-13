<div align="center">
<h3>Addressing Imbalanced Modal Incompleteness in Realistic Medical Image Segmentation via Hierarchical Gradient Alignment</h3>

[Junjie Shi](https://github.com/Jun-Jie-Shi)<sup>1</sup> ,[Zhaobin Sun](https://github.com/szbonaldo)<sup>1</sup> ,[Li Yu](https://eic.hust.edu.cn/professor/yuli/)<sup>1</sup> ,[Xin Yang](https://sites.google.com/view/xinyang/home)<sup>1</sup> ,[Zengqiang Yan](https://mia2i.github.io/home/)<sup>1 :email:</sup>

🏢 <sup>1</sup> Huazhong University of Science and Technology,  (<sup>:email:</sup>) corresponding author.
</div>

For our paper PASSION accepted by ACM MM-2024 Oral, please refer to  [OpenReview](https://openreview.net/forum?id=jttrL7wHLC) or [ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3664647.3681543) and this extension version HGA is used for submission.
The extension version HGA based on PASSION:
1. provide both 2D and 3D training code for MyoPS, MSSEG and BraTS datasets;
2. rethink the uni-modal and multi-modal combination wise co-learning problem; 
3. combine conflict-free meta learning with our preference-aware self-distillation;
4. compare against various Multi-Task Learning methods deployed in IDT setting;
5. work well with various backbones as well as modal-balancing methods;

## 📋️Requirements
We recommend using conda to setup the environment. See the `requirements.txt` for environment configuration.

All our experiments are implemented based on the PyTorch framework with one 24G NVIDIA Geforce RTX 3090 GPU, and we recommend installing the following package versions:
- python=3.8
- pytorch=1.12.1
- torchvision=0.13.1

Dependency packages can be installed using following command:

```bash
git clone https://github.com/Jun-Jie-Shi/HGA.git
cd HGA
conda create --name hga python=3.8
conda activate hga
pip install -r requirements.txt
```

## 📊Datasets Preparation
### Directly download preprocessed dataset
You can download the preprocessed dataset (e.g. BraTS2020) from [RFNet](https://drive.google.com/drive/folders/1AwLwGgEBQwesIDTlWpubbwqxxd8brt5A?usp=sharing) and unzip them in the `datasets/BraTS` folder.
```bash
  tar -xzf BRATS2020_Training_none_npy.tar.gz
```
The data-split is available in the `datasets/BraTS/BRATS2020_Training_none_npy` folder, and our imbalanced missing rates data-split is available in the `datasets/BraTS/brats_split` folder.

### How to preprocess by yourself
If you want to preprocess by yourself, the preprocessing code `preprocess_brats.py` is also provided, just download `BRATS2020_Training_Data` in `datasets/BraTS` folder. 

`Notes:` Here our default path `BRATS2020_Training_Data` may refer to `BRATS2020_Training_Data/MICCAI_BRATS2020_Training_Data` if you download in [kaggle](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation).

And if you want to divide data by yourself, `data_split.py` and `generate_imb_mr.py` in the `code/preprocessing` folder is available. (Here we only provide the preprocessing for BraTS, if you want to use other datasets, just do it similarly)

If your folder structure (especially for datasets path) is as follows:
```
HGA/
├── datasets
│   ├── BraTS
│   │   ├── BRATS2020_Training_Data
│   │   │   ├── BraTS20_Training_001
│   │   │   │   ├── BraTS20_Training_001_flair.nii.gz
│   │   │   │   ├── BraTS20_Training_001_seg.nii.gz
│   │   │   │   ├── BraTS20_Training_001_t1.nii.gz
│   │   │   │   ├── BraTS20_Training_001_t1ce.nii.gz
│   │   │   │   ├── BraTS20_Training_001_t2.nii.gz
│   │   │   ├── BraTS20_Training_002
│   │   │   │   ├── ...
│   │   │   ├── ...
│   │   │   ├── BraTS20_Training_369
│   │   │   │   ├── ...
├── BraTS-Trainer
│   ├── ...
└── ...
```
you can simply conduct the preprocessing as following:
``` python
python code/preprocessing/preprocess_brats.py
python code/preprocessing/data_split.py
python code/preprocessing/generate_imb_mr.py
```
After preprocessing, your folder structure is assumed to be:
```
HGA/
├── datasets
│   ├── BraTS
│   │   ├── BRATS2020_Training_Data
│   │   │   ├── ...
│   │   ├── BRATS2020_Training_none_npy
│   │   │   ├── seg
│   │   │   ├── vol
│   │   │   ├── test.txt
│   │   │   ├── train.txt
│   │   │   ├── val.txt
│   │   ├── brats_split
│   │   │   ├── Brats2020_imb_split_mr2468.csv
├── BraTS-Trainer
│   ├── ...
└── ...
```

## 🔧Options Setting
Before start training, you should check the options in `BraTS-Trainer/options.py`,  especially for datasets path. Our code-notes may help you.

Our default relative datapath is according to our folder structure, if your datapath is different, just change `datarootPath` and `datasetPath` as your absolute data-saving root-path and dataset-saving path. 

Other path setting like `imbmrpath` and `savepath` is also noteworthy.

## 🚀Running
You can conduct the experiment as following if everything is ready.
```
cd ./BraTS-Trainer
python train.py
```

For evaluation, the `eval.py` is simply implemented, just change the corresponding checkpoint path `resume` and other path settings.


