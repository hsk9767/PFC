Pose Feature Correction for resnet based code

### Note:
- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.

## Environment
The code is developed using python 3.6 on Ubuntu 16.04. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA 3090 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v0.4.0 following [official instruction](https://pytorch.org/).
  if you use 3090 GPU, then we highly recommend to install pytorch with "pip install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu111/torch_nightly.html"

2. Install dependencies:
   ```
   pip install -r requirements.txt

   ```
3. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
   When an error ("Value 'sm_xx' is not defined for option 'gpu-architecture'") happens, check compatible versions by typing 'lib/nms/setup.py' and then modify the '-arch=sm_xx' in line 127 of 'lib/nms/setup.py'.
3. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user

4. Download coco pretrained models from "https://drive.google.com/drive/folders/1Hc-TzrPEieLPTGdCejwW2ldbKrVtj8hY?usp=sharing"

   ```
   ${POSE_ROOT}
    `-- pretrained_weights
            |-- res152_256x192.pth.tar
            |-- res152_384x288.pth.tar
            |-- res50_256x192.pth.tar
            |-- res50_384x288.pth.tar
            |-- hr32_256x192.pth.tar
            |-- hr32_384x288.pth.tar
            |-- hr48_256x192.pth.tar
            '-- hr48_384x288.pth.tar

   ```
   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── pretrained_weights
   ├── output
   ├── tools
   ├── visualization
   ├── README.md
   └── requirements.txt
   ```
   
### Data preparation

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```
### Valid on COCO val2017 using pretrained models

```
python pose_estimation/valid.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml --model-file pretrained_weights/res50_256x192.pth.tar
```

### Training on COCO train2017

```
python pose_estimation/train.py \
    --cfg experiments/coco/resnet50/256x192_d256x3_adam_lr1e-3.yaml
```


Our code was referenced below:
```
@inproceedings{xiao2018simple,
    author={Xiao, Bin and Wu, Haiping and Wei, Yichen},
    title={Simple Baselines for Human Pose Estimation and Tracking},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2018}
}

@article{xie2021segmenting,
  title={Segmenting transparent object in the wild with transformer},
  author={Xie, Enze and Wang, Wenjia and Wang, Wenhai and Sun, Peize and Xu, Hang and Liang, Ding and Luo, Ping},
  journal={arXiv preprint arXiv:2101.08461},
  year={2021}
}
```
