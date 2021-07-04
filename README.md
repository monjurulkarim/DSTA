# DSTA
This is the implementation code for the paper, <a href="https://arxiv.org/abs/2106.10197"> "A Dynamic Spatial-temporal Attention Network for Early Anticipation of Traffic Accidentss"</a>.</p>




<a name="dataset"></a>
## Dataset Preparation

The code currently supports three datasets., DAD and CCD. These datasets need to be prepared under the folder `data/`. 

> * For CCD dataset, please refer to the [CarCrashDataset Official](https://github.com/Cogito2012/CarCrashDataset) repo for downloading and deployment. 
> * For DAD dataset, you can acquire it from [DAD official](https://github.com/smallcorgi/Anticipating-Accidents). The officially provided features are grouped into batches while it is more standard to split them into separate files for training and testing. To this end, you can use the script `./script/split_dad.py`.

<a name="install"></a>
## Installation Guide

### 1. Setup Python Environment

The code is implemented with `Python=3.7.4` and `PyTorch=1.0.0` with `CUDA=10.0.130` and `cuDNN=7.6.3`. We highly recommend using Anaconda to create virtual environment to run this code. Please follow the following installation dependencies strictly:
```shell
# create python environment
conda create -n py37 python=3.7

# activate environment
conda activate py37

# install dependencies
pip install -r requirements.txt
```

### 2. Setup MMDetection Environment (Optional)

If you need to use mmdetection for training and testing Cascade R-CNN models, you may need to setup an mmdetection environment separately such as `mmlab`. Please follow the [official mmdetection installation guide](https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md).
```shell
# create python environment
conda create -n mmlab python=3.7

# activate environment
conda activate mmlab

# install dependencies
pip install torch==1.4.0+cu100 torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv==0.4.2

# Follow the instructions at https://github.com/open-mmlab/mmdetection/blob/master/docs/install.md
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v1.1.0  # important!
cp -r ../Cascade\ R-CNN/* ./  # copy the downloaded files into mmdetection folder

# compile & install
pip install -v -e .
python setup.py install

# Then you are all set!
``` 
### 3. Train UString from scratch.

To train UString model from scratch, run the following commands for DAD dataset:
```shell
# For dad dataset, use GPU_ID=0 and batch_size=10.
bash run_train_test.sh train 0 dad 10
```
By default, the snapshot of each checkpoint file will be saved in `output/DSTA/vgg16/snapshot/`.
