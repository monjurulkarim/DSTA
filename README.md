[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-dynamic-spatial-temporal-attention-network/accident-anticipation-on-ccd)](https://paperswithcode.com/sota/accident-anticipation-on-ccd?p=a-dynamic-spatial-temporal-attention-network)
# DSTA
This is the implementation code for the paper, <a href="https://arxiv.org/abs/2106.10197"> "A Dynamic Spatial-temporal Attention Network for Early Anticipation of Traffic Accidentss"</a>. <i> IEEE Transaction on Intelligent Transportation Systems,</i> 2022.</p>
<!-- <div align=center> -->
<!-- <img src="demo/visualization_000142.gif" alt="Visualization Demo" width="800"/>  -->
<!-- </div>   -->



The aim of this project is to predict an accident as early as possible using dashcam video data.

<a name="dataset"></a>
## Dataset Preparation

The code currently supports two datasets, DAD and CCD. These datasets need to be prepared under the folder `data/`. 

> * For CCD dataset, please refer to the [CarCrashDataset Official](https://github.com/Cogito2012/CarCrashDataset) repo for downloading and deployment. 
> * For DAD dataset, you can acquire it from [DAD official](https://github.com/smallcorgi/Anticipating-Accidents). The officially provided features are grouped into batches while it is more standard to split them into separate files for training and testing. To this end, you can use the script `./script/split_dad.py`.

<a name="install"></a>
## Installation Guide

### 1. Setup Python Environment

The code is implemented and tested with `Python=3.7.9` and `PyTorch=1.2.0` with `CUDA=10.2`. We highly recommend using Anaconda to create virtual environment to run this code. Please follow the following installation dependencies strictly:
```shell
# create python environment
conda create -n py37 python=3.7

# activate environment
conda activate py37

# install dependencies
pip install -r requirements.txt
```

### 1.1.(Optional) Setup MMDetection Environment

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


### 2.  Pre-trained Models
> * [**Cascade R-CNN**](https://drive.google.com/drive/folders/1fbjKrzgXv_FobuIAS37k9beCkxYzVavi?usp=sharing): This is the pre-trained object detector trained by UString. Please, directly download the pre-trained Cascade R-CNN model files and modified source files from their website. Please download and extract them under `lib/mmdetection/`.
> * [**Pre-trained DSTA Models**](https://drive.google.com/drive/folders/1QpfxS58XqBUex6zwZG1DWD0rd0Nb-aAu?usp=sharing): The pretrained model weights for testing and demo usages. If you want to use the pre-trained model for demo, then please put it inside the directory `demo/`.

### 3. Demo

The following script will generate a video with an accident prediction curve. Note that before you run the following script, both the python and mmdetection environments above are needed. The following command is an example using the pretrained model on CCD dataset. 

```shell
bash run_demo.sh demo/000007.mp4
```
Results will be saved in the same folder `demo/`.


### 4. Train DSTA from scratch.

To train DSTA model from scratch, run the following commands for DAD dataset:
```shell
# For dad dataset, use GPU_ID=0 and batch_size=10.
bash run_train_test.sh train 0 dad 10
```
By default, the snapshot of each checkpoint file will be saved in `output/DSTA/vgg16/snapshot/`.


### 5. Test the trained DSTA model

Take the DAD dataset as an example, after training with the DAD dataset and configuring the dataset correctly, run the following command. By default the model file will be placed at `output/DSTA/vgg16/snapshot/final_model.pth`.
```shell
# For dad dataset, use GPU_ID=0 and batch_size=10.
bash run_train_test.sh test 0 dad 10
```
The evaluation results on test set will be reported, and visualization results will be saved in `output/DSTA/vgg16/test/`.

<a name="citation"></a>
## Citation

Please cite our paper if you find the code useful.

```
@article{karim2022dynamic,
  title={A dynamic Spatial-temporal attention network for early anticipation of traffic accidents},
  author={Karim, Muhammad Monjurul and Li, Yu and Qin, Ruwen and Yin, Zhaozheng},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={23},
  number={7},
  pages={9590--9600},
  year={2022},
  publisher={IEEE}
}
```

Parts of the code are adopted from [UString](https://github.com/Cogito2012/UString) project. Many thanks to the contributors of that repository.
