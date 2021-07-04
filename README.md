# DSTA
This is the implementation code for the paper, <a href="https://arxiv.org/abs/2106.10197"> "A Dynamic Spatial-temporal Attention Network for Early Anticipation of Traffic Accidentss"</a>.</p> "




<a name="dataset"></a>
## :file_cabinet:  Dataset Preparation

The code currently supports three datasets., DAD, A3D, and CCD. These datasets need to be prepared under the folder `data/`. 

> * For CCD dataset, please refer to the [CarCrashDataset](https://github.com/Cogito2012/CarCrashDataset) repo for downloading and deployment. 
> * For DAD dataset, you can acquire it from [DAD official](https://github.com/smallcorgi/Anticipating-Accidents). The officially provided features are grouped into batches while it is more standard to split them into separate files for training and testing. To this end, you can use the script `./script/split_dad.py`.
