# MDANet
```
This repository provides a PyTorch implementation for underwater image enhancement based on multi-scale and dual attention mechanisms.

The current code supports training on NVIDIA GPU. CPU can also be used, but the training speed may be slow.

```
## Requirements
```
pip install -r requirements.txt
```

## Train the model
```
$ python train.py TRAIN_RAW_IMAGE_FOLDER TRAIN_REFERENCE_IMAGE_FOLDER
```

## Test the model
```
$ python test.py --checkpoint CHECKPOINTS_PATH
```

## Contact

If you have any questions, please contact at 13783428056@163.com.
