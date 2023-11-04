# **vocal-remover**

<p align="center"><img src="https://img.shields.io/github/commit-activity/t/filispeen/vocal-remover-cli?style=flat-square" alt="Latest release">
<img src="https://img.shields.io/github/forks/filispeen/vocal-remover-cli?style=flat-square" alt="GitHub forks">
<img src="https://img.shields.io/github/contributors/filispeen/vocal-remover-cli?style=flat-square" alt="GitHub contributors">
<img src="https://img.shields.io/github/issues/filispeen/vocal-remover-cli?style=flat-square" alt="GitHub issues">
<img src="https://img.shields.io/github/downloads/filispeen/vocal-remover-cli/total?style=flat-square" alt="GitHub all releases"></p>

This is a deep-learning-based tool to extract instrumental track from your songs.

## Installation

### Official vocal-remover
Download the original version from [here](https://github.com/tsurumeso/vocal-remover).

### Install PyTorch
**See**: [GET STARTED](https://pytorch.org/get-started/locally/)

### Install package
```
pip install git+https://github.com/filispeen/vocal-remover-cli@UVR_v1
```

## Usage
The following command separates the input into instrumental and vocal tracks. They are saved as `*_Instruments.wav` and `*_Vocals.wav`.

### Run on CPU
```
vc-remover infer --input path/to/an/audio/file
```

### Run on GPU
```
vc-remover infer --input path/to/an/audio/file --gpu 0
```

### Advanced options
`--tta` option performs Test-Time-Augmentation to improve the separation quality.
```
vc-remover infer --input path/to/an/audio/file --tta --gpu 0
```

`--postprocess` option masks instrumental part based on the vocals volume to improve the separation quality.  
**Experimental Warning**: If you get any problems with this option, please disable it.
```
vc-remover infer --input path/to/an/audio/file --postprocess --gpu 0
```

## Train your own model

### Place your dataset
```
path/to/dataset/
  +- instruments/
  |    +- 01_foo_inst.wav
  |    +- 02_bar_inst.mp3
  |    +- ...
  +- mixtures/
       +- 01_foo_mix.wav
       +- 02_bar_mix.mp3
       +- ...
```

### Train a model
```
vc-remover train --dataset path/to/dataset --reduction_rate 0.5 --mixup_rate 0.5 --gpu 0
```

## References
- [1] Jansson et al., "Singing Voice Separation with Deep U-Net Convolutional Networks", https://ejhumphrey.com/assets/pdf/jansson2017singing.pdf
- [2] Takahashi et al., "Multi-scale Multi-band DenseNets for Audio Source Separation", https://arxiv.org/pdf/1706.09588.pdf
- [3] Takahashi et al., "MMDENSELSTM: AN EFFICIENT COMBINATION OF CONVOLUTIONAL AND RECURRENT NEURAL NETWORKS FOR AUDIO SOURCE SEPARATION", https://arxiv.org/pdf/1805.02410.pdf
- [4] Liutkus et al., "The 2016 Signal Separation Evaluation Campaign", Latent Variable Analysis and Signal Separation - 12th International Conference
