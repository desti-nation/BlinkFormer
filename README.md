# [BMVC-23] SynBlink and BlinkFormer
This is the official code and data for paper [SynBlink and BlinkFormer: A Synthetic Dataset and Transformer-Based Method for Video Blink Detection](http://phi-ai.buaa.edu.cn), accepted by BMVC 2023.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## Introduction

Accurate blink detection algorithms have significant implications in numerous fields, including human-computer interaction, driving safety, cognitive science, and medical diagnostics. Despite considerable efforts, the dataset volume for blink detection remains relatively small due to the cost of data collection and annotation, and there is still room for improvement in the accuracy of current algorithms.
In this paper, we introduce a workflow for synthesizing video data in Blender. Fully-rigged 3D human models are programmatically controlled, with variations in head movement, blinking, camera angles, background types, and lighting intensities. We used this workflow to create the [**SynBlink**](https://github.com/desti-nation/BlinkFormer/blob/main/README.md#synBlink-dataset) dataset, which includes 50,000 video clips and their corresponding annotations. Additionally, we present [**BlinkFormer**](https://github.com/desti-nation/BlinkFormer/blob/main/README.md#blinkformer), an innovative blink detection algorithm based on Transformer architecture that fully exploits temporal information from video clips.

## SynBlink Dataset

## BlinkFormer



## Getting Started

SN-Net is a general framework. However, as different model families are trained differently, we use their own code for stitching experiments. In this repo, we provide examples for several model families, such as plain ViTs, hierarchical ViTs, CNNs, CNN-ViT, and lightweight ViTs.

To use our repo, we suggest creating a Python virtual environment.

```bash
conda create -n snnet python=3.9
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install fvcore
pip install timm==0.6.12
```

Next, you can feel free to experiment with different settings.

For experiments with plain ViTs, please refer to [stitching_deit](./stitching_deit).

For experiments with hierarchical ViTs, please refer to [stitching_swin](./stitching_swin).

For experiments with CNNs and CNN-ViT, please refer to [stitching_resnet_swin](./stitching_resnet_swin).

For experiments with lightweight ViTs, please refer to [stitching_levit](./stitching_levit).


## Best Practice for Extension

Please feel free to extend SN-Net into other model familiy. The following tips may help your experiments.

### For Better Stitching

1. For paired stitching (equal depth) such as on plain ViTs, using a small sliding window for stitching usually achieves a smoother performance curve.
2. For unpaired stitching (unequal depth) such as on hierarchical ViTs, split the architecture into different stages and stitch within the same stage.
3. Note that many existing models allocate most blocks/layers into the 3rd stage, thus stitching at the 3rd stage can help to obtain more stitches. 
4. Remember to initialize your stitching layers. A few samples can be enough.


### For Better Training

1. Uniformly decreasing the learning rate (the training time LR) by 10x can serve as a good starting point. See our settings in DeiT-based experiments.
2. If the above is not good, try to decrease the learning rate for anchors while using a relatively larger learning rate for stitching layers. See our Swin-based experiments.
3. Training with more epochs (e.g., 100) can be better, but it also comes at a higher computational cost.


## Citation

If you use SynBlink or BlinkFormer in your research, please consider the following BibTeX entry and giving us a star

```BibTeX
@inproceedings{bo2023synblink,
  title={SynBlink and BlinkFormer: A Synthetic Dataset and Transformer-Based Method for Video Blink Detection},
  author={Bo Liu, Yang Xu, Feng Lu},
  booktitle={BMVC},
  year={2023}
}
```

## Acknowledgement

This implementation is built upon [VideoTransformer-pytorch](https://github.com/mx-mark/VideoTransformer-pytorch) and [vit-pytorch](https://github.com/lucidrains/vit-pytorch). We thank the authors for their released code.

## Contact

Welcome to raise issues or email to bliu03@buaa.edu.cn for any question.

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/ziplab/SN-Net/blob/main/LICENSE) file.
