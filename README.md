# [BMVC-23] SynBlink and BlinkFormer
This is the official code and data for paper [SynBlink and BlinkFormer: A Synthetic Dataset and Transformer-Based Method for Video Blink Detection](http://phi-ai.buaa.edu.cn), accepted by BMVC 2023.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>

## Introduction

Accurate blink detection algorithms have significant implications in numerous fields, including human-computer interaction, driving safety, cognitive science, and medical diagnostics. Despite considerable efforts, the dataset volume for blink detection remains relatively small due to the cost of data collection and annotation, and there is still room for improvement in the accuracy of current algorithms.
In this paper, we introduce a workflow for synthesizing video data in Blender. Fully-rigged 3D human models are programmatically controlled, with variations in head movement, blinking, camera angles, background types, and lighting intensities. We used this workflow to create the [**SynBlink**](https://github.com/desti-nation/BlinkFormer/blob/main/README.md#synBlink-dataset) dataset, which includes 50,000 video clips and their corresponding annotations. Additionally, we present [**BlinkFormer**](https://github.com/desti-nation/BlinkFormer/blob/main/README.md#blinkformer), an innovative blink detection algorithm based on Transformer architecture that fully exploits temporal information from video clips.

## SynBlink Dataset

download link: 

## BlinkFormer

### Requirements

### Test on HUST-LEBW dataset

### Train


## Citation

If you use SynBlink or BlinkFormer in your research, please consider the following BibTeX entry and giving us a star

```BibTeX
@inproceedings{bo2023synblink,
  title={SynBlink and BlinkFormer: A Synthetic Dataset and Transformer-Based Method for Video Blink Detection},
  author={Bo Liu, Yang Xu, Feng Lu},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2023}
}
```

## Acknowledgement

This implementation is built upon [VideoTransformer-pytorch](https://github.com/mx-mark/VideoTransformer-pytorch) and [vit-pytorch](https://github.com/lucidrains/vit-pytorch). We thank the authors for their released code.

## Contact

Welcome to raise issues or email to bliu03@buaa.edu.cn for any question.

## License

This repository is released under the Apache 2.0 license as found in the [LICENSE](https://github.com/ziplab/SN-Net/blob/main/LICENSE) file.
