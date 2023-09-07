# Data Preparation

- HUST-LEBW
  1. Download HUST-LEBW dataset from [link](https://thorhu.github.io/Eyeblink-in-the-wild/)
  2. Change the root_path of HUSTDataset to your data_path
  3. Put json files in data_split/HUST-LEBW to your data_path

- SynBlink
  
    1. Download SybBlink dataset from [link](https://pan.baidu.com/s/1bJ0nj0SxfVCxRmICKz5p8A?pwd=synb)
    2. Crop eyes and save as .npy with shape 13\*48\*48\*3
    3. Change the root_path and npy_path of SynBlinkDataset to your data_path
    4. Put json files in data_split/SynBlink to your data_path

# Test on HUST-LEBW dataset

downlaod weights from [link](https://pan.baidu.com/s/1NN_Y5Uiwpxx7-sAA4L0Cyg?pwd=synb)

```bash
conda activate your_venv

python main.py \
    -batch_size 128 \
    -num_workers 4 \
    -arch 'BlinkFormer' \
    -dataset 'hust' \
    -exp_name 'BlinkFormer-HUST-LEBW' \
    -dim 1024 \
    -depth 6 \
    -heads 16 \
    -mlp_dim 2048 \
    -mode 'test' \
    -test_ckpt_path 'weights_BlinkFormer_HUST-LEBW_F1_0.843.pth'
```

# Train

- train BlinkFormer on SynBlink dataset
```bash
conda activate your_venv

python main.py \
    -lr 0.0001  \
    -epoch 50 \
    -batch_size 128 \
    -num_workers 4 \
    -arch 'BlinkFormer' \
    -dataset 'synblink50knpy' \
    -exp_name 'BlinkFormer-SynBlink-train' \
    -mode 'train' \
    -dim 512 \
    -depth 6 \
    -heads 16 \
    -mlp_dim 512 \
    -mode 'train'
```
- train BlinkFormer_with_BSE_head  on SynBlink dataset
```bash
conda activate your_venv

python main.py \
    -lr 0.0001  \
    -epoch 50 \
    -batch_size 128 \
    -num_workers 4 \
    -arch 'BlinkFormer_with_BSE_head' \
    -dataset 'synblink50knpy' \
    -exp_name 'BlinkFormer_with_BSE_head-SynBlink-train' \
    -dim 512 \
    -depth 6 \
    -heads 16 \
    -mlp_dim 512 \
    -mode 'train'
```

- train on HUST-LEBW dataset
  
```bash
conda activate your_venv

python main.py \
    -lr 0.00001  \
    -epoch 500 \
    -batch_size 128 \
    -num_workers 4 \
    -arch 'BlinkFormer' \
    -dataset 'hust' \
    -exp_name 'BlinkFormer-HUST-LEBW-train' \
    -mode 'train' \
    -dim 1024 \
    -depth 6 \
    -heads 16 \
    -mlp_dim 2048
```