# PanDerm - Skin Lesion Segmentation

This implementation uses PanDerm for skin lesion segmentation.

## Install Environment

Main dependencies: Pytorch, Pytorch Lightning, MMSegmentation

```bash
cd segmentation
pip install -r requirements.txt
```

## Data Preparation

1. Download datasets:
   - [ISIC2018 task1](https://challenge.isic-archive.com/data/#2018)
   - [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
2. Update `_load_name()` in `./datasets/dataset_seg.py` with your data path

## Fine-tuning with PanDerm Weight

1. **Download Pretrained Weights**
   - [PanDerm weights](https://drive.google.com/file/d/1XHKRk2p-dS1PFQE-xRbOM3yx47i3bXmi/view?usp=sharing)

2. **Set Pretrained Path**
   - Update `cae_weight` in `./models/cae_seg.py`
   - Modify `pretrained` parameter in `run.sh`

3. **Start Training**
   ```bash
   cd segmentation
   bash run.sh
   ```
   Note: Adjust the path config to your desired storage location

4. **Evaluation**
   - Add `--evaluate` to `run.sh` for evaluation mode
   - Model fine-tuned weights for evaluation on HAM10000 dataset [Download here](https://drive.google.com/drive/folders/1BsSjl1h3mxU6JNSbqvgZdyiTvV_2QBsH?usp=sharing)
   ```bash
   cd segmentation
   bash run.sh --evaluate
   ```
   This loads the checkpoint from your specified model storage path

## Starter Code for Your Own Dermoscopic Images

See [**evaluate.ipynb**](segmentation/evaluate.ipynb) to learn:
- Loading our pre-trained model
- Using it for skin lesion segmentation
