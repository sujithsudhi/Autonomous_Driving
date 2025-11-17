# Autonomous Driving – BEVFormer Lite

Camera-only BEV perception experiments based on nuScenes. The repository provides a Swin-transformer backbone, BEV transformer encoder/head, and nuScenes dataloaders wired together through `training/train_bevformer_lite.py`.

## Usage

1. Install dependencies listed in `requirements.txt` and make sure the nuScenes devkit plus dataset are available.
2. Adjust `configs/bevformer_lite_nuscenes.yaml` for dataset paths, camera selection, BEV bounds/resolution, model hyperparameters, loss weights, and optimization parameters (including grad clipping). The YAML is parsed into a structured configuration (mirroring the layout used in the Transformers repo) and a copy is saved alongside training outputs for reproducibility.
3. Train (with built-in validation + mixed precision + chunked attention) via:
   ```bash
   python training/train_bevformer_lite.py --config configs/bevformer_lite_nuscenes.yaml
   ```

The script prints per-epoch train/val progress on a single line, saves checkpoints to `outputs/`, and summarizes losses at the end. Extend the config/model files as needed for your experiments.
