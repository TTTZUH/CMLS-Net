# CMLS-Net: Collaborative Multi-Level Semantic Network

Official PyTorch implementation of **CMLS-Net** for remote sensing image semantic segmentation.

## 📋 Overview

CMLS-Net achieves hierarchical semantic modeling through three innovative modules:
- **MVFE (Multi-View Feature Enhancement)**: Integrates spatial-frequency dual perspectives with adaptive receptive field expansion
- **CSFT (Collaborative Semantic Fusion Transformer)**: Establishes deep associations between heterogeneous features through cross-branch key-value interaction
- **PMD (Progressive Multi-Path Decoder)**: Realizes gradual feature refinement through multi-path parallelism

## 🎯 Key Features

- ✅ State-of-the-art performance on WHDLD, Vaihingen, and Potsdam datasets
- ✅ Three model variants (Tiny, Small, Large) for different computational budgets
- ✅ Efficient implementation with mixed precision training
- ✅ Comprehensive data augmentation pipeline
- ✅ Easy-to-use training and inference scripts

## 📊 Performance

| Model | WHDLD (mIoU) | Vaihingen (mIoU) | Potsdam (mIoU) | Params (M) |
|-------|--------------|------------------|----------------|------------|
| CMLS-T | 60.15% | 79.45% | 69.78% | 29.3 |
| CMLS-S | 62.89% | 81.67% | 72.01% | 37.1 |
| CMLS-L | **64.73%** | **83.28%** | **73.45%** | 45.7 |

## 🚀 Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CMLS-Net.git
cd CMLS-Net

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

#### WHDLD Dataset
```
WHDLD/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

#### Vaihingen/Potsdam Dataset
```
Vaihingen/
├── training/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Training

1. Update the dataset path in config file (`configs/config_whdld.yaml`)

2. Start training:
```bash
python scripts/train.py --config configs/config_whdld.yaml --gpu 0
```

### Testing

```bash
python scripts/test.py \
    --config configs/config_whdld.yaml \
    --checkpoint checkpoints/whdld/best_model.pth \
    --gpu 0
```

### Inference

```bash
python scripts/inference.py \
    --config configs/config_whdld.yaml \
    --checkpoint checkpoints/whdld/best_model.pth \
    --image path/to/image.jpg \
    --output output.png \
    --gpu 0
```

## 📁 Project Structure

```
CMLS-Net/
├── models/
│   ├── __init__.py
│   ├── mvfe.py              # Multi-View Feature Enhancement
│   ├── csft.py              # Collaborative Semantic Fusion Transformer
│   ├── pmd.py               # Progressive Multi-Path Decoder
│   ├── hrnet.py             # HRNet backbone
│   └── cmls_net.py          # Main network
├── utils/
│   ├── __init__.py
│   ├── losses.py            # Loss functions
│   ├── metrics.py           # Evaluation metrics
│   └── logger.py            # Logging utilities
├── data/
│   ├── __init__.py
│   ├── dataset.py           # Dataset loaders
│   └── transforms.py        # Data augmentation
├── configs/
│   ├── config_whdld.yaml
│   ├── config_vaihingen.yaml
│   └── config_potsdam.yaml
├── scripts/
│   ├── train.py             # Training script
│   ├── test.py              # Testing script
│   └── inference.py         # Inference script
├── checkpoints/             # Model checkpoints
├── requirements.txt
└── README.md
```

## 🔧 Configuration

Key configuration options in `configs/config_*.yaml`:

```yaml
# Model settings
model:
  name: 'cmls_net_large'  # Options: cmls_net_tiny, cmls_net_small, cmls_net_large
  
# Training settings
training:
  batch_size: 16
  num_epochs: 100
  optimizer:
    type: 'adamw'
    lr: 0.0001
  loss:
    type: 'combined'  # CE + Dice loss
```

## 📈 Training Tips

1. **Batch Size**: Adjust based on GPU memory (16 for 24GB GPU)
2. **Learning Rate**: Start with 1e-4, use warmup for stable training
3. **Augmentation**: Use 'normal' level for best results
4. **Mixed Precision**: Enable `use_amp: true` for faster training

## 🔬 Model Variants

| Variant | Groups | Heads | Window | Use Case |
|---------|--------|-------|--------|----------|
| CMLS-T | 8 | 4 | 2 | Fast inference, limited resources |
| CMLS-S | 9 | 4 | 4 | Balanced accuracy and speed |
| CMLS-L | 10 | 8 | 4 | Best accuracy, sufficient resources |

## 📝 Citation

If you find this work useful, please cite:

```bibtex
@article{cmls_net_2025,
  title={Collaborative Multi-Level Semantic Network for Remote Sensing Image Segmentation via Progressive Feature Refinement},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

## 📄 License

This project is released under the MIT License.

## 🤝 Acknowledgements

- HRNet backbone implementation based on [HRNet-Semantic-Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation)
- Albumentations for data augmentation
- PyTorch team for the deep learning framework

## 📧 Contact

For questions and feedback:
- Email: your.email@example.com
- Issues: https://github.com/yourusername/CMLS-Net/issues

---

**Note**: This is a research implementation. For production use, please ensure thorough testing and validation.
