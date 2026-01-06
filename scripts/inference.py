"""
Inference script for CMLS-Net
"""

import os
import sys
import yaml
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_cmls_net_tiny, build_cmls_net_small, build_cmls_net_large
import albumentations as A
from albumentations.pytorch import ToTensorV2


def parse_args():
    parser = argparse.ArgumentParser(description='Inference with CMLS-Net')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--output', type=str, default='output.png')
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()


def load_image(image_path, size=256):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    transform = A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    augmented = transform(image=image)
    return augmented['image'].unsqueeze(0), image


def visualize_prediction(image, pred_mask, class_names, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    axes[1].imshow(pred_mask, cmap='tab10')
    axes[1].set_title('Prediction')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Result saved to {output_path}")


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Build and load model
    if config['model']['name'] == 'cmls_net_tiny':
        model = build_cmls_net_tiny(config['dataset']['num_classes'])
    elif config['model']['name'] == 'cmls_net_small':
        model = build_cmls_net_small(config['dataset']['num_classes'])
    else:
        model = build_cmls_net_large(config['dataset']['num_classes'])
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and predict
    image_tensor, original_image = load_image(args.image, config['dataset']['image_size'])
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        pred = torch.argmax(output, dim=1).cpu().numpy()[0]
    
    # Visualize
    visualize_prediction(original_image, pred, config['dataset']['class_names'], args.output)


if __name__ == '__main__':
    main()
