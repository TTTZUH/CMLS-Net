"""
Test script for CMLS-Net
"""

import os
import sys
import yaml
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_cmls_net_tiny, build_cmls_net_small, build_cmls_net_large
from data import get_dataset, get_transform
from utils import SegmentationMetrics, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Test CMLS-Net')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--gpu', type=str, default='0')
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger('CMLS-Net-Test')
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    
    # Build model
    logger.info("Loading model...")
    if config['model']['name'] == 'cmls_net_tiny':
        model = build_cmls_net_tiny(config['dataset']['num_classes'])
    elif config['model']['name'] == 'cmls_net_small':
        model = build_cmls_net_small(config['dataset']['num_classes'])
    else:
        model = build_cmls_net_large(config['dataset']['num_classes'])
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load test dataset
    logger.info("Loading test dataset...")
    test_transform = get_transform('test', image_size=config['dataset']['image_size'])
    test_dataset = get_dataset(config['dataset']['name'],
                               config['dataset']['root_dir'],
                               'test', test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config['evaluation']['batch_size'],
                            shuffle=False, num_workers=config['evaluation']['num_workers'])
    
    # Evaluate
    logger.info("Testing...")
    seg_metrics = SegmentationMetrics(config['dataset']['num_classes'])
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            seg_metrics.update(outputs, labels)
    
    results = seg_metrics.get_all_metrics()
    logger.info("=" * 80)
    logger.info("Test Results:")
    logger.info(f"  mIoU: {results['mIoU']:.4f}")
    logger.info(f"  mPA: {results['mPA']:.4f}")
    logger.info(f"  mF1: {results['mF1']:.4f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
