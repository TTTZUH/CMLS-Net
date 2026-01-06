"""
Training script for CMLS-Net
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_cmls_net_tiny, build_cmls_net_small, build_cmls_net_large
from data import get_dataset, get_transform
from utils import (
    get_loss_function, SegmentationMetrics,
    setup_logger, MetricLogger, log_config, log_epoch_info, log_model_info
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train CMLS-Net')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU id to use')
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def build_model(config):
    num_classes = config['dataset']['num_classes']
    model_name = config['model']['name']
    
    if model_name == 'cmls_net_tiny':
        model = build_cmls_net_tiny(num_classes)
    elif model_name == 'cmls_net_small':
        model = build_cmls_net_small(num_classes)
    elif model_name == 'cmls_net_large':
        model = build_cmls_net_large(num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model


def build_optimizer(model, config):
    opt_config = config['training']['optimizer']
    opt_type = opt_config['type'].lower()
    
    if opt_type == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=opt_config['lr'],
            betas=opt_config.get('betas', [0.9, 0.999]),
            weight_decay=opt_config.get('weight_decay', 0.0001)
        )
    elif opt_type == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_config['lr'],
            betas=opt_config.get('betas', [0.9, 0.999]),
            weight_decay=opt_config.get('weight_decay', 0.0001)
        )
    elif opt_type == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=opt_config['lr'],
            momentum=opt_config.get('momentum', 0.9),
            weight_decay=opt_config.get('weight_decay', 0.0001)
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")
    
    return optimizer


def build_scheduler(optimizer, config, steps_per_epoch):
    sch_config = config['training']['scheduler']
    sch_type = sch_config['type'].lower()
    num_epochs = config['training']['num_epochs']
    warmup_epochs = sch_config.get('warmup_epochs', 0)
    
    if sch_type == 'polynomial':
        def poly_lr_lambda(epoch):
            if epoch < warmup_epochs:
                return epoch / warmup_epochs
            else:
                power = sch_config.get('power', 0.9)
                min_lr = sch_config.get('min_lr', 0.000001)
                base_lr = config['training']['optimizer']['lr']
                return max(min_lr / base_lr, 
                          (1 - (epoch - warmup_epochs) / (num_epochs - warmup_epochs)) ** power)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, poly_lr_lambda)
    
    elif sch_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=num_epochs - warmup_epochs,
            eta_min=sch_config.get('min_lr', 0.000001)
        )
    
    elif sch_type == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=sch_config.get('step_size', 30),
            gamma=sch_config.get('gamma', 0.1)
        )
    
    else:
        raise ValueError(f"Unknown scheduler: {sch_type}")
    
    return scheduler


def train_one_epoch(model, dataloader, criterion, optimizer, scaler, device, logger, epoch):
    model.train()
    metric_logger = MetricLogger()
    
    pbar = tqdm(dataloader, desc=f'Train Epoch {epoch}')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            outputs = model(images)
            if isinstance(criterion.forward(outputs, labels), tuple):
                loss, ce_loss, dice_loss = criterion(outputs, labels)
                metric_logger.update(loss=loss.item(), ce=ce_loss.item(), dice=dice_loss.item())
            else:
                loss = criterion(outputs, labels)
                metric_logger.update(loss=loss.item())
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return metric_logger.get_metrics()


def evaluate(model, dataloader, criterion, device, num_classes, logger, epoch):
    model.eval()
    metric_logger = MetricLogger()
    seg_metrics = SegmentationMetrics(num_classes, ignore_index=255)
    
    pbar = tqdm(dataloader, desc=f'Val Epoch {epoch}')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            if isinstance(criterion.forward(outputs, labels), tuple):
                loss, ce_loss, dice_loss = criterion(outputs, labels)
                metric_logger.update(loss=loss.item(), ce=ce_loss.item(), dice=dice_loss.item())
            else:
                loss = criterion(outputs, labels)
                metric_logger.update(loss=loss.item())
            
            seg_metrics.update(outputs, labels)
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    seg_results = seg_metrics.get_all_metrics()
    results = metric_logger.get_metrics()
    results.update({
        'mIoU': seg_results['mIoU'],
        'mPA': seg_results['mPA'],
        'mF1': seg_results['mF1']
    })
    
    return results


def main():
    args = parse_args()
    
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Setup directories
    os.makedirs(config['training']['save_dir'], exist_ok=True)
    os.makedirs(config['training']['log_dir'], exist_ok=True)
    
    # Setup logger
    log_file = os.path.join(config['training']['log_dir'], 'train.log')
    logger = setup_logger('CMLS-Net', log_file)
    log_config(logger, config)
    
    # Device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Build model
    logger.info("Building model...")
    model = build_model(config)
    model = model.to(device)
    log_model_info(logger, model)
    
    # Build datasets
    logger.info("Loading datasets...")
    train_transform = get_transform('train', config['training']['augmentation']['level'],
                                   config['dataset']['image_size'])
    val_transform = get_transform('val', image_size=config['dataset']['image_size'])
    
    train_dataset = get_dataset(config['dataset']['name'], 
                               config['dataset']['root_dir'],
                               'train', train_transform)
    val_dataset = get_dataset(config['dataset']['name'],
                             config['dataset']['root_dir'],
                             'val', val_transform)
    
    train_loader = DataLoader(train_dataset, 
                             batch_size=config['training']['batch_size'],
                             shuffle=True,
                             num_workers=config['training']['num_workers'],
                             pin_memory=config['training']['pin_memory'])
    val_loader = DataLoader(val_dataset,
                           batch_size=config['evaluation']['batch_size'],
                           shuffle=False,
                           num_workers=config['evaluation']['num_workers'])
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Build loss
    criterion = get_loss_function(config['training']['loss']['type'],
                                 config['dataset']['num_classes'])
    
    # Build optimizer and scheduler
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, len(train_loader))
    
    # Mixed precision scaler
    scaler = GradScaler() if config['training']['use_amp'] else None
    
    # Training loop
    best_miou = 0.0
    start_epoch = 1
    
    logger.info("Starting training...")
    for epoch in range(start_epoch, config['training']['num_epochs'] + 1):
        # Train
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer,
                                       scaler, device, logger, epoch)
        log_epoch_info(logger, epoch, config['training']['num_epochs'], 
                      train_metrics, "Train")
        
        # Evaluate
        if epoch % config['training']['eval_interval'] == 0:
            val_metrics = evaluate(model, val_loader, criterion, device,
                                  config['dataset']['num_classes'], logger, epoch)
            log_epoch_info(logger, epoch, config['training']['num_epochs'],
                          val_metrics, "Val")
            
            # Save best model
            if val_metrics['mIoU'] > best_miou:
                best_miou = val_metrics['mIoU']
                save_path = os.path.join(config['training']['save_dir'], 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_miou': best_miou,
                    'config': config
                }, save_path)
                logger.info(f"Saved best model with mIoU: {best_miou:.4f}")
        
        # Save checkpoint
        if epoch % config['training']['save_interval'] == 0:
            save_path = os.path.join(config['training']['save_dir'], 
                                    f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }, save_path)
        
        scheduler.step()
    
    logger.info(f"Training completed! Best mIoU: {best_miou:.4f}")


if __name__ == '__main__':
    main()
