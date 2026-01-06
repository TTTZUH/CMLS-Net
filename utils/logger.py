"""
Logger utility for training and evaluation
"""

import logging
import os
from datetime import datetime


def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Format
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file provided)
    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class MetricLogger:
    """Log and track metrics during training"""
    
    def __init__(self, delimiter=" | "):
        self.meters = {}
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        """Update metrics"""
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = AverageMeter()
            self.meters[k].update(v)
    
    def __str__(self):
        """String representation"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {meter.avg:.4f}")
        return self.delimiter.join(loss_str)
    
    def get_metrics(self):
        """Get all metric values"""
        return {name: meter.avg for name, meter in self.meters.items()}
    
    def reset(self):
        """Reset all meters"""
        for meter in self.meters.values():
            meter.reset()


class AverageMeter:
    """Compute and store the average and current value"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def log_config(logger, config):
    """Log configuration"""
    logger.info("=" * 80)
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 80)


def log_epoch_info(logger, epoch, total_epochs, metrics, prefix="Train"):
    """Log epoch information"""
    metric_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"{prefix} Epoch [{epoch}/{total_epochs}] | {metric_str}")


def log_model_info(logger, model):
    """Log model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info("=" * 80)
    logger.info("Model Information:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Non-trainable parameters: {total_params - trainable_params:,}")
    logger.info("=" * 80)


if __name__ == "__main__":
    # Test logger
    print("Testing logger utilities...")
    
    # Setup logger
    logger = setup_logger('test', log_file='test_log.txt')
    logger.info("This is a test message")
    
    # Test MetricLogger
    print("\nTesting MetricLogger:")
    metric_logger = MetricLogger()
    
    # Simulate training loop
    for i in range(5):
        metric_logger.update(loss=0.5 - i*0.05, acc=0.8 + i*0.02)
    
    print(f"Metrics: {metric_logger}")
    print(f"Final metrics: {metric_logger.get_metrics()}")
    
    # Test AverageMeter
    print("\nTesting AverageMeter:")
    meter = AverageMeter()
    for i in range(10):
        meter.update(i)
    print(f"Average: {meter.avg}, Sum: {meter.sum}, Count: {meter.count}")
    
    print("\nAll logger utilities tested successfully!")
    
    # Clean up
    if os.path.exists('test_log.txt'):
        os.remove('test_log.txt')
