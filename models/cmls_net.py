"""
CMLS-Net: Collaborative Multi-Level Semantic Network
Main network architecture integrating MVFE, CSFT, and PMD modules
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrnet import HRNet
from .mvfe import MVFE
from .csft import CSFT
from .pmd import PMD


class CMLS_Net(nn.Module):
    """
    Collaborative Multi-Level Semantic Network
    
    Architecture:
        - HRNet-based encoder with multi-resolution parallel branches
        - MVFE modules for multi-view feature enhancement
        - CSFT modules for collaborative semantic fusion
        - PMD for progressive multi-path decoding
    """
    
    def __init__(self, 
                 num_classes=6,
                 backbone_config='W40',
                 mvfe_groups=10,
                 csft_heads=8,
                 csft_window=4):
        super(CMLS_Net, self).__init__()
        
        # HRNet Backbone
        self.backbone = HRNet(config=backbone_config)
        
        # Get channel dimensions based on config
        if backbone_config == 'W40':
            self.branch_channels = [40, 80, 160, 320]
        elif backbone_config == 'W32':
            self.branch_channels = [32, 64, 128, 256]
        else:
            raise ValueError(f"Unknown backbone config: {backbone_config}")
        
        # MVFE modules for each stage (after Stage 2, 3, 4)
        self.mvfe_stage2 = nn.ModuleList([
            MVFE(ch, num_groups=mvfe_groups) 
            for ch in self.branch_channels[:2]
        ])
        
        self.mvfe_stage3 = nn.ModuleList([
            MVFE(ch, num_groups=mvfe_groups) 
            for ch in self.branch_channels[:3]
        ])
        
        self.mvfe_stage4 = nn.ModuleList([
            MVFE(ch, num_groups=mvfe_groups) 
            for ch in self.branch_channels
        ])
        
        # CSFT modules for semantic fusion
        # After each HRNet stage with MVFE enhancement
        self.csft_stage2 = CSFT(self.branch_channels[0], csft_heads, csft_window)
        self.csft_stage3 = CSFT(self.branch_channels[0], csft_heads, csft_window)
        self.csft_stage4 = CSFT(self.branch_channels[0], csft_heads, csft_window)
        
        # Feature alignment layers for CSFT
        # Align all features to the highest resolution (branch 0)
        self.align_stage2 = self._make_alignment_layers(self.branch_channels[:2])
        self.align_stage3 = self._make_alignment_layers(self.branch_channels[:3])
        self.align_stage4 = self._make_alignment_layers(self.branch_channels)
        
        # Progressive Multi-Path Decoder
        self.decoder = PMD(
            encoder_channels=self.branch_channels[::-1],  # Reverse order
            decoder_channels=[self.branch_channels[1], self.branch_channels[0], 
                            self.branch_channels[0] // 2],
            num_paths_per_stage=[2, 3, 4],
            num_classes=num_classes
        )
        
    def _make_alignment_layers(self, channels):
        """Create alignment layers to unify multi-scale features"""
        target_ch = channels[0]
        layers = nn.ModuleList()
        
        for i, ch in enumerate(channels):
            if i == 0:
                layers.append(None)  # No alignment needed for highest resolution
            else:
                layers.append(nn.Sequential(
                    nn.Conv2d(ch, target_ch, 1, bias=False),
                    nn.BatchNorm2d(target_ch),
                    nn.ReLU(inplace=True)
                ))
        
        return layers
    
    def _apply_mvfe(self, features, mvfe_modules):
        """Apply MVFE to multi-scale features"""
        enhanced = []
        for i, feat in enumerate(features):
            if i < len(mvfe_modules):
                enhanced.append(mvfe_modules[i](feat))
            else:
                enhanced.append(feat)
        return enhanced
    
    def _fuse_with_csft(self, features, csft_module, align_layers):
        """
        Fuse multi-scale features using CSFT
        
        Args:
            features: List of multi-scale features
            csft_module: CSFT module
            align_layers: Alignment layers
            
        Returns:
            Fused feature at highest resolution
        """
        # Align all features to highest resolution
        target_size = features[0].shape[2:]
        aligned = []
        
        for i, feat in enumerate(features):
            # Channel alignment
            if align_layers[i] is not None:
                feat = align_layers[i](feat)
            
            # Spatial alignment
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, 
                                   mode='bilinear', align_corners=True)
            aligned.append(feat)
        
        # Prepare three types of features for CSFT
        # Spatial: highest resolution feature
        f_spatial = aligned[0]
        
        # Global: Average of all scales
        f_global = torch.stack(aligned, dim=0).mean(dim=0)
        
        # Multi-scale: Concatenate and project
        if len(aligned) > 1:
            f_multiscale = torch.cat(aligned, dim=1)
            # Project back to target channels
            conv_proj = nn.Conv2d(f_multiscale.shape[1], aligned[0].shape[1], 1).to(f_multiscale.device)
            f_multiscale = conv_proj(f_multiscale)
        else:
            f_multiscale = aligned[0]
        
        # Apply CSFT
        fused = csft_module(f_spatial, f_global, f_multiscale)
        
        return fused
    
    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W]
            
        Returns:
            Segmentation map [B, num_classes, H, W]
        """
        B, C, H, W = x.shape
        
        # Backbone - returns multi-scale features
        # [high_res -> low_res]: [f1, f2, f3, f4]
        backbone_features = self.backbone(x)
        
        # Stage 2: MVFE + CSFT (2 branches)
        features_2 = self._apply_mvfe(backbone_features[:2], self.mvfe_stage2)
        fused_2 = self._fuse_with_csft(features_2, self.csft_stage2, self.align_stage2)
        
        # Stage 3: MVFE + CSFT (3 branches)
        features_3 = self._apply_mvfe(backbone_features[:3], self.mvfe_stage3)
        fused_3 = self._fuse_with_csft(features_3, self.csft_stage3, self.align_stage3)
        
        # Stage 4: MVFE + CSFT (4 branches)
        features_4 = self._apply_mvfe(backbone_features, self.mvfe_stage4)
        fused_4 = self._fuse_with_csft(features_4, self.csft_stage4, self.align_stage4)
        
        # Prepare decoder inputs
        # Use original multi-scale features with fusion results
        decoder_inputs = [
            backbone_features[3],  # Lowest resolution
            backbone_features[2],
            backbone_features[1],
            fused_4               # Highest resolution with fusion
        ]
        
        # Progressive Multi-Path Decoder
        seg_map = self.decoder(decoder_inputs)
        
        # Upsample to input size
        if seg_map.shape[2:] != (H, W):
            seg_map = F.interpolate(seg_map, size=(H, W), 
                                   mode='bilinear', align_corners=True)
        
        return seg_map


def build_cmls_net_tiny(num_classes=6):
    """Build CMLS-Net-Tiny variant"""
    return CMLS_Net(
        num_classes=num_classes,
        backbone_config='W32',
        mvfe_groups=8,
        csft_heads=4,
        csft_window=2
    )


def build_cmls_net_small(num_classes=6):
    """Build CMLS-Net-Small variant"""
    return CMLS_Net(
        num_classes=num_classes,
        backbone_config='W32',
        mvfe_groups=9,
        csft_heads=4,
        csft_window=4
    )


def build_cmls_net_large(num_classes=6):
    """Build CMLS-Net-Large variant"""
    return CMLS_Net(
        num_classes=num_classes,
        backbone_config='W40',
        mvfe_groups=10,
        csft_heads=8,
        csft_window=4
    )


if __name__ == "__main__":
    # Test CMLS-Net
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x = torch.randn(2, 3, 256, 256).to(device)
    
    # Test different variants
    print("Testing CMLS-Net variants...")
    
    model_tiny = build_cmls_net_tiny(num_classes=6).to(device)
    out_tiny = model_tiny(x)
    print(f"CMLS-Net-Tiny output shape: {out_tiny.shape}")
    
    model_small = build_cmls_net_small(num_classes=6).to(device)
    out_small = model_small(x)
    print(f"CMLS-Net-Small output shape: {out_small.shape}")
    
    model_large = build_cmls_net_large(num_classes=6).to(device)
    out_large = model_large(x)
    print(f"CMLS-Net-Large output shape: {out_large.shape}")
    
    print("CMLS-Net test passed!")
