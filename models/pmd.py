"""
Progressive Multi-Path Decoder (PMD) Module
Realizes gradual feature refinement through multi-path parallelism 
and inter-branch information exchange
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderBlock(nn.Module):
    """Basic decoder block with upsampling and feature refinement"""
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(DecoderBlock, self).__init__()
        
        self.scale_factor = scale_factor
        
        # Channel reduction
        self.conv_reduce = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True)
        )
        
        # Feature refinement
        self.conv_refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Channel reduction
        x = self.conv_reduce(x)
        
        # Bilinear upsampling
        if self.scale_factor > 1:
            x = F.interpolate(x, scale_factor=self.scale_factor, 
                            mode='bilinear', align_corners=True)
        
        # Feature refinement
        x = self.conv_refine(x)
        
        return x


class MultiPathFusion(nn.Module):
    """Multi-path feature fusion with inter-branch information exchange"""
    def __init__(self, num_paths, channels):
        super(MultiPathFusion, self).__init__()
        
        self.num_paths = num_paths
        
        # Cross-path convolutions for information exchange
        self.cross_convs = nn.ModuleList([
            nn.Conv2d(channels, channels, 3, padding=1)
            for _ in range(num_paths)
        ])
        
        # Attention weights for adaptive fusion
        self.attention = nn.Sequential(
            nn.Conv2d(channels * num_paths, num_paths, 1),
            nn.Softmax(dim=1)
        )
        
        # Final fusion
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, path_features):
        """
        Args:
            path_features: List of features from multiple paths, 
                          each with shape [B, C, H, W]
        """
        # Ensure all features have the same spatial size
        target_size = path_features[0].shape[2:]
        aligned_features = []
        for feat in path_features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, 
                                   mode='bilinear', align_corners=True)
            aligned_features.append(feat)
        
        # Cross-path information exchange
        exchanged_features = []
        for i, feat in enumerate(aligned_features):
            exchanged = self.cross_convs[i](feat)
            exchanged_features.append(exchanged)
        
        # Concatenate for attention
        concat_feat = torch.cat(exchanged_features, dim=1)
        
        # Compute attention weights
        attn_weights = self.attention(concat_feat)  # [B, num_paths, H, W]
        
        # Weighted fusion
        fused = 0
        for i, feat in enumerate(exchanged_features):
            weight = attn_weights[:, i:i+1, :, :]  # [B, 1, H, W]
            fused = fused + weight * feat
        
        # Final refinement
        out = self.fusion_conv(fused)
        
        return out


class ProgressiveDecoderStage(nn.Module):
    """Single stage of progressive decoder with multiple paths"""
    def __init__(self, num_paths, in_channels, out_channels, scale_factor=2):
        super(ProgressiveDecoderStage, self).__init__()
        
        self.num_paths = num_paths
        
        # Decoder blocks for each path
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(in_channels, out_channels, scale_factor)
            for _ in range(num_paths)
        ])
        
        # Multi-path fusion
        self.fusion = MultiPathFusion(num_paths, out_channels)
        
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of encoder features at different resolutions
        """
        # Process each path
        path_outputs = []
        for i in range(self.num_paths):
            if i < len(encoder_features):
                path_out = self.decoder_blocks[i](encoder_features[i])
            else:
                # If not enough encoder features, reuse the last one
                path_out = self.decoder_blocks[i](encoder_features[-1])
            path_outputs.append(path_out)
        
        # Fuse multiple paths
        fused_output = self.fusion(path_outputs)
        
        return fused_output, path_outputs


class PMD(nn.Module):
    """
    Progressive Multi-Path Decoder
    Cascaded fusion strategy with multi-path parallelism
    """
    def __init__(self, encoder_channels=[256, 128, 64, 32], 
                 decoder_channels=[128, 64, 32], 
                 num_paths_per_stage=[2, 3, 4],
                 num_classes=6):
        super(PMD, self).__init__()
        
        assert len(decoder_channels) == len(num_paths_per_stage), \
            "decoder_channels and num_paths_per_stage must have same length"
        
        self.num_stages = len(decoder_channels)
        
        # Progressive decoder stages
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            in_ch = encoder_channels[i]
            out_ch = decoder_channels[i]
            num_paths = num_paths_per_stage[i]
            
            stage = ProgressiveDecoderStage(
                num_paths=num_paths,
                in_channels=in_ch,
                out_channels=out_ch,
                scale_factor=2
            )
            self.stages.append(stage)
        
        # Final upsampling and segmentation head
        self.final_upsample = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], decoder_channels[-1], 3, padding=1),
            nn.BatchNorm2d(decoder_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(decoder_channels[-1], num_classes, 1)
        )
        
    def forward(self, encoder_features):
        """
        Args:
            encoder_features: List of encoder features from low to high resolution
                            [f_32x, f_16x, f_8x, f_4x]
        Returns:
            segmentation map [B, num_classes, H, W]
        """
        # Reverse to process from low resolution to high resolution
        encoder_features = list(reversed(encoder_features))
        
        # Progressive decoding
        x = encoder_features[0]
        for i, stage in enumerate(self.stages):
            # Get features for this stage
            if i == 0:
                stage_features = [encoder_features[0]]
            else:
                stage_features = encoder_features[:i+1]
            
            # Progressive fusion
            x, _ = stage(stage_features)
            
            # Add skip connection from encoder if available
            if i + 1 < len(encoder_features):
                skip = encoder_features[i + 1]
                # Align spatial size
                if x.shape[2:] != skip.shape[2:]:
                    x = F.interpolate(x, size=skip.shape[2:], 
                                    mode='bilinear', align_corners=True)
                # Add skip connection
                x = x + skip
        
        # Final segmentation
        out = self.final_upsample(x)
        
        return out


if __name__ == "__main__":
    # Test PMD module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Simulated encoder features at different resolutions
    # [low_res -> high_res]
    f_32x = torch.randn(2, 256, 8, 8).to(device)    # 32x downsampled
    f_16x = torch.randn(2, 128, 16, 16).to(device)  # 16x downsampled
    f_8x = torch.randn(2, 64, 32, 32).to(device)    # 8x downsampled
    f_4x = torch.randn(2, 32, 64, 64).to(device)    # 4x downsampled
    
    encoder_features = [f_32x, f_16x, f_8x, f_4x]
    
    pmd = PMD(
        encoder_channels=[256, 128, 64, 32],
        decoder_channels=[128, 64, 32],
        num_paths_per_stage=[2, 3, 4],
        num_classes=6
    ).to(device)
    
    out = pmd(encoder_features)
    print(f"Encoder feature shapes: {[f.shape for f in encoder_features]}")
    print(f"Output shape: {out.shape}")
    print("PMD module test passed!")
