"""
Multi-View Feature Enhancement (MVFE) Module
Integrates spatial-frequency dual perspectives with adaptive receptive field expansion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftPool2d(nn.Module):
    """Soft pooling that preserves gradient flow"""
    def __init__(self, kernel_size=3, stride=1):
        super(SoftPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        
    def forward(self, x):
        x_exp = torch.exp(x)
        x_exp_pool = F.avg_pool2d(x_exp, self.kernel_size, self.stride, padding=self.kernel_size//2)
        x_pool = F.avg_pool2d(x * x_exp, self.kernel_size, self.stride, padding=self.kernel_size//2)
        return x_pool / (x_exp_pool + 1e-8)


class SpatialAttention(nn.Module):
    """
    Spatial Attention mechanism with grouped feature processing 
    and multi-directional aggregation
    """
    def __init__(self, in_channels, num_groups=8):
        super(SpatialAttention, self).__init__()
        self.num_groups = num_groups
        self.channels_per_group = in_channels // num_groups
        
        # Soft pooling for directional features
        self.softpool = SoftPool2d()
        
        # Convolutions for attention generation
        self.conv_x = nn.Conv2d(self.channels_per_group, self.channels_per_group, 1)
        self.conv_y = nn.Conv2d(self.channels_per_group, self.channels_per_group, 1)
        
        # Channel attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.channels_per_group, self.channels_per_group, 1),
            nn.Sigmoid()
        )
        
        self.bn = nn.BatchNorm2d(in_channels)
        self.final_conv = nn.Conv2d(in_channels, in_channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Split into groups
        x_groups = x.view(B, self.num_groups, self.channels_per_group, H, W)
        
        enhanced_groups = []
        for g in range(self.num_groups):
            x_g = x_groups[:, g, :, :, :]  # [B, C/g, H, W]
            
            # Directional pooling
            x_pool_h = F.adaptive_avg_pool2d(x_g, (H, 1))  # [B, C/g, H, 1]
            x_pool_w = F.adaptive_avg_pool2d(x_g, (1, W))  # [B, C/g, 1, W]
            
            # Generate attention maps
            attn_x = torch.softmax(self.conv_x(x_pool_h), dim=2)
            attn_y = torch.softmax(self.conv_y(x_pool_w), dim=3)
            
            # Combine attentions
            attn_spatial = torch.matmul(attn_x, attn_y)  # [B, C/g, H, W]
            attn_spatial = self.bn(attn_spatial)
            
            # Apply spatial weights
            spatial_weighted = torch.sigmoid(attn_spatial) * torch.sigmoid(
                F.adaptive_avg_pool2d(attn_spatial, 1)
            )
            
            # Channel attention
            channel_weights = self.channel_attention(x_g)
            
            # Dual weighting
            x_enhanced = (spatial_weighted * torch.sigmoid(x_g)) * channel_weights
            enhanced_groups.append(x_enhanced)
        
        # Concatenate groups
        out = torch.cat(enhanced_groups, dim=1)
        out = self.final_conv(out)
        
        return out


class FrequencyAwareAdaptiveReceptiveField(nn.Module):
    """
    Frequency-aware Adaptive Receptive Field (FARM) module
    Decomposes features into frequency bands and applies adaptive dilation
    """
    def __init__(self, in_channels, r_low=4, r_mid=2, r_high=1):
        super(FrequencyAwareAdaptiveReceptiveField, self).__init__()
        
        self.r_low = r_low
        self.r_mid = r_mid
        self.r_high = r_high
        
        # Frequency-specific weight predictors
        self.conv_low = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_mid = nn.Conv2d(in_channels, in_channels, 1)
        self.conv_high = nn.Conv2d(in_channels, in_channels, 1)
        
        # Dilation rate predictor
        self.dilation_predictor = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # Adaptive dilated convolutions for different dilation rates
        self.dilated_conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1)
        self.dilated_conv_2 = nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2)
        self.dilated_conv_3 = nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3)
        self.dilated_conv_4 = nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4)
        
        self.fusion = nn.Conv2d(in_channels, in_channels, 1)
        
    def frequency_decomposition(self, x):
        """Decompose input into low, mid, high frequency components"""
        B, C, H, W = x.shape
        
        # FFT
        x_fft = torch.fft.fft2(x, dim=(-2, -1))
        x_fft_shifted = torch.fft.fftshift(x_fft, dim=(-2, -1))
        
        # Create frequency masks
        center_h, center_w = H // 2, W // 2
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        y, x = y.to(x_fft.device).float(), x.to(x_fft.device).float()
        
        # Normalized distance from center
        dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
        max_dist = torch.sqrt(torch.tensor(center_h**2 + center_w**2, dtype=torch.float32))
        dist_norm = dist / max_dist
        
        # Frequency masks
        mask_low = (dist_norm <= 0.1).float().unsqueeze(0).unsqueeze(0)
        mask_mid = ((dist_norm > 0.1) & (dist_norm <= 0.25)).float().unsqueeze(0).unsqueeze(0)
        mask_high = ((dist_norm > 0.25) & (dist_norm <= 0.5)).float().unsqueeze(0).unsqueeze(0)
        
        # Apply masks and inverse FFT
        x_low = torch.fft.ifft2(torch.fft.ifftshift(x_fft_shifted * mask_low, dim=(-2, -1)), dim=(-2, -1)).real
        x_mid = torch.fft.ifft2(torch.fft.ifftshift(x_fft_shifted * mask_mid, dim=(-2, -1)), dim=(-2, -1)).real
        x_high = torch.fft.ifft2(torch.fft.ifftshift(x_fft_shifted * mask_high, dim=(-2, -1)), dim=(-2, -1)).real
        
        return x_low, x_mid, x_high
    
    def forward(self, x):
        # Frequency decomposition
        x_low, x_mid, x_high = self.frequency_decomposition(x)
        
        # Frequency-guided weighting
        weight_low = torch.sigmoid(self.conv_low(x))
        weight_mid = torch.sigmoid(self.conv_mid(x))
        weight_high = torch.sigmoid(self.conv_high(x))
        
        x_freq = weight_low * x_low + weight_mid * x_mid + weight_high * x_high
        
        # Predict dilation rates (1-4)
        dilation_map = self.dilation_predictor(x)  # [B, 1, H, W], values in [0, 1]
        dilation_map = 1 + dilation_map * 3  # Scale to [1, 4]
        
        # Apply multiple dilated convolutions
        feat_d1 = self.dilated_conv_1(x_freq)
        feat_d2 = self.dilated_conv_2(x_freq)
        feat_d3 = self.dilated_conv_3(x_freq)
        feat_d4 = self.dilated_conv_4(x_freq)
        
        # Adaptive fusion based on dilation map
        # Use soft selection
        weight_d1 = torch.relu(1 - torch.abs(dilation_map - 1))
        weight_d2 = torch.relu(1 - torch.abs(dilation_map - 2))
        weight_d3 = torch.relu(1 - torch.abs(dilation_map - 3))
        weight_d4 = torch.relu(1 - torch.abs(dilation_map - 4))
        
        # Normalize weights
        weight_sum = weight_d1 + weight_d2 + weight_d3 + weight_d4 + 1e-8
        weight_d1 = weight_d1 / weight_sum
        weight_d2 = weight_d2 / weight_sum
        weight_d3 = weight_d3 / weight_sum
        weight_d4 = weight_d4 / weight_sum
        
        # Weighted combination
        out = weight_d1 * feat_d1 + weight_d2 * feat_d2 + weight_d3 * feat_d3 + weight_d4 * feat_d4
        out = self.fusion(out)
        
        return out


class MVFE(nn.Module):
    """
    Multi-View Feature Enhancement Module
    Combines Spatial Attention and Frequency-aware Adaptive Receptive Field
    """
    def __init__(self, in_channels, num_groups=8, r_low=4, r_mid=2, r_high=1):
        super(MVFE, self).__init__()
        
        # Spatial Enhancement branch
        self.spatial_attention = SpatialAttention(in_channels, num_groups)
        
        # Frequency-aware Adaptive Receptive Field branch
        self.farm = FrequencyAwareAdaptiveReceptiveField(in_channels, r_low, r_mid, r_high)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Spatial branch
        x_spatial = self.spatial_attention(x)
        
        # Frequency-aware branch
        x_freq = self.farm(x)
        
        # Fusion
        out = torch.cat([x_spatial, x_freq], dim=1)
        out = self.fusion(out)
        
        # Residual connection
        out = out + x
        
        return out


if __name__ == "__main__":
    # Test MVFE module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x = torch.randn(2, 64, 32, 32).to(device)
    mvfe = MVFE(in_channels=64, num_groups=8).to(device)
    
    out = mvfe(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("MVFE module test passed!")
