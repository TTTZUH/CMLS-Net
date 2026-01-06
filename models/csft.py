"""
Collaborative Semantic Fusion Transformer (CSFT) Module
Establishes deep associations between heterogeneous features through 
three-branch parallel architecture and cross-branch key-value interaction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiScaleConv(nn.Module):
    """Multi-scale convolutions for multi-scale feature branch"""
    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels // 3, 1)
        self.conv3 = nn.Conv2d(in_channels, out_channels // 3, 3, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels - 2 * (out_channels // 3), 5, padding=2)
        
    def forward(self, x):
        feat1 = self.conv1(x)
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)
        return torch.cat([feat1, feat3, feat5], dim=1)


class CrossAttention(nn.Module):
    """Cross-branch attention with Query from one branch, Key-Value from others"""
    def __init__(self, dim, num_heads=4, window_size=4):
        super(CrossAttention, self).__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
    def forward(self, q, k, v):
        """
        Args:
            q: Query tensor [B, C, H, W]
            k: Key tensor [B, C, H, W]
            v: Value tensor [B, C, H, W]
        """
        B, C, H, W = q.shape
        
        # Reshape for multi-head attention
        q = q.flatten(2).transpose(1, 2)  # [B, HW, C]
        k = k.flatten(2).transpose(1, 2)  # [B, HW, C]
        v = v.flatten(2).transpose(1, 2)  # [B, HW, C]
        
        # Split heads
        q = q.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # [B, heads, HW, head_dim]
        k = k.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, HW, HW]
        attn = torch.softmax(attn, dim=-1)
        
        # Gating mechanism
        attn_max = attn.max(dim=-1, keepdim=True)[0]
        gate = torch.sigmoid(attn_max)
        attn = attn * gate
        
        # Apply attention to values
        out = attn @ v  # [B, heads, HW, head_dim]
        
        # Merge heads
        out = out.transpose(1, 2).reshape(B, H*W, C)  # [B, HW, C]
        out = out.transpose(1, 2).reshape(B, C, H, W)  # [B, C, H, W]
        
        return out


class SpatialFeatureBranch(nn.Module):
    """Spatial feature processing branch with q_s, k_g, v_m"""
    def __init__(self, dim, num_heads=4, window_size=4):
        super(SpatialFeatureBranch, self).__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.q_linear = nn.Linear(dim, dim)
        self.cross_attn = CrossAttention(dim, num_heads, window_size)
        
        # Post-attention processing
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1)
        )
        
    def forward(self, f_s, k_g, v_m):
        """
        Args:
            f_s: Spatial features [B, C, H, W]
            k_g: Keys from global branch [B, C, H, W]
            v_m: Values from multi-scale branch [B, C, H, W]
        """
        B, C, H, W = f_s.shape
        
        # Layer norm
        f_s_norm = f_s.flatten(2).transpose(1, 2)  # [B, HW, C]
        f_s_norm = self.norm(f_s_norm).transpose(1, 2).reshape(B, C, H, W)
        
        # Generate query
        q_s = f_s_norm.flatten(2).transpose(1, 2)  # [B, HW, C]
        q_s = self.q_linear(q_s).transpose(1, 2).reshape(B, C, H, W)
        
        # Cross attention
        attn_out = self.cross_attn(q_s, k_g, v_m)
        
        # Pooling aggregation
        avg_pool = F.adaptive_avg_pool2d(attn_out, 1)
        max_pool = F.adaptive_max_pool2d(attn_out, 1)
        pool_out = (avg_pool + max_pool)
        pool_out_norm = pool_out.flatten(2).transpose(1, 2)
        pool_out_norm = self.norm(pool_out_norm).transpose(1, 2).reshape(B, C, 1, 1)
        
        out = pool_out_norm * attn_out
        
        # MLP
        out = out + self.mlp(out)
        
        return out


class GlobalFeatureBranch(nn.Module):
    """Global feature processing branch with q_g, k_m, v_s"""
    def __init__(self, dim, num_heads=4, window_size=4):
        super(GlobalFeatureBranch, self).__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.q_linear = nn.Linear(dim, dim)
        self.cross_attn = CrossAttention(dim, num_heads, window_size)
        
        # Self-attention enhancement
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
    def forward(self, f_g, k_m, v_s):
        """
        Args:
            f_g: Global features [B, C, H, W]
            k_m: Keys from multi-scale branch [B, C, H, W]
            v_s: Values from spatial branch [B, C, H, W]
        """
        B, C, H, W = f_g.shape
        
        # Layer norm
        f_g_norm = f_g.flatten(2).transpose(1, 2)  # [B, HW, C]
        f_g_norm = self.norm(f_g_norm).transpose(1, 2).reshape(B, C, H, W)
        
        # Generate query
        q_g = f_g_norm.flatten(2).transpose(1, 2)  # [B, HW, C]
        q_g = self.q_linear(q_g).transpose(1, 2).reshape(B, C, H, W)
        
        # Cross attention
        attn_out = self.cross_attn(q_g, k_m, v_s)
        
        # Self-attention for global context
        attn_flat = attn_out.flatten(2).transpose(1, 2)  # [B, HW, C]
        self_attn_out, _ = self.self_attn(attn_flat, attn_flat, attn_flat)
        out = self_attn_out.transpose(1, 2).reshape(B, C, H, W)
        
        return out


class MultiScaleFeatureBranch(nn.Module):
    """Multi-scale feature processing branch with q_m, k_s, v_g"""
    def __init__(self, dim, num_heads=4, window_size=4):
        super(MultiScaleFeatureBranch, self).__init__()
        
        self.norm = nn.LayerNorm(dim)
        self.q_multiscale = MultiScaleConv(dim, dim)
        self.cross_attn = CrossAttention(dim, num_heads, window_size)
        
        # Post-processing
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, f_m, k_s, v_g):
        """
        Args:
            f_m: Multi-scale features [B, C, H, W]
            k_s: Keys from spatial branch [B, C, H, W]
            v_g: Values from global branch [B, C, H, W]
        """
        B, C, H, W = f_m.shape
        
        # Layer norm
        f_m_norm = f_m.flatten(2).transpose(1, 2)  # [B, HW, C]
        f_m_norm = self.norm(f_m_norm).transpose(1, 2).reshape(B, C, H, W)
        
        # Generate multi-scale query
        q_m = self.q_multiscale(f_m_norm)
        
        # Cross attention
        attn_out = self.cross_attn(q_m, k_s, v_g)
        
        # Normalization and convolution
        attn_norm = attn_out.flatten(2).transpose(1, 2)
        attn_norm = self.norm(attn_norm).transpose(1, 2).reshape(B, C, H, W)
        out = self.conv(attn_norm)
        
        return out


class CSFT(nn.Module):
    """
    Collaborative Semantic Fusion Transformer
    Three-branch architecture with cross-branch key-value interactions
    """
    def __init__(self, dim, num_heads=4, window_size=4):
        super(CSFT, self).__init__()
        
        # Three parallel branches
        self.spatial_branch = SpatialFeatureBranch(dim, num_heads, window_size)
        self.global_branch = GlobalFeatureBranch(dim, num_heads, window_size)
        self.multiscale_branch = MultiScaleFeatureBranch(dim, num_heads, window_size)
        
        # Key and Value generators for cross-branch interaction
        self.k_spatial = nn.Conv2d(dim, dim, 1)
        self.k_global = nn.Conv2d(dim, dim, 1)
        self.k_multiscale = MultiScaleConv(dim, dim)
        
        self.v_spatial = nn.Conv2d(dim, dim, 1)
        self.v_global = nn.Conv2d(dim, dim, 1)
        self.v_multiscale = nn.Conv2d(dim, dim, 1)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )
        
        # MLP for final refinement
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, 1)
        )
        
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, f_s, f_g, f_m):
        """
        Args:
            f_s: Spatial features [B, C, H, W]
            f_g: Global features [B, C, H, W]
            f_m: Multi-scale features [B, C, H, W]
        """
        B, C, H, W = f_s.shape
        
        # Generate Keys and Values for each branch
        k_s = self.k_spatial(f_s)
        k_g = self.k_global(f_g)
        k_m = self.k_multiscale(f_m)
        
        v_s = self.v_spatial(f_s)
        v_g = self.v_global(f_g)
        v_m = self.v_multiscale(f_m)
        
        # Three-branch cross attention
        # Spatial branch: q_s, k_g, v_m
        out_s = self.spatial_branch(f_s, k_g, v_m)
        
        # Global branch: q_g, k_m, v_s
        out_g = self.global_branch(f_g, k_m, v_s)
        
        # Multi-scale branch: q_m, k_s, v_g
        out_m = self.multiscale_branch(f_m, k_s, v_g)
        
        # Concatenate and fuse
        out = torch.cat([out_s, out_g, out_m], dim=1)
        out_fused = self.fusion(out)
        
        # MLP with residual
        out_flat = out_fused.flatten(2).transpose(1, 2)
        out_norm = self.norm(out_flat).transpose(1, 2).reshape(B, C, H, W)
        out_final = out_fused + self.mlp(out_norm)
        
        return out_final


if __name__ == "__main__":
    # Test CSFT module
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Three types of features
    f_s = torch.randn(2, 128, 32, 32).to(device)  # Spatial features
    f_g = torch.randn(2, 128, 32, 32).to(device)  # Global features
    f_m = torch.randn(2, 128, 32, 32).to(device)  # Multi-scale features
    
    csft = CSFT(dim=128, num_heads=4, window_size=4).to(device)
    
    out = csft(f_s, f_g, f_m)
    print(f"Input shapes: f_s={f_s.shape}, f_g={f_g.shape}, f_m={f_m.shape}")
    print(f"Output shape: {out.shape}")
    print("CSFT module test passed!")
