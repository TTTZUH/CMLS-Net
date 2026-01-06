"""
HRNet Backbone for CMLS-Net
Simplified implementation based on HRNet architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Basic residual block for HRNet"""
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out


class HighResolutionModule(nn.Module):
    """High Resolution Module with multi-scale parallel branches"""
    def __init__(self, num_branches, num_blocks, in_channels, num_channels):
        super(HighResolutionModule, self).__init__()
        
        self.num_branches = num_branches
        
        # Create branches
        self.branches = nn.ModuleList([
            self._make_branch(num_blocks[i], in_channels[i], num_channels[i])
            for i in range(num_branches)
        ])
        
        # Fusion layers
        self.fuse_layers = self._make_fuse_layers(num_channels)
        
    def _make_branch(self, num_blocks, in_channels, out_channels):
        """Create one branch with multiple blocks"""
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(
                in_channels if i == 0 else out_channels,
                out_channels
            ))
        return nn.Sequential(*layers)
    
    def _make_fuse_layers(self, num_channels):
        """Create fusion layers for multi-scale feature exchange"""
        fuse_layers = nn.ModuleList()
        
        for i in range(self.num_branches):
            fuse_layer = nn.ModuleList()
            for j in range(self.num_branches):
                if j > i:
                    # Upsample
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_channels[j], num_channels[i], 1, bias=False),
                        nn.BatchNorm2d(num_channels[i]),
                        nn.Upsample(scale_factor=2**(j-i), mode='bilinear', align_corners=True)
                    ))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    # Downsample
                    conv_layers = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_layers.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[i], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[i])
                            ))
                        else:
                            conv_layers.append(nn.Sequential(
                                nn.Conv2d(num_channels[j], num_channels[j], 3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_channels[j]),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv_layers))
            
            fuse_layers.append(fuse_layer)
        
        return fuse_layers
    
    def forward(self, x):
        """
        Args:
            x: List of features from different branches
        """
        # Process each branch
        branch_outputs = [
            self.branches[i](x[i]) for i in range(self.num_branches)
        ]
        
        # Fuse features
        fused_outputs = []
        for i in range(self.num_branches):
            fused = None
            for j in range(self.num_branches):
                if i == j:
                    fused = branch_outputs[j] if fused is None else fused + branch_outputs[j]
                else:
                    if fused is None:
                        fused = self.fuse_layers[i][j](branch_outputs[j])
                    else:
                        fused = fused + self.fuse_layers[i][j](branch_outputs[j])
            
            fused_outputs.append(F.relu(fused))
        
        return fused_outputs


class HRNet(nn.Module):
    """
    HRNet Backbone
    Multi-resolution parallel architecture
    """
    def __init__(self, config='W40'):
        super(HRNet, self).__init__()
        
        # Configuration
        if config == 'W40':
            self.stage_channels = [[40], [40, 80], [40, 80, 160], [40, 80, 160, 320]]
            self.num_modules = [1, 1, 4, 3]
            self.num_blocks = [[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]]
        elif config == 'W32':
            self.stage_channels = [[32], [32, 64], [32, 64, 128], [32, 64, 128, 256]]
            self.num_modules = [1, 1, 4, 3]
            self.num_blocks = [[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]]
        else:
            raise ValueError(f"Unknown config: {config}")
        
        # Stem
        self.conv1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        
        # Stage 1
        self.layer1 = self._make_layer(BasicBlock, 64, 64, 4)
        
        # Transition to multi-branch
        self.transition1 = self._make_transition_layer([256], self.stage_channels[1])
        self.stage2 = self._make_stage(self.stage_channels[1], self.num_modules[1], 
                                       self.num_blocks[1])
        
        self.transition2 = self._make_transition_layer(self.stage_channels[1], 
                                                        self.stage_channels[2])
        self.stage3 = self._make_stage(self.stage_channels[2], self.num_modules[2], 
                                       self.num_blocks[2])
        
        self.transition3 = self._make_transition_layer(self.stage_channels[2], 
                                                        self.stage_channels[3])
        self.stage4 = self._make_stage(self.stage_channels[3], self.num_modules[3], 
                                       self.num_blocks[3])
        
    def _make_layer(self, block, in_channels, out_channels, blocks):
        """Make a layer with multiple blocks"""
        layers = []
        layers.append(block(in_channels, out_channels))
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def _make_transition_layer(self, num_channels_pre, num_channels_cur):
        """Make transition layer to change number of branches"""
        num_branches_cur = len(num_channels_cur)
        num_branches_pre = len(num_channels_pre)
        
        transition_layers = nn.ModuleList()
        
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur[i] != num_channels_pre[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre[i], num_channels_cur[i], 3, 1, 1, bias=False),
                        nn.BatchNorm2d(num_channels_cur[i]),
                        nn.ReLU(inplace=True)
                    ))
                else:
                    transition_layers.append(None)
            else:
                conv_layers = []
                for j in range(i + 1 - num_branches_pre):
                    in_ch = num_channels_pre[-1]
                    out_ch = num_channels_cur[i] if j == i - num_branches_pre else in_ch
                    conv_layers.append(nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False),
                        nn.BatchNorm2d(out_ch),
                        nn.ReLU(inplace=True)
                    ))
                transition_layers.append(nn.Sequential(*conv_layers))
        
        return transition_layers
    
    def _make_stage(self, num_channels, num_modules, num_blocks):
        """Make a stage with multiple HR modules"""
        modules = []
        for i in range(num_modules):
            modules.append(HighResolutionModule(
                len(num_channels),
                num_blocks,
                num_channels,
                num_channels
            ))
        return nn.Sequential(*modules)
    
    def forward(self, x):
        """
        Args:
            x: Input image [B, 3, H, W]
        Returns:
            Multi-scale features [f1, f2, f3, f4] from high to low resolution
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Stage 1
        x = self.layer1(x)
        
        # Stage 2
        x_list = []
        for i in range(len(self.transition1)):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)
        
        # Stage 3
        x_list = []
        for i in range(len(self.transition2)):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)
        
        # Stage 4
        x_list = []
        for i in range(len(self.transition3)):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        
        return y_list


if __name__ == "__main__":
    # Test HRNet
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    x = torch.randn(2, 3, 256, 256).to(device)
    model = HRNet(config='W40').to(device)
    
    features = model(x)
    print(f"Input shape: {x.shape}")
    for i, feat in enumerate(features):
        print(f"Feature {i+1} shape: {feat.shape}")
    print("HRNet test passed!")
