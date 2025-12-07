import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class BackgammonResNet(BaseFeaturesExtractor):
    """
    Custom Feature Extractor for Backgammon.
    Input: 198 floats
    Structure:
    - 192 floats -> Reshaped to (Batch, 8, 24) representing (8 features per point, 24 points)
    - 6 floats -> Global features (Bar, Off, Turn, Cube)
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(BackgammonResNet, self).__init__(observation_space, features_dim)
        
        # Board processing (Spatial)
        # Input channels: 8 (4 for p0, 4 for p1 per point encoding)
        self.board_net = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResNetBlock(64, 64),
            ResNetBlock(64, 128),
            ResNetBlock(128, 128),
            nn.AdaptiveAvgPool1d(1), # Squeeze spatial dim -> (Batch, 128, 1)
            nn.Flatten() # (Batch, 128)
        )
        
        # Global processing
        # 6 global features
        # Concatenated size: 128 + 6 = 134
        self.final_net = nn.Sequential(
            nn.Linear(128 + 6, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Split input
        # First 192 are board (24 points * 4 encodings * 2 players)
        # However, our env encodes: [MyBoard (24*4), OppBoard (24*4), ...]
        # MyBoard: 0..95. OppBoard: 96..191.
        # We want to stack them per point.
        # current format: [MyPt0_f0, MyPt0_f1... MyPt0_f3, MyPt1_f0... ]
        # Length 192.
        
        bs = observations.shape[0]
        board_features = observations[:, :192]
        global_features = observations[:, 192:]
        
        # Reshape to (Batch, 24, 8) or (Batch, 8, 24)?
        # Torch Conv1d expects (Batch, Channels, Length). Length=24.
        # We need to interleave MyBoard and OppBoard per point?
        # MyBoard: (Batch, 96). -> Reshape (Batch, 24, 4)?
        # OppBoard: (Batch, 96). -> Reshape (Batch, 24, 4)?
        # Stack -> (Batch, 24, 8). Then permute to (Batch, 8, 24).
        
        my_board = board_features[:, :96].view(bs, 24, 4)
        opp_board = board_features[:, 96:].view(bs, 24, 4)
        
        # Concatenate along channel dimension (last dim currently)
        combined_board = torch.cat([my_board, opp_board], dim=2) # (Batch, 24, 8)
        
        # Permute for Conv1d: (Batch, Channels, Length) -> (Batch, 8, 24)
        combined_board = combined_board.permute(0, 2, 1) 
        
        # Pass through ResNet
        board_out = self.board_net(combined_board)
        
        # Concatenate Global
        combined = torch.cat([board_out, global_features], dim=1)
        
        return self.final_net(combined)
