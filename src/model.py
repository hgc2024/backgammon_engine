import torch
import torch.nn as nn

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

class BackgammonValueNet(nn.Module):
    """
    Afterstate Value Network.
    Input: 198 floats (Board State)
    Output: 1 float (Scalar Value V(s) -> Probability of Winning)
    Activation: Tanh (-1 to 1) or Sigmoid (0 to 1). 
    Backgammon usually uses Sigmoid (Win Prob) or Linear (-1 to 1).
    Let's use Tanh for Reward symmetricity (-1 Loss, +1 Win).
    """
    def __init__(self):
        super(BackgammonValueNet, self).__init__()
        
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
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh() # Output -1 to 1
        )

    def forward(self, x):
        # x shape: (Batch, 198)
        bs = x.shape[0]
        board_features = x[:, :192]
        global_features = x[:, 192:]
        
        # Reshape board features to (Batch, 8, 24)
        # See src/features.py for logic
        # 0..95 is MyBoard, 96..191 is OppBoard
        my_board = board_features[:, :96].view(bs, 24, 4)
        opp_board = board_features[:, 96:].view(bs, 24, 4)
        
        combined_board = torch.cat([my_board, opp_board], dim=2) # (Batch, 24, 8)
        combined_board = combined_board.permute(0, 2, 1) # (Batch, 8, 24)
        
        # CNN
        board_out = self.board_net(combined_board)
        
        # Concat Global
        combined = torch.cat([board_out, global_features], dim=1)
        
        return self.final_net(combined)
