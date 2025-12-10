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
    Gen 4 Multi-Head Network.
    Input: 198 floats (Board State)
    
    Heads:
    1. Value Output: 6 logits (LossBG, LossG, Loss, Win, WinG, WinBG)
       - Used with CrossEntropyLoss
    2. Pip Output: 2 scalars (MyPip, OppPip)
       - Used with MSELoss (Likely normalized / 100.0)
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
        
        # Shared Dense Layer
        # Concatenated size: 128 + 6 = 134
        self.shared_fc = nn.Sequential(
            nn.Linear(128 + 6, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Head 1: Value (6 Classes)
        self.value_head = nn.Linear(128, 6)
        
        # Head 2: Pip Count (2 Scalars)
        self.pip_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # [MyPip, OppPip]
        )

    def forward(self, x):
        # x shape: (Batch, 198)
        bs = x.shape[0]
        board_features = x[:, :192]
        global_features = x[:, 192:]
        
        # Reshape board features to (Batch, 8, 24)
        my_board = board_features[:, :96].view(bs, 24, 4)
        opp_board = board_features[:, 96:].view(bs, 24, 4)
        
        combined_board = torch.cat([my_board, opp_board], dim=2) # (Batch, 24, 8)
        combined_board = combined_board.permute(0, 2, 1) # (Batch, 8, 24)
        
        # CNN
        board_out = self.board_net(combined_board)
        
        # Concat Global
        combined = torch.cat([board_out, global_features], dim=1)
        
        # Shared Representation
        latent = self.shared_fc(combined)
        
        # Heads
        value_logits = self.value_head(latent) # (Batch, 6)
        pip_preds = self.pip_head(latent)      # (Batch, 2)
        
        # Return both
        return value_logits, pip_preds
