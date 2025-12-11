import torch
import torch.nn as nn
import torch.nn.functional as F

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: t.reshape(b, n, h, -1).permute(0, 2, 1, 3), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(b, n, -1)
        return self.to_out(out)

class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

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

class BackgammonValueNetGen5(nn.Module):
    """
    Gen 5 Transformer-Lite Network.
    Input: 200 floats (Board + Global Features)
           - 192 Board Features (8 per point * 24 points)
           - 6 Global Features (Turn, Cube, etc) from Gen 4
           - 2 NEW Global Features (MyScore/MatchLen, OppScore/MatchLen)
    """
    def __init__(self):
        super(BackgammonValueNetGen5, self).__init__()
        
        # --- SPATIAL BACKBONE (ResNet) ---
        self.board_net = nn.Sequential(
            nn.Conv1d(8, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            ResNetBlock(64, 128),
            ResNetBlock(128, 256), # Widened
        )
        
        # --- TEMPORAL/GLOBAL CONTEXT (Transformer) ---
        # Input to Transformer: Sequence of 24 points (Dim 256)
        self.transformer = TransformerBlock(dim=256, depth=2, heads=4, dim_head=64, mlp_dim=512)
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # --- FUSION & HEADS ---
        # Flattened Board Dim: 256
        # Global Features: 8
        self.shared_fc = nn.Sequential(
            nn.Linear(256 + 8, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU()
        )
        
        # Head 1: Value (6 Classes)
        self.value_head = nn.Linear(256, 6)
        
        # Head 2: Pip Count (2 Scalars)
        self.pip_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x shape: (Batch, 200)
        bs = x.shape[0]
        
        # Split Board (192) and Global (8)
        board_features = x[:, :192]
        global_features = x[:, 192:]
        
        # Reshape board to (Batch, 8, 24) for CNN
        # Note: Original input is flat list of points 0..23, with 8 features each.
        # Reshape to (Batch, 24, 8) then Permute to (Batch, 8, 24)
        board_spatial = board_features.view(bs, 24, 8).permute(0, 2, 1)
        
        # 1. CNN Backbone
        # Input: (Batch, 8, 24) -> Output: (Batch, 256, 24)
        feat_map = self.board_net(board_spatial)
        
        # 2. Transformer Feature mixing
        # Transformer expects (Batch, SeqLen, Dim) -> (Batch, 24, 256)
        feat_seq = feat_map.permute(0, 2, 1)
        feat_trans = self.transformer(feat_seq)
        
        # 3. Pooling
        # (Batch, 24, 256) -> (Batch, 256, 1) -> (Batch, 256)
        feat_pooled = feat_trans.permute(0, 2, 1).mean(dim=2) 
        
        # 4. Fusion
        combined = torch.cat([feat_pooled, global_features], dim=1)
        latent = self.shared_fc(combined)
        
        # 5. Heads
        value_logits = self.value_head(latent)
        pip_preds = self.pip_head(latent)
        
        return value_logits, pip_preds
