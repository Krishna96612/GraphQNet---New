import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.utils import dense_to_sparse


# ---------------------------------------------------------
# 1. Core GAT Layer
# ---------------------------------------------------------

class MultiHopGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=8, dropout=0.1):
        super().__init__()

        self.conv = GATConv(
            in_channels,
            out_channels // num_heads,
            heads=num_heads,
            dropout=dropout,
            add_self_loops=True
        )

        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x_gnn, edge_index):
        x = self.conv(x_gnn, edge_index)
        x = self.bn(x)
        return self.act(x)


# ---------------------------------------------------------
# 2. Channel Attention (SE)
# ---------------------------------------------------------

class SEBlock(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(c, c // r, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // r, c, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.pool(x)
        w = self.fc(w)
        return x * w


# ---------------------------------------------------------
# 3. Multi-Hop GNN Block (Improved)
# ---------------------------------------------------------

class MultiHopGNNBlock(nn.Module):

    def __init__(self, c1, c2, num_hops=2):
        super().__init__()

        self.cv_identity = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()

        self.conv_in = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
        self.conv_out = nn.Conv2d(c2, c2, 1, 1, 0, bias=False)

        self.gnn_layers = nn.ModuleList([
            MultiHopGATLayer(c2, c2)
            for _ in range(num_hops)
        ])

        self.se = SEBlock(c2)

        self.register_buffer('edge_index', None, persistent=True)
        self.register_buffer('last_HW', torch.tensor((-1, -1), dtype=torch.long), persistent=False)

    def _prepare_graph(self, H, W, device):

        if self.last_HW[0].item() == H and self.last_HW[1].item() == W:
            return

        N = H * W
        adj_dense = torch.zeros(N, N, dtype=torch.bool, device=device)

        for i in range(H):
            for j in range(W):

                node = i * W + j

                for ni in range(max(0, i-1), min(H, i+2)):
                    for nj in range(max(0, j-1), min(W, j+2)):
                        adj_dense[node, ni * W + nj] = True

                # long-range stride-2 connections
                if i+2 < H:
                    adj_dense[node, (i+2) * W + j] = True
                if j+2 < W:
                    adj_dense[node, i * W + (j+2)] = True

        edge_index, _ = dense_to_sparse(adj_dense)

        self.edge_index = edge_index
        self.last_HW = torch.tensor((H, W), dtype=torch.long, device=device)

    def forward(self, x):

        identity = self.cv_identity(x)

        x = self.conv_in(x)

        B, C, H, W = x.shape
        device = x.device

        x_flat = x.permute(0,2,3,1).reshape(B*H*W, C)

        self._prepare_graph(H, W, device)

        if B == 1:
            edge_index_batched = self.edge_index
        else:

            node_offset = torch.arange(B, device=device) * (H*W)
            node_offset = node_offset.view(-1,1,1)

            edge_index_unbatched = self.edge_index.unsqueeze(0).repeat(B,1,1)
            edge_index_batched = (edge_index_unbatched + node_offset).view(2,-1)

        for layer in self.gnn_layers:

            residual = x_flat
            x_flat = layer(x_flat, edge_index_batched)

            # hop residual
            x_flat = x_flat + residual

        x = x_flat.reshape(B,H,W,C).permute(0,3,1,2)

        x = self.conv_out(x)

        x = self.se(x)

        return x + identity


# ---------------------------------------------------------
# 4. Quantum Utilities (Improved Encoding)
# ---------------------------------------------------------

def quantum_encode(x):
    xr = torch.cos(x)
    xi = torch.sin(x)
    return xr, xi


def modReLU(xr, xi, q_bias):

    mag = torch.sqrt(xr**2 + xi**2 + 1e-8)

    scale = F.relu(mag + q_bias) / (mag + 1e-8)

    return xr * scale, xi * scale


class ComplexBatchNorm2d(nn.Module):

    def __init__(self, num_features):
        super().__init__()

        self.bn_real = nn.BatchNorm2d(num_features)
        self.bn_imag = nn.BatchNorm2d(num_features)

    def forward(self, xr, xi):
        return self.bn_real(xr), self.bn_imag(xi)


class ComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()

        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)

    def forward(self, xr, xi):

        real = self.conv_r(xr) - self.conv_i(xi)
        imag = self.conv_r(xi) + self.conv_i(xr)

        return real, imag


# ---------------------------------------------------------
# 5. Quantum Inspired Block (Improved)
# ---------------------------------------------------------

class QuantumInspiredBlock(nn.Module):

    def __init__(self, c1, c2, kernel_size=5):
        super().__init__()

        self.cv_identity = nn.Conv2d(c1, c2, 1, 1, 0, bias=False) if c1 != c2 else nn.Identity()

        self.complex_conv = ComplexConv2d(
            c1,
            c2,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.bn = ComplexBatchNorm2d(c2)

        self.q_bias = nn.Parameter(torch.zeros(1, c2, 1, 1))

        self.projection = nn.Conv2d(c2 * 2, c2, 1)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):

        identity = self.cv_identity(x)

        xr, xi = quantum_encode(x)

        xr, xi = self.complex_conv(xr, xi)

        xr, xi = self.bn(xr, xi)

        xr, xi = modReLU(xr, xi, self.q_bias)

        x_complex = torch.cat((xr, xi), dim=1)

        x_out = self.projection(x_complex)

        x_out = self.dropout(x_out)

        return x_out + identity