import torch
import torch.nn as nn
import torch.nn.functional as F

class CrissCrossAttention(nn.Module):
    def __init__(self, in_ch, inter_ch):
        super().__init__()

        self.query = nn.Conv2d(in_ch, inter_ch, 1)
        self.key   = nn.Conv2d(in_ch, inter_ch, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.permute(0, 2, 3, 1)  
        K = K.permute(0, 2, 3, 1)
        V = V.permute(0, 2, 3, 1)

        out = torch.zeros_like(V)

        for i in range(H):
            for j in range(W):

                q = Q[:, i, j, :] 

                k_row = K[:, i, :, :]  
                k_col = K[:, :, j, :]   

                k = torch.cat([k_row, k_col], dim=1)  

                energy = torch.bmm(k, q.unsqueeze(-1)).squeeze(-1)

                attn = F.softmax(energy, dim=1)

                v_row = V[:, i, :, :]
                v_col = V[:, :, j, :]
                v = torch.cat([v_row, v_col], dim=1)

                out[:, i, j, :] = torch.sum(v * attn.unsqueeze(-1), dim=1)

        out = out.permute(0, 3, 1, 2)

        return x + self.gamma * out
