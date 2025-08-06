from models.bsmamba.mamba_utils import *

class BrightnessMamba(nn.Module):
    def __init__(self, dim, d_state, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

    def forward(self, x, x_size, light_score):
        B, n, C = x.shape
        H, W = x_size

        attention_score = light_score.view(B, -1)  # (B, HW)
        # Sort pixels by brightness score
        x_sort_values, x_sort_indices = torch.sort(attention_score, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        # Project input features
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        # Semantic-guided neighbor unfolding
        semantic_x = semantic_neighbor(x, x_sort_indices)  # SGN-unfold
        y = self.selectiveScan(semantic_x)  # No prompt, use brightness score
        y = self.out_proj(self.out_norm(y))

        # Semantic-guided neighbor folding
        x = semantic_neighbor(y, x_sort_indices_reverse)  # SGN-fold

        return x



