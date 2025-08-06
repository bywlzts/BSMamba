from models.bsmamba.mamba_utils import *

class SemanticMamba(nn.Module):
    def __init__(self, dim, d_state, mlp_ratio=2.0):
        """
        ASSM structure with external attention map.
        Args:
            dim: Input and output feature dimension.
            d_state: State dimension for Selective_Scan.
            mlp_ratio: Expansion ratio for the MLP.
        """
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.mlp_ratio = mlp_ratio

        # Selective Scan module
        hidden = int(self.dim * self.mlp_ratio)
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        # Input projection
        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        # Convolutional Positional Encoding (CPE)
        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        # Variables to store scan order
        self.x_sort_indices = None
        self.x_sort_indices_reverse = None

    def forward(self, x, x_size, semantic_score):
        B, n, C = x.shape
        H, W = x_size


        # Flatten attention map and calculate sorting indices
        attention_score = semantic_score.view(B, -1)  # (B, HW)
        _, self.x_sort_indices = torch.sort(attention_score, dim=-1, stable=False)
        self.x_sort_indices_reverse = index_reverse(self.x_sort_indices)

            # Project input features
        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # (B, HW, C)

        # Semantic-guided neighbor unfolding
        semantic_x = semantic_neighbor(x, self.x_sort_indices)  # SGN-unfold
        y = self.selectiveScan(semantic_x)  # No prompt, use attention score
        y = self.out_proj(self.out_norm(y))

        # Semantic-guided neighbor folding
        x = semantic_neighbor(y, self.x_sort_indices_reverse)  # SGN-fold

        return x
