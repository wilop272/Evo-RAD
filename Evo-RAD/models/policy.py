"""
Policy Network with Query as Node-0 in GCN Graph

GCN receives [B, K+1, D] input where node 0 = query image.
Delete actions only target candidate nodes 1..K.
Action space: 0=Stop, 1..K=Delete candidate i-1, K+1=Insert.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x, adj):
        # adj: [B, N, N] (text similarity or identity)
        B, N, _ = x.shape
        # Normalized adjacency with self-loops
        I = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        A_hat = adj + I
        D = A_hat.sum(dim=2).clamp(min=1e-6)
        D_inv_sqrt = torch.pow(D, -0.5).diag_embed()
        A_norm = torch.bmm(torch.bmm(D_inv_sqrt, A_hat), D_inv_sqrt)
        return torch.bmm(A_norm, self.linear(x)) + self.bias


class PolicyNetwork(nn.Module):
    """
    GCN-based policy that operates on a graph with K+1 nodes:
      node 0  = query image (visual features only, zero text)
      nodes 1..K = candidate images

    Actions:
      0       = Stop
      1 .. K  = Delete candidate at slot (action - 1)  [nodes 1..K]
      K + 1   = Insert best candidate from pool
    """

    def __init__(self, input_dim, hidden_dim=128, stats_dim=8, device='cuda', ablation_mode='full'):
        super().__init__()
        self.device = device
        self.ablation_mode = ablation_mode
        self.stats_dim = stats_dim

        # Determine actual stats dim from ablation mode
        if self.ablation_mode == 'no_stats':
            actual_stats_dim = 0
        elif self.ablation_mode in ('no_dev', 'no_ego'):
            actual_stats_dim = 4
        else:  # 'full'
            actual_stats_dim = stats_dim

        # GCN takes [vis_features ∥ stats]
        self.gcn1 = SimpleGCNLayer(input_dim + actual_stats_dim, hidden_dim)
        self.gcn2 = SimpleGCNLayer(hidden_dim, hidden_dim)

        # Action heads
        self.active_mlp = nn.Linear(hidden_dim, 1)      # Per-candidate delete score
        self.stop_mlp = nn.Linear(hidden_dim, 1)         # Global stop score
        self.new_insert_mlp = nn.Linear(hidden_dim, 1)   # Global insert score

    def forward(self, active_features, active_stats, active_mask, text_features=None, **kwargs):
        """
        Args:
            active_features: [B, K+1, D_vis]   (node 0 = query, nodes 1..K = candidates)
            active_stats:    [B, K+1, 8]
            active_mask:     [B, K]            (candidate-only mask, does NOT include query)
            text_features:   [B, K+1, D_txt]   (node 0 text = zero vector)
        """
        B, N_all, _ = active_features.shape  # N_all = K + 1
        K = N_all - 1  # number of candidate slots

        # 1. Dynamic Graph Construction from text similarity [B, K+1, K+1]
        if text_features is not None:
            if torch.isnan(text_features).any() or (text_features.abs().sum(dim=-1) == 0).all():
                adj = torch.eye(N_all, device=self.device).unsqueeze(0).expand(B, -1, -1)
            else:
                norm_text = F.normalize(text_features, p=2, dim=-1)
                adj = torch.bmm(norm_text, norm_text.transpose(1, 2))
        else:
            adj = torch.eye(N_all, device=self.device).unsqueeze(0).expand(B, -1, -1)

        # 2. Feature fusion (vis + stats) with ablation support
        if self.ablation_mode == 'no_stats':
            x = active_features
        elif self.ablation_mode == 'no_dev':
            x = torch.cat([active_features, active_stats[:, :, :4]], dim=-1)
        elif self.ablation_mode == 'no_ego':
            x = torch.cat([active_features, active_stats[:, :, 4:]], dim=-1)
        else:
            x = torch.cat([active_features, active_stats], dim=-1)

        # 3. GCN message passing on K+1 nodes
        x = F.relu(self.gcn1(x, adj))   # [B, K+1, H]
        x = self.gcn2(x, adj)            # [B, K+1, H]

        # 4. Split: query node (0) vs candidate nodes (1..K)
        query_repr = x[:, 0:1, :]   # [B, 1, H]  — query node embedding
        cand_repr  = x[:, 1:, :]    # [B, K, H]  — candidate node embeddings

        # 5. Action logits
        # Delete logits: per-candidate scores (nodes 1..K only)
        del_logits = self.active_mlp(cand_repr).squeeze(-1)  # [B, K]

        # Stop logit: pool over ALL nodes (query included aids global view)
        stop_pool = x.max(dim=1)[0]                          # [B, H]
        stop_logit = self.stop_mlp(stop_pool)                # [B, 1]

        # Insert logit: pool over ALL nodes
        insert_pool = x.mean(dim=1)                          # [B, H]
        insert_logit = self.new_insert_mlp(insert_pool)      # [B, 1]

        # Concat: [0:Stop, 1..K:Delete, K+1:Insert]
        all_logits = torch.cat([stop_logit, del_logits, insert_logit], dim=1)  # [B, K+2]

        # 6. Action masks
        current_size = active_mask.sum(dim=1, keepdim=True)
        can_delete = active_mask & (current_size > 2)        # [B, K]
        can_insert = torch.ones((B, 1), dtype=torch.bool, device=self.device)
        can_stop = torch.ones((B, 1), dtype=torch.bool, device=self.device)

        full_mask = torch.cat([can_stop, can_delete, can_insert], dim=1)  # [B, K+2]

        return all_logits.masked_fill(~full_mask, -1e9)
