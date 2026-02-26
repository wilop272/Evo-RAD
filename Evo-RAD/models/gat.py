"""
2-Layer Graph Attention Network (GAT) for feature refinement
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """
    Single Graph Attention Layer
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        num_heads: Number of attention heads
        dropout: Dropout rate
        concat: Whether to concatenate heads (True) or average (False)
    """
    
    def __init__(
        self, 
        in_features: int, 
        out_features: int,
        num_heads: int = 1,
        dropout: float = 0.2,
        concat: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        
        # Linear transformations for each head
        self.W = nn.Parameter(torch.zeros(num_heads, in_features, out_features))
        self.a = nn.Parameter(torch.zeros(num_heads, 2 * out_features, 1))
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        
        Returns:
            out: Output features [N, out_features * num_heads] if concat
                 or [N, out_features] if not concat
        """
        N = x.shape[0]
        H = self.num_heads
        
        # Linear transformation for each head: [N, H, out_features]
        Wh = torch.einsum('nd,hdo->nho', x, self.W)
        
        # Compute attention coefficients
        # Concatenate features: [N, N, H, 2*out_features]
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [N, N, H, out]
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1, -1)  # [N, N, H, out]
        concat_features = torch.cat([Wh_i, Wh_j], dim=-1)  # [N, N, H, 2*out]
        
        # Attention scores: [N, N, H]
        e = torch.einsum('ijho,hok->ijh', concat_features, self.a).squeeze(-1)
        e = self.leakyrelu(e)
        
        # Mask attention scores with adjacency matrix
        # Add self-loops
        adj_with_self = adj.clone()
        adj_with_self.fill_diagonal_(1.0)
        
        # Expand adj for multi-head: [N, N, H]
        adj_expanded = adj_with_self.unsqueeze(-1).expand(-1, -1, H)
        
        # Mask and normalize
        e_masked = e.masked_fill(adj_expanded == 0, float('-inf'))
        attention = F.softmax(e_masked, dim=1)  # [N, N, H]
        attention = self.dropout_layer(attention)
        
        # Aggregate features: [N, H, out_features]
        h_prime = torch.einsum('ijh,jho->iho', attention, Wh)
        
        if self.concat:
            # Concatenate heads: [N, H * out_features]
            return h_prime.reshape(N, -1)
        else:
            # Average heads: [N, out_features]
            return h_prime.mean(dim=1)


class GAT(nn.Module):
    """
    2-Layer Graph Attention Network
    
    Args:
        in_features: Input feature dimension
        hidden_features: Hidden layer dimension
        out_features: Output feature dimension (same as input for residual)
        num_heads_layer1: Number of attention heads in layer 1
        num_heads_layer2: Number of attention heads in layer 2
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        in_features: int = 512,
        hidden_features: int = 256,
        out_features: int = 512,
        num_heads_layer1: int = 4,
        num_heads_layer2: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Layer 1: Multi-head attention with concatenation
        self.gat1 = GATLayer(
            in_features=in_features,
            out_features=hidden_features,
            num_heads=num_heads_layer1,
            dropout=dropout,
            concat=True
        )
        
        # Layer 2: Single-head attention
        self.gat2 = GATLayer(
            in_features=hidden_features * num_heads_layer1,
            out_features=out_features,
            num_heads=num_heads_layer2,
            dropout=dropout,
            concat=False
        )
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection (if dimensions match)
        self.use_residual = (in_features == out_features)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_features]
            adj: Adjacency matrix [N, N]
        
        Returns:
            out: Refined features [N, out_features]
        """
        identity = x if self.use_residual else None
        
        # Layer 1
        h = self.gat1(x, adj)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Layer 2
        out = self.gat2(h, adj)
        
        # Residual connection
        if self.use_residual:
            out = out + identity
        
        # Normalize output
        out = F.normalize(out, dim=-1)
        
        return out


if __name__ == '__main__':
    # Test GAT
    print("Testing GAT model...")
    
    N = 100  # Number of nodes
    D = 512  # Feature dimension
    
    # Create dummy data
    x = torch.randn(N, D)
    x = F.normalize(x, dim=-1)
    
    # Create random adjacency matrix
    adj = torch.rand(N, N) > 0.7
    adj = adj.float()
    adj = (adj + adj.T) > 0  # Make symmetric
    adj = adj.float()
    
    print(f"Input shape: {x.shape}")
    print(f"Adjacency shape: {adj.shape}")
    print(f"Number of edges: {adj.sum().item() / 2}")
    
    # Create model
    model = GAT(
        in_features=D,
        hidden_features=256,
        out_features=D,
        num_heads_layer1=4,
        num_heads_layer2=1,
        dropout=0.2
    )
    
    # Forward pass
    out = model(x, adj)
    
    print(f"\nOutput shape: {out.shape}")
    print(f"Output norm: {torch.norm(out, dim=-1).mean():.4f}")
    
    # Test gradient flow
    loss = out.sum()
    loss.backward()
    print(f"\nGradient check: {model.gat1.W.grad is not None}")
    
    print("\n✓ GAT model test passed!")
