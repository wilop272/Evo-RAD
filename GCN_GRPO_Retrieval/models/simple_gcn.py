"""
Simple GCN with Batch Support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGCNLayer(nn.Module):
    """
    Simple GCN layer: H' = σ(D^{-1/2} A D^{-1/2} H W)
    Supports batched inputs.
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D] or [N, D]
            adj: [B, N, N] or [N, N]
        """
        is_batched = (x.dim() == 3)
        if not is_batched:
            x = x.unsqueeze(0)
            adj = adj.unsqueeze(0)
            
        B, N, _ = x.shape
        
        # Self-loops
        eye = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        adj = adj + eye
        
        # Degree Normalization
        # Degree: [B, N]
        degree = adj.sum(dim=2) 
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
        
        # D_inv_sqrt: [B, N, N] (Batch diagonal)
        D_inv_sqrt = torch.diag_embed(degree_inv_sqrt)
        
        # Norm Adj: [B, N, N]
        adj_norm = torch.bmm(torch.bmm(D_inv_sqrt, adj), D_inv_sqrt)
        
        # Support: XW -> [B, N, Out]
        # x is [B, N, In], weight is [In, Out]
        # We can use matmul broadcasting or reshape
        support = torch.matmul(x, self.weight) 
        
        # Output: A_norm @ Support
        output = torch.bmm(adj_norm, support)
        
        if self.bias is not None:
            output = output + self.bias
            
        if not is_batched:
            output = output.squeeze(0)
            
        return output

class SimpleGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.gcn1 = SimpleGCNLayer(in_features, hidden_features)
        self.gcn2 = SimpleGCNLayer(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, adj):
        h = self.gcn1(x, adj)
        h = F.relu(h)
        h = self.dropout(h)
        h = self.gcn2(h, adj)
        if x.shape[-1] == h.shape[-1]:
            h = h + x
        return h

def create_gnn(model_type, in_features, hidden_features=256, out_features=None, dropout=0.1):
    if out_features is None: out_features = in_features
    if model_type == 'simple_gcn':
        return SimpleGCN(in_features, hidden_features, out_features, dropout)
    # ... support others if needed
    return SimpleGCN(in_features, hidden_features, out_features, dropout) # Default
