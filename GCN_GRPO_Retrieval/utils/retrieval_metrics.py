"""
Unified Retrieval Evaluation Module for GCN Project
Supports Hard Vote, Soft Vote, and 5 Standard Metrics
Ported from retizero_retrieval_baseline.py with robust AUC handling
"""
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    recall_score
)
import warnings
warnings.filterwarnings('ignore')


def differentiable_soft_vote(
    query_features: torch.Tensor,
    candidate_features: torch.Tensor,
    candidate_labels: torch.Tensor,
    num_classes: int,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Compute differentiable retrieval probabilities using temperature-scaled softmax.
    
    This method is superior to hard averaging for AUC calculation because it:
    1. Treats all candidates with soft weights (not just top-K)
    2. Uses temperature to control sharpness (lower = more like hard vote)
    3. Produces smooth probability distributions ideal for AUC
    
    Mathematical formulation:
        P(y=c|q) = Σ_i [exp(sim(q,i)/τ) / Σ_j exp(sim(q,j)/τ)] × 𝕀(y_i = c)
    
    Args:
        query_features: [B, D] - Query features (normalized)
        candidate_features: [M, D] - Candidate features (normalized)
        candidate_labels: [M] - Candidate labels
        num_classes: int - Number of classes
        temperature: float - Temperature parameter
            - τ → 0: Approaches hard vote (only max similarity matters)
            - τ = 1: Standard softmax
            - τ → ∞: Uniform distribution
            - Recommended: 0.05-0.1 for retrieval tasks
    
    Returns:
        class_probs: [B, num_classes] - Probability distribution over classes
    
    Example:
        >>> probs = differentiable_soft_vote(
        ...     test_features, train_features, train_labels,
        ...     num_classes=20, temperature=0.07
        ... )
        >>> metrics = compute_all_metrics(probs.cpu().numpy(), test_labels.numpy())
    """
    # 1. Compute similarity matrix [B, M]
    # Assumes features are already normalized
    sims = torch.mm(query_features, candidate_features.t())
    
    # 2. Apply temperature scaling and softmax [B, M]
    # This converts similarities to "importance weights"
    # weights[b, i] = how much candidate i contributes to query b's prediction
    weights = F.softmax(sims / temperature, dim=1)
    
    # 3. Aggregate to class probabilities [B, num_classes]
    class_probs = torch.zeros(
        query_features.size(0), num_classes,
        device=query_features.device, dtype=query_features.dtype
    )
    
    # Expand labels to match batch dimension [B, M]
    labels_expanded = candidate_labels.unsqueeze(0).expand(query_features.size(0), -1)
    
    # Scatter-add weights to corresponding classes (fully differentiable)
    class_probs.scatter_add_(1, labels_expanded, weights)
    
    return class_probs



def compute_all_metrics(
    logits: np.ndarray,
    targets: np.ndarray,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute all 5 standard metrics: ACC, F1, AUC, Sensitivity, Specificity
    
    Args:
        logits: [N, C] prediction logits or probabilities (numpy array)
        targets: [N] ground truth labels (numpy array)
        num_classes: number of classes
    
    Returns:
        Dictionary with all metrics
    """
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
        
    if num_classes is None:
        if logits.ndim > 1:
            num_classes = logits.shape[1]
        else:
            num_classes = max(int(targets.max()) + 1, 2)
            
    # Get predictions
    if logits.ndim > 1 and logits.shape[1] > 1:
        # Multi-class or Binary with 2 outputs
        preds = np.argmax(logits, axis=1)
        
        # === Robust probability conversion for AUC ===
        # Handle negative values (Soft Vote cosine similarity can be negative)
        if np.any(logits < 0):
            # Use Softmax normalization (numerically stable)
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        else:
            # Use L1 normalization (preserves linear ratios, suitable for Hard Vote)
            row_sums = logits.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0  # Avoid division by zero
            probs = logits / row_sums
        
        # Double normalization to fix sklearn "sum must be 1.0" error
        # Float precision may cause sum to be 1.0000001
        probs = probs / probs.sum(axis=1, keepdims=True)
        probs = np.clip(probs, 0.0, 1.0)  # Clip to prevent -0.0
             
    else:
        # Binary case with single output
        probs = 1 / (1 + np.exp(-logits)) 
        preds = (probs > 0.5).astype(int).reshape(-1)
    
    # 1. Accuracy
    acc = accuracy_score(targets, preds)
    
    # 2. F1 Score
    if num_classes == 2:
        f1 = f1_score(targets, preds, average='binary', zero_division=0)
    else:
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
    
    # 3. AUC (Macro-AUC: One-vs-Rest)
    try:
        if num_classes == 2:
            if probs.ndim > 1 and probs.shape[1] > 1:
                score = probs[:, 1]
            else:
                score = probs.flatten()
            auc = roc_auc_score(targets, score)
        else:
            # Multi-class: one-vs-rest macro average
            if len(np.unique(targets)) < 2:
                auc = 0.5
            else:
                auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
            
    except Exception as e:
        print(f"Warning: AUC computation failed: {e}")
        auc = 0.5
    
    # 4 & 5. Sensitivity (Recall) and Specificity
    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Multi-class: macro average sensitivity
        sensitivity = recall_score(targets, preds, average='macro', zero_division=0)
        
        # Specificity: average of per-class specificity
        cm = confusion_matrix(targets, preds, labels=range(num_classes))
        specificities = []
        for i in range(num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificities.append(spec)
        specificity = np.mean(specificities)
    
    return {
        'ACC': acc * 100,
        'F1': f1 * 100,
        'AUC': auc * 100,
        'Sensitivity': sensitivity * 100,
        'Specificity': specificity * 100
    }


def hard_vote(
    top_k_indices: torch.Tensor,
    top_k_sims: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Hard voting with tie-breaking using average similarity.
    
    Args:
        top_k_indices: [N, K] - Indices of top-K neighbors
        top_k_sims: [N, K] - Similarity scores
        train_labels: [M] - Labels of all training samples
        num_classes: int - Number of classes
    
    Returns:
        vote_probs: [N, num_classes] - Vote probability matrix
    """
    N, K = top_k_indices.shape
    device = top_k_indices.device
    
    # Gather labels of top-K neighbors
    if train_labels.device != device:
        train_labels = train_labels.to(device)
        
    top_k_labels = train_labels[top_k_indices]  # [N, K]
    
    # Initialize votes [N, num_classes]
    vote_scores = torch.zeros(N, num_classes, device=device)
    
    # 1. Count basic votes
    for i in range(N):
        counts = torch.bincount(top_k_labels[i], minlength=num_classes).float()
        vote_scores[i] = counts
        
    # 2. Add tie-breaking using AVERAGE similarity (not max)
    # Epsilon increased to 1e-2 to avoid float32 precision loss
    epsilon = 1e-2
    
    # Compute average similarity for each class per sample
    avg_sims = torch.zeros(N, num_classes, device=device)
    class_counts = torch.zeros(N, num_classes, device=device)
    
    for i in range(N):
        labels_i = top_k_labels[i]  # [K]
        sims_i = top_k_sims[i]      # [K]
        
        for k_idx in range(K):
            lbl = labels_i[k_idx]
            sim = sims_i[k_idx]
            avg_sims[i, lbl] += sim
            class_counts[i, lbl] += 1
    
    # Compute average (avoid division by zero)
    mask = class_counts > 0
    avg_sims[mask] = avg_sims[mask] / class_counts[mask]
    
    # Add tie-breaking score (shift to positive range)
    tie_break = (avg_sims + 2.0) * epsilon
    
    vote_scores += tie_break * mask.float()
    
    # Normalize to probabilities
    return vote_scores / K


def soft_vote(
    top_k_indices: torch.Tensor,
    top_k_sims: torch.Tensor,
    train_labels: torch.Tensor,
    num_classes: int
) -> torch.Tensor:
    """
    Soft voting: AVERAGE similarity per class (not sum).
    
    Args:
        top_k_indices: [N, K]
        top_k_sims: [N, K]
        train_labels: [M]
        num_classes: int
    
    Returns:
        score_matrix: [N, num_classes] - Average similarity per class
    """
    N, K = top_k_indices.shape
    device = top_k_indices.device
    
    # Gather labels of top-K neighbors
    top_k_labels = train_labels[top_k_indices]  # [N, K]
    
    # Initialize accumulators
    sum_scores = torch.zeros(N, num_classes, device=device)
    class_counts = torch.zeros(N, num_classes, device=device)
    
    # Accumulate similarities and counts per class
    sum_scores.scatter_add_(1, top_k_labels, top_k_sims)
    class_counts.scatter_add_(1, top_k_labels, torch.ones_like(top_k_sims))
    
    # Compute average (avoid division by zero)
    mask = class_counts > 0
    avg_scores = torch.zeros_like(sum_scores)
    avg_scores[mask] = sum_scores[mask] / class_counts[mask]
    
    return avg_scores


def evaluate_retrieval(
    query_features: torch.Tensor,
    query_labels: torch.Tensor,
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    k: int,
    num_classes: int,
    use_differentiable: bool = False,
    temperature: float = 0.07
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate retrieval using Hard Vote, Soft Vote, and optionally Differentiable Soft Vote.
    
    Args:
        query_features: [N, D] - Query features (should be normalized)
        query_labels: [N] - Query labels
        train_features: [M, D] - Training features (should be normalized)
        train_labels: [M] - Training labels
        k: int - Top-K neighbors (for hard/soft vote)
        num_classes: int - Number of classes
        use_differentiable: bool - Whether to compute differentiable soft vote
        temperature: float - Temperature for differentiable soft vote (default: 0.07)
    
    Returns:
        results: Dict with 'hard', 'soft', and optionally 'diff_soft' metrics
    """
    # Compute similarity matrix [N, M]
    sims = query_features @ train_features.t()
    
    # Get top-K
    top_k_sims, top_k_indices = torch.topk(sims, k=k, dim=1)
    
    # Hard Vote
    hard_scores = hard_vote(top_k_indices, top_k_sims, train_labels, num_classes)
    hard_metrics = compute_all_metrics(
        hard_scores.numpy(),
        query_labels.numpy(),
        num_classes=num_classes
    )
    
    # Soft Vote (Average similarity per class from top-K)
    soft_scores = soft_vote(top_k_indices, top_k_sims, train_labels, num_classes)
    soft_metrics = compute_all_metrics(
        soft_scores.numpy(),
        query_labels.numpy(),
        num_classes=num_classes
    )
    
    results = {
        'hard': hard_metrics,
        'soft': soft_metrics
    }
    
    # Differentiable Soft Vote (Temperature-scaled softmax over ALL candidates)
    if use_differentiable:
        diff_scores = differentiable_soft_vote(
            query_features, train_features, train_labels,
            num_classes=num_classes, temperature=temperature
        )
        diff_metrics = compute_all_metrics(
            diff_scores.cpu().numpy(),
            query_labels.numpy(),
            num_classes=num_classes
        )
        results['diff_soft'] = diff_metrics
    
    return results


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics"""
    print(f"{prefix}ACC: {metrics['ACC']:.2f}%")
    print(f"{prefix}F1: {metrics['F1']:.2f}%")
    print(f"{prefix}AUC: {metrics['AUC']:.2f}%")
    print(f"{prefix}Sensitivity: {metrics['Sensitivity']:.2f}%")
    print(f"{prefix}Specificity: {metrics['Specificity']:.2f}%")
