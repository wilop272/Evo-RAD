"""
Standardized Evaluation Metrics Module
Supports: ACC, F1, AUC, Sensitivity, Specificity
Multi-seed experiments with mean/std reporting
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    roc_auc_score, 
    confusion_matrix,
    recall_score,
    precision_score
)
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def compute_all_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute all 5 standard metrics: ACC, F1, AUC, Sensitivity, Specificity
    
    Args:
        logits: [N, C] prediction logits or probabilities
        targets: [N] ground truth labels
        num_classes: number of classes (auto-detected if None)
    
    Returns:
        Dictionary with all metrics
    """
    # Convert to numpy
    if isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()
    

    if num_classes is None:
        # If [N, 1], we assume binary (2 classes) if not inferred from targets
        if logits.shape[1] == 1:
            num_classes = max(int(targets.max()) + 1, 2)
        else:
            num_classes = max(int(targets.max()) + 1, logits.shape[1])
            
    # Get predictions and probabilities
    if logits.shape[1] > 1:
        # Multi-class or Binary with 2 outputs
        preds = np.argmax(logits, axis=1)
        
        # Softmax if not already probabilities
        if not np.allclose(logits.sum(axis=1), 1.0, atol=1e-2):
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
            probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        else:
            probs = logits
    else:
        # Binary case with single output (BCE) -> [N, 1]
        # Apply Sigmoid
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
            # Binary classification
            if probs.shape[1] > 1:
                score = probs[:, 1]
            else:
                score = probs.flatten()
            auc = roc_auc_score(targets, score)
        else:
            # Multi-class: one-vs-rest macro average
            # This treats each class as a 'current vs rest' binary problem and averages results.
            # Good for imbalanced classes (treats all classes equally).
            auc = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
            
    except (ValueError, IndexError) as e:
        # Handle edge cases (e.g. only one class present in batch)
        # Or if a sample has only one class (which is always true for single-label, wait)
        # roc_auc_score(average='samples') works even if each sample has only 1 positive label
        # provided there's at least one negative label (which is true if num_classes > 1)
        auc = 0.5
    
    # 4 & 5. Sensitivity (Recall) and Specificity
    if num_classes == 2:
        # Binary classification
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Multi-class: macro average
        sensitivity = recall_score(targets, preds, average='macro', zero_division=0)
        
        # Specificity for multi-class: average of per-class specificity
        
        # Specificity for multi-class: average of per-class specificity
        # Fix: Force labels to include all classes to avoid shape mismatch if some classes are missing
        cm = confusion_matrix(targets, preds, labels=range(num_classes))
        specificities = []
        for i in range(num_classes):
            tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
            fp = cm[:, i].sum() - cm[i, i]
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificities.append(spec)
        specificity = np.mean(specificities)
    
    return {
        'ACC': acc * 100,  # Convert to percentage
        'F1': f1 * 100,
        'AUC': auc * 100,
        'Sensitivity': sensitivity * 100,
        'Specificity': specificity * 100
    }


def compute_topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk=(1, 5)) -> Dict[str, float]:
    """
    Compute top-k accuracy (for compatibility with existing code)
    
    Args:
        logits: [N, C] prediction logits
        targets: [N] ground truth labels
        topk: tuple of k values
    
    Returns:
        Dictionary with Acc@k for each k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = targets.size(0)

        _, pred = logits.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))

        res = {}
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res[f'Acc@{k}'] = correct_k.item() * 100.0 / batch_size
        return res


class MultiSeedEvaluator:
    """
    Run experiments with multiple seeds and compute mean/std
    """
    def __init__(self, seeds: List[int] = [1, 2, 3]):
        self.seeds = seeds
        self.results = {seed: {} for seed in seeds}
    
    def add_result(self, seed: int, metrics: Dict[str, float]):
        """Add results for a specific seed"""
        if seed not in self.results:
            self.results[seed] = {}
        self.results[seed].update(metrics)
    
    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute mean and std across all seeds
        
        Returns:
            Dictionary with 'mean' and 'std' for each metric
        """
        # Collect all metric names
        all_metrics = set()
        for seed_results in self.results.values():
            all_metrics.update(seed_results.keys())
        
        stats = {}
        for metric in all_metrics:
            values = []
            for seed in self.seeds:
                if metric in self.results[seed]:
                    values.append(self.results[seed][metric])
            
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                    'values': values
                }
        
        return stats
    
    def format_results(self, stats: Optional[Dict] = None) -> str:
        """
        Format results as mean ± std
        
        Returns:
            Formatted string for reporting
        """
        if stats is None:
            stats = self.compute_statistics()
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"Results across {len(self.seeds)} seeds: {self.seeds}")
        lines.append("=" * 60)
        
        # Standard metrics order
        metric_order = ['ACC', 'F1', 'AUC', 'Sensitivity', 'Specificity']
        
        for metric in metric_order:
            if metric in stats:
                mean = stats[metric]['mean']
                std = stats[metric]['std']
                lines.append(f"{metric:15s}: {mean:6.2f} ± {std:5.2f}")
        
        # Add any other metrics not in standard order
        for metric in sorted(stats.keys()):
            if metric not in metric_order:
                mean = stats[metric]['mean']
                std = stats[metric]['std']
                lines.append(f"{metric:15s}: {mean:6.2f} ± {std:5.2f}")
        
        lines.append("=" * 60)
        
        # Individual seed results
        lines.append("\nPer-Seed Results:")
        lines.append("-" * 60)
        for seed in self.seeds:
            lines.append(f"\nSeed {seed}:")
            for metric in metric_order:
                if metric in self.results[seed]:
                    val = self.results[seed][metric]
                    lines.append(f"  {metric:15s}: {val:6.2f}")
        
        return "\n".join(lines)
    
    def save_results(self, filepath: str):
        """Save results to file"""
        stats = self.compute_statistics()
        with open(filepath, 'w') as f:
            f.write(self.format_results(stats))
        print(f"Results saved to {filepath}")


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Example usage
if __name__ == "__main__":
    # Test the metrics
    print("Testing metrics computation...")
    
    # Simulate predictions
    logits = torch.randn(100, 10)  # 100 samples, 10 classes
    targets = torch.randint(0, 10, (100,))
    
    metrics = compute_all_metrics(logits, targets)
    print("\nSingle run metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
    
    # Test multi-seed evaluator
    print("\n" + "="*60)
    print("Testing multi-seed evaluation...")
    evaluator = MultiSeedEvaluator(seeds=[1, 2, 3])
    
    for seed in [1, 2, 3]:
        set_seed(seed)
        logits = torch.randn(100, 10)
        targets = torch.randint(0, 10, (100,))
        metrics = compute_all_metrics(logits, targets)
        evaluator.add_result(seed, metrics)
    
    print(evaluator.format_results())
