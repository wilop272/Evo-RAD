"""
Standardized Evaluation Metrics Module
Supports: ACC, F1, Sensitivity
Multi-seed experiments with mean/std reporting
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
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
    Compute 3 standard metrics: ACC, F1, Sensitivity
    
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
        if logits.shape[1] == 1:
            num_classes = max(int(targets.max()) + 1, 2)
        else:
            num_classes = max(int(targets.max()) + 1, logits.shape[1])
            
    # Get predictions and probabilities
    if logits.shape[1] > 1:
        preds = np.argmax(logits, axis=1)
    else:
        probs = 1 / (1 + np.exp(-logits)) 
        preds = (probs > 0.5).astype(int).reshape(-1)
    
    # 1. Accuracy
    acc = accuracy_score(targets, preds)
    
    # 2. F1 Score
    if num_classes == 2:
        f1 = f1_score(targets, preds, average='binary', zero_division=0)
    else:
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
    
    # 3. Sensitivity (Recall)
    if num_classes == 2:
        tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    else:
        sensitivity = recall_score(targets, preds, average='macro', zero_division=0)
    
    return {
        'ACC': acc * 100,
        'F1': f1 * 100,
        'Sensitivity': sensitivity * 100
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
        metric_order = ['ACC', 'F1', 'Sensitivity']
        
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
