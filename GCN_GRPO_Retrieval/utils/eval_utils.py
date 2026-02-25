"""
Evaluation Utilities (Visual Only + Conservative)
"""
import torch
import torch.nn.functional as F
from collections import Counter
from typing import List, Dict, Tuple

class TrajectoryTracker:
    def __init__(self, idx_to_label: Dict[int, str]):
        self.idx_to_label = idx_to_label
        self.history = []
        
    def log_initial(self, active_labels):
        classes = [self.idx_to_label.get(l.item(), str(l.item())) for l in active_labels]
        self.history.append(f"Initial Active Set ({len(classes)}): {classes}")
        
    def log_step(self, step, action_type, node_idx, label):
        lbl_str = self.idx_to_label.get(label.item(), str(label.item()))
        if action_type == 'Delete':
            self.history.append(f"Step {step}: Deleted Index {node_idx} (Label: {lbl_str})")
        elif action_type == 'Insert':
            self.history.append(f"Step {step}: Inserted Index {node_idx} (Label: {lbl_str})")
        elif action_type == 'Stop':
            self.history.append(f"Step {step}: Stop")
            
    def log_final(self, active_labels, acc, purity):
        classes = [self.idx_to_label.get(l.item(), str(l.item())) for l in active_labels]
        self.history.append(f"Final Active Set ({len(classes)}): {classes}")
        self.history.append(f"Result -> Accuracy: {acc:.2f}, Purity: {purity:.2f}")
        
    def get_report(self):
        return "\n".join(self.history)

def evaluate_trajectory(
    env,
    policy_net,
    reward_engine,
    query_features, 
    query_label,    
    candidate_features,
    candidate_text,
    candidate_labels,
    train_kg,
    idx_to_label,
    max_active_size=12,
    device='cuda',
    trainer=None,
    **kwargs
):
    """
    Run a single-sample trajectory for visualization / debugging.
    Uses trainer._refine_features_for_policy to build K+1 node graph
    (query as node 0).
    """
    policy_net.eval()
    tracker = TrajectoryTracker(idx_to_label)
    
    state, _ = env.reset(
        query_features, candidate_features, candidate_text, candidate_labels, query_label, train_kg
    )
    
    initial_labels = state.candidate_labels[:, :env.initial_k].clone()
    tracker.log_initial(initial_labels[0])
    
    for step in range(env.max_steps):
        if state.done.all():
            break

        # Use trainer's method if available (recommended path)
        if trainer is not None:
            active_vis, active_stats, _, _, active_text = trainer._refine_features_for_policy(
                state, candidate_features, query_features, train_kg
            )
        else:
            # Fallback: quick K+1 construction inline
            K = state.active_indices.shape[1]
            D_vis = state.candidate_visual_features.shape[2]
            D_txt = state.candidate_text_features.shape[2]
            B = 1

            idx_vis = state.active_indices.unsqueeze(2).expand(-1, -1, D_vis)
            cand_vis = F.normalize(torch.gather(state.candidate_visual_features, 1, idx_vis), dim=2)
            query_vis = F.normalize(query_features, dim=1).unsqueeze(1)
            active_vis = torch.cat([query_vis, cand_vis], dim=1)

            idx_txt = state.active_indices.unsqueeze(2).expand(-1, -1, D_txt)
            cand_text = F.normalize(torch.gather(state.candidate_text_features, 1, idx_txt), dim=2)
            query_text_zero = torch.zeros(B, 1, D_txt, device=device)
            active_text = torch.cat([query_text_zero, cand_text], dim=1)

            # Simple stats
            cand_sims = torch.gather(state.candidate_sims, 1, state.active_indices)
            cand_ranks = state.active_indices.float() / float(env.pool_size)
            
            query_base = torch.zeros(B, 1, 4, device=device)
            query_base[:, 0, 0] = 1.0
            cand_base = torch.stack([
                cand_sims,
                torch.zeros_like(cand_sims),  # text density placeholder
                cand_ranks,
                torch.zeros_like(cand_sims),  # kg density placeholder
            ], dim=2)
            all_base = torch.cat([query_base, cand_base], dim=1)
            
            query_mask = torch.ones(B, 1, dtype=torch.bool, device=device)
            all_mask = torch.cat([query_mask, state.active_mask], dim=1)
            batch_mask_f = all_mask.unsqueeze(2).float()
            masked = all_base * batch_mask_f
            mean_s = masked.sum(dim=1, keepdim=True) / batch_mask_f.sum(dim=1, keepdim=True).clamp(min=1)
            dev = (all_base - mean_s) * batch_mask_f
            active_stats = torch.cat([all_base, dev], dim=2)

        with torch.no_grad():
            logits = policy_net(
                active_features=active_vis,
                active_stats=active_stats,
                active_mask=state.active_mask,
                text_features=active_text,
            )
             
            action = logits.argmax(dim=1)
            
        # Log
        act = action.item()
        if act == 0: tracker.log_step(step+1, 'Stop', -1, torch.tensor(-1))
        elif act <= env.initial_k: 
             idx = act - 1
             real_idx = state.active_indices[0, idx]
             real_lbl = state.candidate_labels[0, real_idx]
             tracker.log_step(step+1, 'Delete', real_idx, real_lbl)
        else:
             tracker.log_step(step+1, 'Insert', -1, torch.tensor(-1))
             
        state = env.step(state, action)
        
    # Final Metrics
    active_labels_final = torch.gather(state.candidate_labels, 1, state.active_indices)
    
    _, metrics = reward_engine.compute_reward(
        query_label, active_labels_final, state.active_mask, initial_labels=initial_labels
    )
    
    mask = state.active_mask[0]
    final_lbls = active_labels_final[0][mask]
    tracker.log_final(final_lbls, metrics['accuracy'][0], metrics['purity'][0])
    
    return tracker.get_report(), metrics['accuracy'][0]
