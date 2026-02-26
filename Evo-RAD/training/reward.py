"""
Minimalist 5-Dim Reward (Zero Penalty)
1. Acc: +40.0 (Base) + +80.0 (Resurrection).
2. Purity: +20.0 (Base) + +50.0 (Gain).
3. KG Density: +30.0 (Semantic Center Protection).
4. No Penalties (Action Costs Removed).
"""
import torch
from collections import Counter

class RewardEngine:
    def __init__(self, device='cuda', top_k=8, 
                 enable_acc=True, enable_purity=True, enable_density=True,
                 enable_step_insert=True, enable_step_delete=True):
        self.device = device
        self.top_k = top_k
        self.enable_acc = enable_acc
        self.enable_purity = enable_purity
        self.enable_density = enable_density
        self.enable_step_insert = enable_step_insert
        self.enable_step_delete = enable_step_delete
        
    def compute_reward(self, query_labels, active_labels, active_mask, initial_labels=None, 
                       initial_density=None, current_density=None, **kwargs):
        B = query_labels.shape[0]
        # No Rank Weights needed (Density handles protection)
        
        # Default handling for density if missing
        if current_density is None: 
            current_density = torch.zeros(B, device=self.device)
            
        metrics_dict = {}
        rewards = torch.zeros(B, device=self.device)
        
        for b in range(B):
            q_lbl = query_labels[b].item()
            mask = active_mask[b]
            
            # --- 1. Basic Metrics ---
            curr_lbls = active_labels[b][mask]
            curr_size = mask.sum().item()
            
            if curr_size > 0:
                curr_pred = Counter(curr_lbls.tolist()).most_common(1)[0][0]
                curr_acc = 1.0 if curr_pred == q_lbl else 0.0
                curr_purity = (curr_lbls == q_lbl).sum().item() / curr_size
            else:
                curr_acc = 0.0
                curr_purity = 0.0
                rewards[b] = -60.0 # Emergency penalty for empty set
                continue
            
            # Initial Metrics used for Gain calculation
            init_acc = 0.0
            init_purity = 0.0
            if initial_labels is not None:
                init_lbls = initial_labels[b]
                if len(init_lbls) > 0:
                    init_pred = Counter(init_lbls.tolist()).most_common(1)[0][0]
                    init_acc = 1.0 if init_pred == q_lbl else 0.0
                    init_purity = (init_lbls == q_lbl).sum().item() / len(init_lbls)
            
            r = 0.0
            
            # --- 2. State Rewards (Zero Penalty) ---
            
            # A. Accuracy (Base + Resurrection)
            if self.enable_acc:
                if curr_acc == 1.0:
                    r += 40.0
                    if init_acc == 0.0:
                        r += 80.0 # Resurrection Bonus
            
            # B. Purity (Base + Gain)
            if self.enable_purity:
                r += curr_purity * 20.0 # Base Purity
                
                purity_gain = curr_purity - init_purity
                if purity_gain > 0:
                    r += purity_gain * 50.0 # Gain Bonus
            
            # C. KG Density (Semantic Protection)
            if self.enable_density:
                kg_score = current_density[b].item()
                r += kg_score * 30.0
            
            rewards[b] = r
            
            # Metrics
            if 'accuracy' not in metrics_dict: metrics_dict['accuracy'] = []
            if 'purity' not in metrics_dict: metrics_dict['purity'] = []
            metrics_dict['accuracy'].append(curr_acc)
            metrics_dict['purity'].append(curr_purity)
            
        return rewards, metrics_dict

    def compute_step_reward(self, actions, state, query_labels):
        """
        Compute immediate process feedback for actions (vectorized).
        - Insert correct label: +3.0
        - Delete wrong label: +1.0
        - Delete correct / Insert wrong: 0.0 (no penalty)
        """
        B = actions.shape[0]
        step_rewards = torch.zeros(B, device=self.device)
        
        # Action Masks
        num_actions = state.active_mask.shape[1]
        is_insert = (actions == num_actions + 1) # [B]
        is_delete = (actions >= 1) & (actions <= num_actions) # [B]
        delete_indices = actions - 1 # [B] (0-indexed)
        
        pool_size = state.candidate_visual_features.shape[1] # Usually 50
        
        # --- Handle Deletions ---
        if self.enable_step_delete and is_delete.any():
            # Clamp delete_indices to valid active indices range
            safe_del_indices = delete_indices.clamp(0, state.active_indices.shape[1]-1).unsqueeze(1)
            
            # Gather candidate index from active list
            target_cand_indices = torch.gather(state.active_indices, 1, safe_del_indices).squeeze(1)
            
            # Clamp candidate index to valid pool range (Safety against corrupted state)
            safe_cand_indices = target_cand_indices.clamp(0, pool_size - 1)
            
            # Gather labels
            target_labels = torch.gather(state.candidate_labels, 1, safe_cand_indices.unsqueeze(1)).squeeze(1)
            
            good_delete_mask = (target_labels != query_labels) & is_delete
            step_rewards[good_delete_mask] += 1.0
            
        # --- Handle Insertions ---
        if self.enable_step_insert and is_insert.any():
            all_sims = state.candidate_sims
            
            valid_mask = torch.ones((B, pool_size), dtype=torch.bool, device=self.device)
            
            # Clamp active_indices for scatter safety
            safe_active_indices = state.active_indices.clamp(0, pool_size - 1)
            
            valid_mask.scatter_(1, safe_active_indices, False)
            
            masked_sims = all_sims.clone()
            masked_sims[~valid_mask] = -1e9
            
            best_cand_idx = masked_sims.argmax(dim=1)
            
            # Safety checks for best_cand_idx (should be < Pool, but clamp just in case)
            best_cand_idx = best_cand_idx.clamp(0, pool_size - 1)
            
            cand_labels = torch.gather(state.candidate_labels, 1, best_cand_idx.unsqueeze(1)).squeeze(1)
            
            good_insert_mask = (cand_labels == query_labels) & is_insert
            step_rewards[good_insert_mask] += 3.0
            
        return step_rewards
