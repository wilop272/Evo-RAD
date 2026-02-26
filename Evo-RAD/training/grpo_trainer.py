"""
GRPO Trainer for Dynamic Retrieval Evolution (Visual Only + Conservative)
"""
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from tqdm import tqdm
import copy

from models.dynamic_env import RetrievalEnv
from models.policy import PolicyNetwork
from training.reward import RewardEngine

class GRPOTrainer:
    def __init__(
        self,
        policy_net: PolicyNetwork,
        env: RetrievalEnv,
        lr: float = 1e-4,
        num_trajectories: int = 4, 
        device: str = 'cuda',
        initial_k: int = 5,
        kl_beta: float = 0.04,
        enable_acc: bool = True, enable_purity: bool = True, enable_density: bool = True,
        enable_step_insert: bool = True, enable_step_delete: bool = True
    ):
        self.policy_net = policy_net
        self.ref_model = None # Internal management
        
        self.env = env
        # Reward Engine with Ablation Control
        self.reward_engine = RewardEngine(
            device=device, 
            top_k=initial_k,
            enable_acc=enable_acc,
            enable_purity=enable_purity,
            enable_density=enable_density,
            enable_step_insert=enable_step_insert,
            enable_step_delete=enable_step_delete
        )
        self.K = num_trajectories
        self.device = device
        
        self.optimizer = optim.Adam(
            list(policy_net.parameters()),
            lr=lr
        )
        
        self.reference_policy = None
        self.kl_beta = 0.01 
        
        if policy_net is not None:
             self.reference_policy = copy.deepcopy(policy_net)
             self.reference_policy.eval()
             for p in self.reference_policy.parameters():
                 p.requires_grad = False
        
    def _refine_features_for_policy(self, state, candidate_features, query_features, train_kg, **kwargs):
        """
        Build policy input with query image as node-0 in the GCN graph.
        Returns tensors of shape [B, K+1, ...] where node 0 = query.
        Query text is a zero vector (no label leakage).
        """
        B = query_features.shape[0]
        K = state.active_indices.shape[1]  # number of candidate slots
        D_vis = state.candidate_visual_features.shape[2]
        D_txt = state.candidate_text_features.shape[2]

        # ── Candidate features (nodes 1..K) ──────────────────────────────
        # Visual
        idx_vis = state.active_indices.unsqueeze(2).expand(-1, -1, D_vis)
        cand_vis = torch.gather(state.candidate_visual_features, 1, idx_vis)
        cand_vis = F.normalize(cand_vis, dim=2)  # [B, K, D_vis]

        # Text
        idx_txt = state.active_indices.unsqueeze(2).expand(-1, -1, D_txt)
        cand_text = torch.gather(state.candidate_text_features, 1, idx_txt)
        cand_text = F.normalize(cand_text, dim=2)  # [B, K, D_txt]

        # ── Query features (node 0) ──────────────────────────────────────
        query_vis = F.normalize(query_features, dim=1).unsqueeze(1)  # [B, 1, D_vis]
        query_text_zero = torch.zeros(B, 1, D_txt, device=self.device)  # zero = no leakage

        # ── Prepend query → [B, K+1, D] ─────────────────────────────────
        all_vis = torch.cat([query_vis, cand_vis], dim=1)       # [B, K+1, D_vis]
        all_text = torch.cat([query_text_zero, cand_text], dim=1)  # [B, K+1, D_txt]

        # ── Adjacency matrix from text similarity [B, K+1, K+1] ─────────
        # Query row/col will be ~0 because its text is zero → no semantic edges
        txt_adj = torch.bmm(all_text, all_text.transpose(1, 2))
        N_all = K + 1
        mask_diag = 1.0 - torch.eye(N_all, device=self.device).unsqueeze(0)
        txt_adj = txt_adj * mask_diag

        # ── Stats for candidates (nodes 1..K) ───────────────────────────
        # 1. Visual Sim to query
        cand_vis_sims = torch.gather(state.candidate_sims, 1, state.active_indices)  # [B, K]

        # 2. Text Density (pairwise among candidates only, excluding query)
        cand_txt_adj = txt_adj[:, 1:, 1:]  # [B, K, K] — candidate-only sub-block
        cand_txt_adj_no_diag = cand_txt_adj * (1.0 - torch.eye(K, device=self.device).unsqueeze(0))
        denom_txt = max(K - 1, 1)
        cand_text_density = cand_txt_adj_no_diag.sum(dim=2) / denom_txt  # [B, K]

        # 3. Rank
        cand_ranks = state.active_indices.float() / float(self.env.pool_size)  # [B, K]

        # 4. KG Density
        kg_size = train_kg.shape[0]
        c_lbls = torch.gather(state.candidate_labels, 1, state.active_indices).clamp(0, kg_size - 1)
        kg_adj = train_kg[c_lbls.unsqueeze(2), c_lbls.unsqueeze(1)]  # [B, K, K]
        mask_2d = state.active_mask.unsqueeze(2) & state.active_mask.unsqueeze(1)
        kg_adj = kg_adj * mask_2d.float()
        num_valid = state.active_mask.sum(dim=1, keepdim=True)
        denom_kg = torch.clamp(num_valid - 1, min=1e-6)
        cand_kg_den = (kg_adj.sum(dim=2) - torch.diagonal(kg_adj, dim1=1, dim2=2)) / denom_kg
        cand_kg_den = cand_kg_den * state.active_mask.float()  # [B, K]

        # Candidate base stats [B, K, 4]
        cand_base = torch.stack([cand_vis_sims, cand_text_density, cand_ranks, cand_kg_den], dim=2)

        # ── Stats for query (node 0) ────────────────────────────────────
        # sim=1.0 (query to itself), text_density=0, rank=0, kg_density=0
        query_base = torch.zeros(B, 1, 4, device=self.device)
        query_base[:, 0, 0] = 1.0  # query sim to itself

        # ── Combine: [B, K+1, 4] ────────────────────────────────────────
        all_base = torch.cat([query_base, cand_base], dim=1)  # [B, K+1, 4]

        # Build mask: query is always active (True) + candidate mask
        query_mask = torch.ones(B, 1, dtype=torch.bool, device=self.device)
        all_mask = torch.cat([query_mask, state.active_mask], dim=1)  # [B, K+1]

        # Deviation stats (value - masked mean)
        batch_mask_f = all_mask.unsqueeze(2).float()  # [B, K+1, 1]
        masked_stats = all_base * batch_mask_f
        sum_stats = masked_stats.sum(dim=1, keepdim=True)
        count_stats = batch_mask_f.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_stats = sum_stats / count_stats
        deviation = (all_base - mean_stats) * batch_mask_f  # [B, K+1, 4]

        all_stats = torch.cat([all_base, deviation], dim=2)  # [B, K+1, 8]

        # Labels (for reference, not used by policy)
        active_labels = torch.gather(state.candidate_labels, 1, state.active_indices)

        # Buffer (dummy, kept for API compat)
        buffer_feat = torch.zeros(B, 1, D_vis + 3, device=self.device)

        # Return: features=[B,K+1,D], stats=[B,K+1,8], labels=[B,K], buffer, text=[B,K+1,D_txt]
        return all_vis, all_stats, active_labels, buffer_feat, all_text


    def explore_trajectories(self, batch):
        c_feat_k, c_txt_k, c_lbl_k, q_feat_k, q_lbl_k, B, train_kg = self._prepare_batch(batch)
        
        # Environment Reset
        state, _ = self.env.reset(
            query_features=q_feat_k,
            candidate_features=c_feat_k,
            candidate_text=c_txt_k,
            candidate_labels=c_lbl_k, 
            query_labels=q_lbl_k,
            train_kg=train_kg
        )
        
        saved_log_probs = []
        
        initial_indices = state.active_indices.clone()
        initial_indices_k = initial_indices[:, :self.env.initial_k]
        initial_labels = torch.gather(state.candidate_labels, 1, initial_indices_k)
        
        # Track Initial Density (for Reward)
        initial_density = None
        
        # Cumulative step reward
        cumulative_step_rewards = torch.zeros(B, device=self.device)
        
        for step in range(self.env.max_steps):
            if state.done.all():
                break
                
            # Get features, stats, graph
            active_vis, active_stats, active_lbls, buffer_feat, active_text = self._refine_features_for_policy(
                state, c_feat_k, q_feat_k, train_kg, query_labels=q_lbl_k
            )
            
            # Record initial density (step 0)
            if step == 0:
                # Text density channel — skip node 0 (query)
                cand_stats = active_stats[:, 1:, :]  # [B, K, 8]
                batch_mask = state.active_mask.float()
                batch_count = batch_mask.sum(dim=1).clamp(min=1.0)
                densities = (cand_stats[:, :, 1] * batch_mask).sum(dim=1) / batch_count
                initial_density = densities.detach().clone()
                
                # Capture initial visual sims for rank protection
                initial_sims = torch.gather(state.candidate_sims, 1, state.active_indices)
                self.initial_sims = initial_sims.detach().clone()
            
            logits = self.policy_net(
                active_features=active_vis,
                active_stats=active_stats,
                active_mask=state.active_mask,
                text_features=active_text,
                # buffer_features and others ignored by TPP policy
            )
            
            retrieval_k = self.env.initial_k
            min_steps = 0
            if retrieval_k >= 8: min_steps = 3
            elif retrieval_k >= 4: min_steps = 1
            
            if step < min_steps:
                logits[:, 0] = -1e9
            
            current_counts = state.active_mask.sum(dim=1) 
            
            # Action Sampling
            temperature = 1.0 
            dist = Categorical(logits=logits / temperature)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            
            # Reference
            if self.reference_policy is not None:
                with torch.no_grad():
                    # Same logic for reference
                    ref_active_vis, ref_active_stats, ref_active_lbls, ref_buffer_feat, ref_active_text = self._refine_features_for_policy(
                        state, c_feat_k, q_feat_k, train_kg, query_labels=q_lbl_k
                    )
                    
                    ref_logits = self.reference_policy(
                        active_features=ref_active_vis,
                        active_stats=ref_active_stats,
                        active_mask=state.active_mask,
                        text_features=ref_active_text,
                    )
                    
                    if step < min_steps: ref_logits[:, 0] = -1e9
                        
                    ref_dist = Categorical(logits=ref_logits)
                    ref_log_prob = ref_dist.log_prob(actions)
            else:
                ref_log_prob = None
            
            # Compute step reward
            step_r = self.reward_engine.compute_step_reward(
                actions=actions,
                state=state,
                query_labels=q_lbl_k
            )
            cumulative_step_rewards += step_r
            
            active_mask_step = ~state.done
            state = self.env.step(state, actions)
            
            step_entropy = dist.entropy() 
            saved_log_probs.append((log_prob, ref_log_prob, active_mask_step, step_entropy))
            
        # Final Reward Calculation
        # Get Final Density
        active_vis_result, active_stats_final, active_lbls, _, _ = self._refine_features_for_policy(
             state, c_feat_k, q_feat_k, train_kg, query_labels=q_lbl_k
        )
        
        # Density from candidate nodes only (skip node 0 = query)
        cand_stats_final = active_stats_final[:, 1:, :]  # [B, K, 8]
        batch_mask = state.active_mask.float()
        batch_count = batch_mask.sum(dim=1).clamp(min=1.0)
        current_density = (cand_stats_final[:, :, 1] * batch_mask).sum(dim=1) / batch_count
        
        active_labels_final = torch.gather(state.candidate_labels, 1, state.active_indices)
        
        # Final reward calculation
        raw_rewards, metrics = self.reward_engine.compute_reward(
            query_labels=q_lbl_k,
            active_labels=active_labels_final,
            active_mask=state.active_mask,
            initial_labels=initial_labels,
            initial_density=initial_density,
            current_density=current_density,
            initial_sims=self.initial_sims,
            query_features=q_feat_k,
            active_features=active_vis_result
        )

        # Combine process and outcome reward
        raw_rewards = raw_rewards + cumulative_step_rewards
        
        # Advantage
        rewards_matrix = raw_rewards.view(B, K)
        mean_rewards = rewards_matrix.mean(dim=1, keepdim=True)
        std_rewards = rewards_matrix.std(dim=1, keepdim=True) + 1e-2
        advantages = (rewards_matrix - mean_rewards) / std_rewards
        advantages = torch.clamp(advantages, -1.0, 1.0)
        advantages = advantages.view(-1)
      

        
    def train_step(
        self,
        query_features: torch.Tensor,       # [B, D]
        query_labels: torch.Tensor,         # [B]
        candidate_features: torch.Tensor,   # [B, Pool, D_vis]
        candidate_text: torch.Tensor,       # [B, Pool, D_txt]
        candidate_labels: torch.Tensor,     # [B, Pool]
        train_kg: torch.Tensor              # [C, C]
    ):
        self.policy_net.train()
        
        B = query_features.shape[0]
        K = self.K
        
        def expand(t, k):
            return t.repeat_interleave(k, dim=0)
            
        q_feat_k = expand(query_features, K)
        c_feat_k = expand(candidate_features, K)
        c_lbl_k = expand(candidate_labels, K)
        c_txt_k = expand(candidate_text, K) 
        q_lbl_k = expand(query_labels, K)
        
        state, info = self.env.reset(
            query_features=q_feat_k,
            candidate_features=c_feat_k,
            candidate_text=c_txt_k,
            candidate_labels=c_lbl_k, 
            query_labels=q_lbl_k,
            train_kg=train_kg
        )
        
        saved_log_probs = []
        
        initial_indices = state.active_indices.clone()
        initial_indices_k = initial_indices[:, :self.env.initial_k]
        initial_labels = torch.gather(state.candidate_labels, 1, initial_indices_k)
        
        # Track Initial Density (for Reward)
        initial_density = None
        
        # Track Action Counts
        total_inserts = torch.zeros(B * K, device=self.device)
        total_deletes = torch.zeros(B * K, device=self.device)
        total_steps = torch.zeros(B * K, device=self.device)
        
        # Cumulative step reward
        cumulative_step_rewards = torch.zeros(B * K, device=self.device)
        
        for step in range(self.env.max_steps):
            if state.done.all():
                break
                
            # Get features, stats, graph
            active_vis, active_stats, active_lbls, buffer_feat, active_text = self._refine_features_for_policy(
                state, c_feat_k, q_feat_k, train_kg, query_labels=q_lbl_k
            )
            
            # Record Initial Density (Step 0)
            if step == 0:
                # Stats Channel 1 is Text Density
                batch_mask = state.active_mask.float()
                batch_count = batch_mask.sum(dim=1).clamp(min=1.0)
                # Weighted Sum by Mask
                densities = (active_stats[:, :, 1] * batch_mask).sum(dim=1) / batch_count
                initial_density = densities.detach().clone()
                
                # Capture initial visual sims for rank protection
                initial_sims = torch.gather(state.candidate_sims, 1, state.active_indices)
                self.initial_sims = initial_sims.detach().clone()
            
            logits = self.policy_net(
                active_features=active_vis,
                active_stats=active_stats,
                active_mask=state.active_mask,
                text_features=active_text,
            )
            
            retrieval_k = self.env.initial_k
            min_steps = 0
            if retrieval_k >= 8: min_steps = 3
            elif retrieval_k >= 4: min_steps = 1
            
            if step < min_steps:
                logits[:, 0] = -1e9
            
            current_counts = state.active_mask.sum(dim=1) 
            
            # Action Sampling
            temperature = 1.0 
            dist = Categorical(logits=logits / temperature)
            actions = dist.sample()
            log_prob = dist.log_prob(actions)
            
            # Reference
            if self.reference_policy is not None:
                with torch.no_grad():
                    ref_active_vis, ref_active_stats, ref_active_lbls, ref_buffer_feat, ref_txt_adj = self._refine_features_for_policy(
                        state, c_feat_k, q_feat_k, train_kg, query_labels=q_lbl_k
                    )
                    
                    ref_logits = self.reference_policy(
                        active_features=ref_active_vis,
                        active_stats=ref_active_stats,
                        buffer_features=ref_buffer_feat,
                        active_mask=state.active_mask,
                        buffer_mask=state.buffer_mask,
                        candidate_labels=ref_active_lbls,
                        dynamic_adj=ref_txt_adj,
                        max_active_size=self.env.initial_k + 2 
                    )
                    
                    if step < min_steps: ref_logits[:, 0] = -1e9
                        
                    ref_dist = Categorical(logits=ref_logits)
                    ref_log_prob = ref_dist.log_prob(actions)
            else:
                ref_log_prob = None
            
            # Compute step reward
            step_r = self.reward_engine.compute_step_reward(
                actions=actions,
                state=state,
                query_labels=q_lbl_k
            )
            cumulative_step_rewards += step_r
            
            # Track Actions
            is_insert = (actions == self.env.initial_k + 1).float()
            is_delete = ((actions >= 1) & (actions <= self.env.initial_k)).float()
            
            # Only count if not already done?
            not_done = (~state.done).float()
            total_inserts += is_insert * not_done
            total_deletes += is_delete * not_done
            total_steps += not_done
            
            active_mask_step = ~state.done
            state = self.env.step(state, actions)
            
            step_entropy = dist.entropy() 
            saved_log_probs.append((log_prob, ref_log_prob, active_mask_step, step_entropy))
            
        # Final Reward
        active_vis_result, active_stats_final, active_lbls, _, _ = self._refine_features_for_policy(
             state, c_feat_k, q_feat_k, train_kg, query_labels=q_lbl_k
        )
        
        batch_mask = state.active_mask.float()
        batch_count = batch_mask.sum(dim=1).clamp(min=1.0)
        current_density = (active_stats_final[:, :, 1] * batch_mask).sum(dim=1) / batch_count
        
        active_labels_final = torch.gather(state.candidate_labels, 1, state.active_indices)
        
        # Final reward calculation
        raw_rewards, metrics = self.reward_engine.compute_reward(
            query_labels=q_lbl_k,
            active_labels=active_labels_final,
            active_mask=state.active_mask,
            initial_labels=initial_labels,
            initial_density=initial_density,
            current_density=current_density,
            initial_sims=self.initial_sims,
            query_features=q_feat_k,
            active_features=active_vis_result
        )

        # Combine process and outcome reward
        raw_rewards = raw_rewards + cumulative_step_rewards
        
        # Advantage (Batch-Level Normalization)
        # Flatten rewards: [B*K]
        rewards_flat = raw_rewards.view(-1)
        mean_rewards = rewards_flat.mean()
        std_rewards = rewards_flat.std() + 1e-4
        advantages = (rewards_flat - mean_rewards) / std_rewards
        # advantages = torch.clamp(advantages, -2.0, 2.0) # Optional clipping
        # Reshape to [B, K]
        advantages = advantages.view(B, K)
        
        # Expand for steps? No, advantage is per trajectory
        # Saved log probs: List of [B, K]
        # advantages: [B, K] -> Expand to match log_prob shape [B, K] (Already matches)
        advantages = advantages.view(-1) 
        
        policy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        kl_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = torch.tensor(0.0, device=self.device)
        entropy_coef = 0.05
        
        num_steps = len(saved_log_probs)
        
        for log_prob, ref_log_prob, mask, step_entropy in saved_log_probs:
            step_loss = -log_prob * advantages * mask.float()
            policy_loss = policy_loss + step_loss.mean()
            
            if ref_log_prob is not None:
                kl_div = (log_prob - ref_log_prob) * mask.float()
                kl_loss = kl_loss + kl_div.mean()

            entropy_loss = entropy_loss + step_entropy.mean()

        if num_steps > 0:
            policy_loss = policy_loss / num_steps
            kl_loss = kl_loss / num_steps
            entropy_loss = entropy_loss / num_steps

        total_loss = policy_loss + self.kl_beta * kl_loss - entropy_coef * entropy_loss
        
        if num_steps > 0:
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)
            self.optimizer.step()
        
        # Reduce EMA update frequency
        self.step_counter = getattr(self, 'step_counter', 0) + 1
        if self.step_counter % 10 == 0:
            self._update_reference_policy()
        
        return {
            'loss': total_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_loss': kl_loss.item() if isinstance(kl_loss, torch.Tensor) else 0.0,
            'reward_mean': raw_rewards.mean().item(),
            'acc_mean': np.mean(metrics['accuracy']),
            'purity_mean': np.mean(metrics['purity']),
            'steps_mean': state.step_count.float().mean().item()
        }
    
    def _update_reference_policy(self):
        if self.reference_policy is None: return
        ema_alpha = 0.99
        with torch.no_grad():
            for ref_param, curr_param in zip(self.reference_policy.parameters(), self.policy_net.parameters()):
                ref_param.data.mul_(ema_alpha).add_(curr_param.data, alpha=1 - ema_alpha)

    def save_checkpoint(self, path):
        checkpoint = {
            'policy': self.policy_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        if self.reference_policy is not None:
            checkpoint['reference_policy'] = self.reference_policy.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt['policy'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        
        if 'reference_policy' in ckpt:
            from copy import deepcopy
            self.reference_policy = deepcopy(self.policy_net)
            self.reference_policy.load_state_dict(ckpt['reference_policy'])
            self.reference_policy.eval()
            for param in self.reference_policy.parameters():
                param.requires_grad = False
