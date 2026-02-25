"""
Dynamic Retrieval Environment for GRPO

Key design:
- reset(): sorts the entire candidate pool by cosine similarity to the query (descending).
  Active set starts as top-K candidates (most visually similar to the query).
- step() INSERT: always inserts the pool candidate with the HIGHEST query similarity
  that is not currently in the active set — no randomness in insert.
"""
import torch
import torch.nn.functional as F


class RetrievalEnv:
    def __init__(
        self,
        initial_k=16,               # Active set size
        max_steps=10,
        retrieval_pool_size=100,    # Total candidate pool size
        device='cuda'
    ):
        self.initial_k = initial_k
        self.max_steps = max_steps
        self.pool_size = retrieval_pool_size
        self.device = device

    # ------------------------------------------------------------------
    def reset(
        self,
        query_features: torch.Tensor,       # [B, D]
        candidate_features: torch.Tensor,   # [B, Pool, D]
        candidate_text: torch.Tensor,       # [B, Pool, D_txt]
        candidate_labels: torch.Tensor,     # [B, Pool]
        train_kg: torch.Tensor,             # [C, C]
        query_labels: torch.Tensor = None,  # [B]  (optional for test)
    ):
        B = query_features.shape[0]
        self.batch_size = B

        # 1. Cosine similarities: query vs every pool candidate  [B, Pool]
        q_norm = F.normalize(query_features, dim=1).unsqueeze(1)     # [B, 1, D]
        c_norm = F.normalize(candidate_features, dim=2)               # [B, Pool, D]
        sims   = torch.bmm(q_norm, c_norm.transpose(1, 2)).squeeze(1) # [B, Pool]

        # 2. Sort pool by query similarity (descending).
        #    After sorting: slot 0 = most similar to query, slot Pool-1 = least similar.
        sorted_order = torch.argsort(sims, dim=1, descending=True)    # [B, Pool]

        # Reorder all candidate tensors to match sorted_order
        idx_vis = sorted_order.unsqueeze(2).expand(-1, -1, candidate_features.shape[2])
        idx_txt = sorted_order.unsqueeze(2).expand(-1, -1, candidate_text.shape[2])

        self.candidate_visual_features = torch.gather(candidate_features, 1, idx_vis)
        self.candidate_text_features   = torch.gather(candidate_text,     1, idx_txt)
        self.candidate_labels          = torch.gather(candidate_labels,   1, sorted_order)
        self.candidate_sims            = torch.gather(sims,               1, sorted_order)  # descending
        self.query_features            = query_features

        # 3. Active set = top-K most similar (slots 0..K-1 in sorted space)
        self.active_mask    = torch.ones( (B, self.initial_k), dtype=torch.bool, device=self.device)
        self.buffer_mask    = torch.ones( (B, self.pool_size), dtype=torch.bool, device=self.device)
        self.active_indices = torch.arange(self.initial_k, device=self.device) \
                                   .unsqueeze(0).expand(B, -1).clone()
        self.buffer_indices = torch.arange(self.pool_size, device=self.device) \
                                   .unsqueeze(0).expand(B, -1).clone()

        self.query_labels = query_labels if query_labels is not None \
                            else torch.zeros(B, device=self.device)

        self.step_count = torch.zeros(B, device=self.device)
        self.done       = torch.zeros(B, dtype=torch.bool, device=self.device)

        return self, {}

    # ------------------------------------------------------------------
    def step(self, state, actions):
        """
        Actions:
          0       : Stop
          1 .. K  : Delete item at active-set slot (action - 1)
          K + 1   : Insert — adds the pool candidate with the HIGHEST
                    query-image similarity that is NOT already in the active set.
        """
        B = actions.shape[0]
        state.step_count += 1

        is_stop   = (actions == 0)
        is_delete = (actions >= 1) & (actions <= self.initial_k)
        is_insert = (actions == self.initial_k + 1)

        # ── Stop ──────────────────────────────────────────────────────
        state.done = state.done | is_stop | (state.step_count >= self.max_steps)

        # ── Delete ────────────────────────────────────────────────────
        batch_indices = torch.arange(B, device=self.device)

        min_size       = 2  # keep at least 2 active items
        current_counts = state.active_mask.sum(dim=1)
        safe_to_delete = (current_counts > min_size) & (~state.done)

        del_mask = is_delete & safe_to_delete
        if del_mask.any():
            slot_to_del = (actions - 1)[del_mask]
            state.active_mask[batch_indices[del_mask], slot_to_del] = False

        # ── Insert: highest-sim non-active pool candidate ─────────────
        ins_mask = is_insert & (~state.done)
        if ins_mask.any():
            for b in batch_indices[ins_mask].tolist():
                curr_indices = state.active_indices[b]   # [K], pool slot indices
                curr_mask    = state.active_mask[b]      # [K]
                active_slots = curr_indices[curr_mask]   # currently active pool slots

                # candidate_sims[b] is already in descending order (from reset).
                # Zero-out (exclude) active slots to find the best available.
                avail_sims = state.candidate_sims[b].clone()  # [Pool]
                avail_sims[active_slots] = -1e9

                if (avail_sims > -1e8).sum() == 0:
                    continue  # pool exhausted

                best_slot = int(avail_sims.argmax().item())

                if (~curr_mask).any():
                    # Fill the first empty slot in the active set
                    empty = int(torch.nonzero(~curr_mask)[0].item())
                    state.active_indices[b, empty] = best_slot
                    state.active_mask[b, empty]    = True
                else:
                    # Active set is full → replace the item with the LOWEST query sim
                    active_sim_vals = state.candidate_sims[b][curr_indices].clone()
                    active_sim_vals[~curr_mask] = 1e9  # ignore inactive slots
                    worst = int(active_sim_vals.argmin().item())
                    state.active_indices[b, worst] = best_slot
                    # active_mask[b, worst] stays True

        return state
