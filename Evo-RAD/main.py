"""
Main Training Script for Dynamic Retrieval Evolution (GRPO)
"""
import os
import sys
import torch
import torch.nn.functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Tensorboard not found, using dummy writer")
    class SummaryWriter:
        def __init__(self, log_dir=None): pass
        def add_scalar(self, tag, scalar_value, global_step=None, walltime=None): pass
        def close(self): pass

from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
from tqdm import tqdm
from utils.retrieval_metrics import compute_all_metrics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.feature_extractor import extract_all_features
from data.dataset import DataManager
from training.reward import RewardEngine
from training.grpo_trainer import GRPOTrainer
from models.dynamic_env import RetrievalEnv
from models.policy import PolicyNetwork
from data.bioclinical_bert import BioClinicalBERTExtractor
from data.disease_tags import DISEASE_CLINICAL_TAGS
from utils.eval_utils import evaluate_trajectory
from utils.standard_metrics import compute_all_metrics

def parse_args():
    parser = argparse.ArgumentParser(description='GRPO Retrieval Training')
    parser.add_argument('--data_root', type=str, required=True, help='Path to dataset root directory')
    parser.add_argument('--retizero_checkpoint', type=str, default=None, help='Path to RetiZero checkpoint')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_trajectories', type=int, default=8)
    parser.add_argument('--retrieval_pool_size', type=int, default=100)
    parser.add_argument('--initial_k', type=int, default=8)
    parser.add_argument('--max_steps', type=int, default=10)
    # Reward Weights
    parser.add_argument('--alpha', type=float, default=1.0, help='Purity weight')
    parser.add_argument('--beta', type=float, default=0.0, help='Step penalty')
    parser.add_argument('--kl_beta', type=float, default=0.04, help='KL penalty coefficient')
    parser.add_argument('--run_name', type=str, default="default_run", help='Name for Tensorboard/Checkpoints')
    
    # Ablation Flags (Negative flags for easier CLI usage: default is Enabled)
    parser.add_argument('--no_acc', action='store_true', help='Disable Accuracy Reward')
    parser.add_argument('--no_purity', action='store_true', help='Disable Purity Reward')
    parser.add_argument('--no_density', action='store_true', help='Disable Density Reward')
    parser.add_argument('--no_step_insert', action='store_true', help='Disable Step Insert Reward')
    parser.add_argument('--no_step_delete', action='store_true', help='Disable Step Delete Reward')
    
    parser.add_argument('--ablation_mode', type=str, default='full', choices=['full', 'no_stats', 'no_dev', 'no_ego'], help='Ablation mode for PolicyNetwork input dimensionality')

    parser.add_argument('--save_dir', type=str, default='checkpoints/grpo_dynamic')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Running on {device} with seed {args.seed}", flush=True)
    
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.save_dir, 'logs'))
    
    # Data & Features
    print("Loading Data/Features...")
    train_feat, val_feat, test_feat, info = extract_all_features(
        data_root=args.data_root,
        retizero_checkpoint=args.retizero_checkpoint
    )
    
    # Unpack...
    train_img = train_feat['image_features'].to(device)
    train_txt = train_feat['text_features'].to(device)
    train_lbl = train_feat['labels'].to(device)
    
    val_img = val_feat['image_features'].to(device)
    val_txt = val_feat['text_features'].to(device)
    val_lbl = val_feat['labels'].to(device)
    
    test_img = test_feat['image_features'].to(device)
    test_txt = test_feat['text_features'].to(device)
    test_lbl = test_feat['labels'].to(device)
    
    # Knowledge Graph (Semantic Weighted Graph)
    print("Building/Loading Semantic Graph (Weighted by BioClinicalBERT)...")
    kg_path = os.path.join(args.save_dir, 'semantic_kg.pt')
    
    manager = DataManager(args.data_root)
    idx_to_label = info['idx_to_label']
    num_classes = len(idx_to_label)
    disease_names = [idx_to_label[i] for i in range(num_classes)]
    
    # Check label alignment
    print("Verifying Disease Tags alignment...")
    tag_keys = set(DISEASE_CLINICAL_TAGS.keys())
    missing_tags = []
    for d_name in disease_names:
        if d_name not in tag_keys and d_name.lower() not in tag_keys:
            missing_tags.append(d_name)
    
    if missing_tags:
        print(f"⚠️  WARNING: Missing tags for diseases: {missing_tags}")
        print("    These will use 'Disease Name' as fallback tag.")
    else:
        print("✓ All disease labels referencable in tags database.")

    if os.path.exists(kg_path):
        train_kg = torch.load(kg_path, map_location=device)
        print("Loaded Semantic KG from cache.")
    else:
        print("Computing Semantic Similarity Matrix...")
        extractor = BioClinicalBERTExtractor(device=device)
        train_kg = extractor.compute_disease_similarity_matrix(disease_names)
        train_kg = train_kg.to(device)
        torch.save(train_kg, kg_path)
        print(f"Saved Semantic KG to {kg_path}")
        
    train_kg = train_kg.to(device)
    
    # Models
    print("Initializing Models...")
    
    # Cap Initial K at Pool Size (Safety)
    if args.initial_k > args.retrieval_pool_size:
        print(f"WARNING: Initial K ({args.initial_k}) > Pool Size ({args.retrieval_pool_size}). Capping Initial K to {args.retrieval_pool_size}.")
        args.initial_k = args.retrieval_pool_size
        
    env = RetrievalEnv(
        retrieval_pool_size=args.retrieval_pool_size,
        initial_k=args.initial_k,
        max_steps=args.max_steps,
        device=device
    )
    
    # Get visual feature dimension from extracted features
    visual_dim = train_img.shape[1]  # 512 or 1024 depending on model
    text_dim = train_txt.shape[1]     # Text feature dimension
    
    # Policy network (visual features only)
    policy = PolicyNetwork(input_dim=visual_dim, hidden_dim=128, device=device, ablation_mode=args.ablation_mode).to(device)
    

    
    trainer = GRPOTrainer(
        policy_net=policy,
        env=env,
        lr=args.lr,
        num_trajectories=args.num_trajectories,
        device=device,
        initial_k=args.initial_k,
        kl_beta=args.kl_beta,
        enable_acc=not args.no_acc,
        enable_purity=not args.no_purity,
        enable_density=not args.no_density,
        enable_step_insert=not args.no_step_insert,
        enable_step_delete=not args.no_step_delete
    )

    # --- Pre-compute / Load KNN Indices ---
    knn_cache_path = os.path.join(args.save_dir, f'knn_indices_RP_{args.retrieval_pool_size}.pt')
    
    if os.path.exists(knn_cache_path):
        print(f"Loading cached KNN indices from {knn_cache_path}...")
        cached_knn = torch.load(knn_cache_path, map_location=device)
        train_knn_idx = cached_knn['train']
        val_knn_idx = cached_knn['val']
        test_knn_idx = cached_knn.get('test', None)  # Load test if available
    else:
        print("Pre-computing KNN indices (Anti-Cheating enabled)...")
        
        # Debug: Check shapes
        print(f"  train_img shape: {train_img.shape}")
        print(f"  val_img shape: {val_img.shape}")
        
        # Ensure 2D [N, D]
        if train_img.dim() > 2:
            train_img = train_img.reshape(train_img.shape[0], -1)
            print(f"  Reshaped train_img to: {train_img.shape}")
        if val_img.dim() > 2:
            val_img = val_img.reshape(val_img.shape[0], -1)
            print(f"  Reshaped val_img to: {val_img.shape}")
        
        # Ensure normalized features for Cosine Sim
        train_norm = F.normalize(train_img, dim=1)
        val_norm = F.normalize(val_img, dim=1)
        
        # 1. Train vs Train (Anti-Cheating: Exclude self)
        # Use chunked computation to avoid OOM
        print("  Computing Train-Train Similarity (chunked to save memory)...")
        N_train = train_norm.shape[0]
        chunk_size = 500  # Process 500 samples at a time
        train_knn_idx = []
        
        for i in range(0, N_train, chunk_size):
            end_i = min(i + chunk_size, N_train)
            chunk = train_norm[i:end_i]  # [chunk_size, D]
            
            # Compute similarity for this chunk
            sims_chunk = torch.mm(chunk, train_norm.T)  # [chunk_size, N_train]
            
            # Get top K+1 (including self)
            _, top_idx_chunk = torch.topk(sims_chunk, k=args.retrieval_pool_size + 1, dim=1)
            
            # Remove self (first column is always self for diagonal elements)
            # For each row i, remove the index that equals i
            for local_idx, global_idx in enumerate(range(i, end_i)):
                row = top_idx_chunk[local_idx]
                # Find and remove self
                mask = row != global_idx
                filtered = row[mask][:args.retrieval_pool_size]  # Take first K after removing self
                train_knn_idx.append(filtered)
            
            # Free memory
            del sims_chunk, top_idx_chunk
            torch.cuda.empty_cache()
        
        train_knn_idx = torch.stack(train_knn_idx)  # [N_train, Pool]
        
        # 2. Val vs Train (Standard Retrieval)
        print("  Computing Val-Train Similarity (chunked)...")
        N_val = val_norm.shape[0]
        val_knn_idx = []
        
        for i in range(0, N_val, chunk_size):
            end_i = min(i + chunk_size, N_val)
            chunk = val_norm[i:end_i]  # [chunk_size, D]
            
            # Compute similarity
            sims_chunk = torch.mm(chunk, train_norm.T)  # [chunk_size, N_train]
            
            # Get top K
            _, top_idx_chunk = torch.topk(sims_chunk, k=args.retrieval_pool_size, dim=1)
            val_knn_idx.append(top_idx_chunk)
            
            # Free memory
            del sims_chunk, top_idx_chunk
            torch.cuda.empty_cache()
        
        val_knn_idx = torch.cat(val_knn_idx, dim=0)  # [N_val, Pool]
        
        # 3. Test vs Train (Standard Retrieval)
        print("  Computing Test-Train Similarity (chunked)...")
        N_test = F.normalize(test_img, dim=1).shape[0]
        test_norm = F.normalize(test_img, dim=1)
        test_knn_idx = []
        
        for i in range(0, N_test, chunk_size):
            end_i = min(i + chunk_size, N_test)
            chunk = test_norm[i:end_i]  # [chunk_size, D]
            
            # Compute similarity
            sims_chunk = torch.mm(chunk, train_norm.T)  # [chunk_size, N_train]
            
            # Get top K
            _, top_idx_chunk = torch.topk(sims_chunk, k=args.retrieval_pool_size, dim=1)
            test_knn_idx.append(top_idx_chunk)
            
            # Free memory
            del sims_chunk, top_idx_chunk
            torch.cuda.empty_cache()
        
        test_knn_idx = torch.cat(test_knn_idx, dim=0)  # [N_test, Pool]
        
        # Save
        torch.save({
            'train': train_knn_idx.cpu(), 
            'val': val_knn_idx.cpu(),
            'test': test_knn_idx.cpu()
        }, knn_cache_path)
        print(f"Saved KNN indices to {knn_cache_path}")
        
    train_knn_idx = train_knn_idx.to(device)
    val_knn_idx = val_knn_idx.to(device)
    
    # Handle test_knn_idx (may or may not be in cache)
    if os.path.exists(knn_cache_path):
        # Loaded from cache
        cached_knn = torch.load(knn_cache_path, map_location='cpu')
        if 'test' in cached_knn:
            test_knn_idx = cached_knn['test'].to(device)
        else:
            test_knn_idx = None
    else:
        # Just computed, already exists
        test_knn_idx = test_knn_idx.to(device)
    
    # Recompute test_knn_idx if not available
    if test_knn_idx is None:
        print("  Test KNN not in cache, recomputing...")
        test_norm = F.normalize(test_img, dim=1)
        train_norm = F.normalize(train_img, dim=1)
        test_knn_idx = []
        chunk_size = 500
        for i in range(0, test_norm.shape[0], chunk_size):
            end_i = min(i + chunk_size, test_norm.shape[0])
            chunk = test_norm[i:end_i]
            sims_chunk = torch.mm(chunk, train_norm.T)
            _, top_idx_chunk = torch.topk(sims_chunk, k=args.retrieval_pool_size, dim=1)
            test_knn_idx.append(top_idx_chunk)
            del sims_chunk, top_idx_chunk
            torch.cuda.empty_cache()
        test_knn_idx = torch.cat(test_knn_idx, dim=0).to(device)
    
    # Training Loop
    print(f"Starting Training (Epochs: {args.epochs})...")
    N_train = train_img.shape[0]
    
    best_val_acc = 0.0
    best_val_reward = -1e9 
    best_train_reward = -1e9
    
    for epoch in range(1, args.epochs + 1):
        # Shuffle
        perm = torch.randperm(N_train)
    
        epoch_loss = 0
        epoch_rew = 0
        # Aggregators for detailed breakdown
        epoch_r_item = 0
        epoch_r_discovery = 0
        epoch_r_purity = 0
        epoch_r_len = 0
        epoch_r_step = 0
        epoch_r_drop = 0
        
        pbar = tqdm(range(0, N_train, args.batch_size), desc=f"Epoch {epoch}")
        
        for i in pbar:
            idx = perm[i:i+args.batch_size] # Indices of the queries in Train Set
            
            q_img = train_img[idx]
            q_lbl = train_lbl[idx]
            
            # --- Optimized KNN Lookup ---
            # Lookup pre-computed neighbors for these query indices
            top_idx = train_knn_idx[idx] # [B, Pool]
            
            # Gather candidates
            batch_cand = train_img[top_idx] # [B, Pool, D]
            batch_lbl = train_lbl[top_idx]  # [B, Pool]
            batch_txt = train_txt[top_idx]  # [B, Pool, Dt]
            
            metrics = trainer.train_step(
                query_features=q_img,
                query_labels=q_lbl,
                candidate_features=batch_cand,
                candidate_text=batch_txt,
                candidate_labels=batch_lbl,
                train_kg=train_kg
            )
            
            epoch_loss += metrics['loss']
            epoch_rew += metrics['reward_mean']
            
            # Aggregate breakdown metrics
            epoch_r_item += metrics.get('r_item_mean', 0.0)
            epoch_r_discovery += metrics.get('r_discovery_mean', 0.0)
            epoch_r_purity += metrics.get('r_purity_mean', 0.0)
            epoch_r_len += metrics.get('r_len_penalty_mean', 0.0)
            epoch_r_step += metrics.get('r_step_mean', 0.0)
            epoch_r_drop += metrics.get('r_purity_drop_mean', 0.0)
            
            pbar.set_postfix({'rew': metrics['reward_mean'], 'acc': metrics['acc_mean']})
            
        N_batches = len(pbar)
        curr_train_rew = epoch_rew / N_batches
        writer.add_scalar('Train/Loss', epoch_loss / N_batches, epoch)
        writer.add_scalar('Train/Reward', curr_train_rew, epoch)
        print(f"Epoch {epoch} Avg Loss: {epoch_loss / N_batches:.4f} | Avg Reward: {curr_train_rew:.4f}")
        print(f"  -> Breakdown: Item={epoch_r_item/N_batches:.3f}, Disc={epoch_r_discovery/N_batches:.3f}, Purity={epoch_r_purity/N_batches:.3f}, Drop={epoch_r_drop/N_batches:.3f}, Len={epoch_r_len/N_batches:.3f}, Step={epoch_r_step/N_batches:.3f}")
        

        
        # Validation
        print(f"\nValidating Epoch {epoch}...")
        
        # --- Visualization (Top-3) - Every 5 Epochs Only ---
        if epoch == 1 or epoch % 5 == 0:
            print("Visualization (Random 3):")
            # Randomly select 3 indices
            viz_indices = torch.randperm(val_img.shape[0])[:3]
            
            for j in viz_indices:
                j = j.item() # convert to int
                # Validation Query index j
                q = val_img[j:j+1]
                l = val_lbl[j:j+1]
                l_name = info['idx_to_label'][l.item()]
                print(f"Query Label: {l_name} (ID: {l.item()})")
                
                # KNN Retrieval Lookup
                top_idx_v = val_knn_idx[j:j+1] # [1, Pool]
                
                c_img = train_img[top_idx_v]
                c_lbl = train_lbl[top_idx_v]
                c_txt = train_txt[top_idx_v]
                
                report, acc = evaluate_trajectory(
                    env, policy, trainer.reward_engine,
                    q, l, c_img, c_txt, c_lbl, train_kg, 
                    info['idx_to_label'], 
                    max_active_size=args.initial_k + 2,
                    device=device,
                    trainer=trainer
                )
                print(f"Model Report Sample {j}:")
                print(report)
                print("-" * 50)
                
        # --- Full Validation Loop (Every Epoch) ---
        print("Running Full Validation...")
        val_accs = []
        val_purities = []
        val_rewards = []
        
        # Use a simple loop
        N_val = val_img.shape[0]
        val_pbar = tqdm(range(0, N_val, args.batch_size), desc="Val")
        
        policy.eval()
        
        with torch.no_grad():
            for i in val_pbar:
                # Batch slice
                end = min(i + args.batch_size, N_val)
                if end <= i: break # Safety
                
                q_batch = val_img[i:end]
                l_batch = val_lbl[i:end]
                
                # KNN Lookup
                top_idx_batch = val_knn_idx[i:end] # [B, Pool]
                    
                c_img = train_img[top_idx_batch]
                c_lbl = train_lbl[top_idx_batch]
                c_txt = train_txt[top_idx_batch]
                
                # Run Episode (No Learning)
                state, _ = env.reset(q_batch, c_img, c_txt, c_lbl, l_batch, train_kg)
                
                # Capture initial labels from sorted state (top-K as purity baseline)
                initial_labels = state.candidate_labels[:, :args.initial_k].clone()

                for step in range(env.max_steps):
                    if state.done.all(): break
                    
                    active_vis, active_stats, active_lbls, buffer_feat, active_text = trainer._refine_features_for_policy(
                        state, c_img, q_batch, train_kg, query_labels=None
                    )
                    
                    logits = policy(
                        active_features=active_vis,
                        active_stats=active_stats,
                        active_mask=state.active_mask,
                        text_features=active_text,
                    )
                    
                    # Dynamic constraints (force exploration)

                    min_steps = 0
                    if args.initial_k >= 8:
                        min_steps = 3
                    elif args.initial_k >= 4:
                        min_steps = 1
                        
                    if step < min_steps:
                        logits[:, 0] = -1e9
                    
                    # Disable delete when only 1 item remains
                    current_counts = state.active_mask.sum(dim=1)
                    single_item_mask = (current_counts == 1)
                    if single_item_mask.any():
                        for b_idx in range(logits.shape[0]):
                            if single_item_mask[b_idx]:
                                logits[b_idx, 1:args.initial_k+1] = -1e9
                        
                    actions = logits.argmax(dim=1)
                    state = env.step(state, actions)
                
                # Final refinement and reward
                active_visual_final, _, _, _, _ = trainer._refine_features_for_policy(
                    state, c_img, q_batch, train_kg, query_labels=None
                )
                
                # Extract visual features only for reward (skip query node 0)
                refined_features = active_visual_final[:, 1:, :]
                refined_features_vis_only = refined_features[:, :, :visual_dim]
                

                
                # Reward calculation
                active_labels_final = torch.gather(state.candidate_labels, 1, state.active_indices)

                
                batch_rewards, batch_metrics = trainer.reward_engine.compute_reward(
                    query_labels=l_batch, 
                    active_labels=active_labels_final, 
                    active_mask=state.active_mask,
                    query_features=q_batch,
                    active_features=refined_features_vis_only
                )
                
                val_accs.extend(batch_metrics['accuracy'])
                val_purities.extend(batch_metrics['purity'])
                val_rewards.extend(batch_rewards.cpu().tolist())
            
            mean_val_acc = np.mean(val_accs)
            mean_val_purity = np.mean(val_purities)
            mean_val_reward = np.mean(val_rewards)
            print(f"Validation Results - Acc: {mean_val_acc:.4f}, Purity: {mean_val_purity:.4f}, Reward: {mean_val_reward:.4f}")
            writer.add_scalar('Val/Accuracy', mean_val_acc, epoch)
            writer.add_scalar('Val/Purity', mean_val_purity, epoch)
            writer.add_scalar('Val/Reward', mean_val_reward, epoch)
            
            # Full retrieval evaluation (5 metrics)
            print("Running Comprehensive Retrieval Evaluation (Hard/Soft Vote)...")
            
            # Collect ALL inputs for global metric computation
            all_logits_hard = []
            all_logits_soft = []
            all_targets = []
            

            
            policy.eval()
            
            with torch.no_grad():
                val_pbar_eval = tqdm(range(0, N_val, args.batch_size), desc="Val (Metrics)")
                for i in val_pbar_eval:
                    # Batch slice
                    end = min(i + args.batch_size, N_val)
                    if end <= i: break
                    
                    q_batch = val_img[i:end]
                    B_curr = q_batch.shape[0]
                    l_batch = val_lbl[i:end]
                    top_idx_batch = val_knn_idx[i:end]
                    
                    c_img = train_img[top_idx_batch]
                    c_lbl = train_lbl[top_idx_batch]
                    c_txt = train_txt[top_idx_batch]
                    
                    # Policy Rollout
                    state, _ = env.reset(q_batch, c_img, c_txt, c_lbl, l_batch, train_kg)
                    for step in range(env.max_steps):
                        if state.done.all(): break
                        active_vis, active_stats, active_lbls, buffer_feat, active_text = trainer._refine_features_for_policy(
                            state, c_img, q_batch, train_kg, query_labels=None
                        )
                        
                        # Gather active labels for adjacency construction
                        active_labels_batch = torch.gather(state.candidate_labels, 1, state.active_indices)
                        
                        logits = policy(
                            active_features=active_vis,
                            active_stats=active_stats,
                            active_mask=state.active_mask,
                            text_features=active_text,
                        )
                        if step < 3: logits[:, 0] = -1e9
                        actions = logits.argmax(dim=1)
                        state = env.step(state, actions)
                    
                    # Final feature gathering
                    active_visual_final, _, _, _, _ = trainer._refine_features_for_policy(
                        state, c_img, q_batch, train_kg, query_labels=None
                    )
                    refined_features = active_visual_final
                    
                    # Skip node 0 (query) — only use candidate nodes for voting
                    refined_features_cand = refined_features[:, 1:, :]  # [B, K, D_vis]
                    refined_features_vis_only = refined_features_cand[:, :, :visual_dim]
                    

                    
                    # Compute Vote Logits
                    # Refined Features vs Query Feature Similarity
                    # Sim [B, K]
                    q_norm = F.normalize(q_batch, dim=1).unsqueeze(1) # [B, 1, D]
                    feat_norm = F.normalize(refined_features_vis_only, dim=2)  # [B, K, D] (Use Visual Only)
                    sims = (q_norm * feat_norm).sum(dim=2) # [B, K]
                    
                    # Labels [B, K]
                    final_labels = torch.gather(state.candidate_labels, 1, state.active_indices)
                    
                    # Exclude deleted samples via active mask
                    active_masks = state.active_mask  # [B, K]
                    
                    num_classes = info.get('num_classes', 20)
                    
                    # Hard vote (with active mask filtering)
                    hv_logits = torch.zeros(B_curr, num_classes, device=device)
                    for b in range(B_curr):
                        valid_idx = torch.nonzero(active_masks[b]).squeeze(1)
                        if len(valid_idx) == 0:
                            continue
                        
                        valid_labels = final_labels[b, valid_idx]
                        valid_sims = sims[b, valid_idx]
                        

                        counts = torch.bincount(valid_labels, minlength=num_classes).float()
                        hv_logits[b] = counts
                        
                        # Tie-breaking with average similarity per class
                        avg_sims = torch.zeros(num_classes, device=device)
                        class_counts = torch.zeros(num_classes, device=device)
                        for idx in range(len(valid_labels)):
                            lbl = valid_labels[idx]
                            sim = valid_sims[idx]
                            avg_sims[lbl] += sim
                            class_counts[lbl] += 1
                        mask = class_counts > 0
                        avg_sims[mask] = avg_sims[mask] / class_counts[mask]
                        hv_logits[b] += (avg_sims + 2.0) * 1e-2 * mask.float()
                    
                    # Soft vote (similarity accumulation, with active mask filtering)
                    sv_logits = torch.zeros(B_curr, num_classes, device=device)
                    for b in range(B_curr):
                        valid_idx = torch.nonzero(active_masks[b]).squeeze(1)
                        if len(valid_idx) == 0:
                            continue
                        
                        valid_labels = final_labels[b, valid_idx]
                        valid_sims = sims[b, valid_idx]
                        
                        # Accumulate similarities per class
                        sum_scores = torch.zeros(num_classes, device=device)
                        class_counts = torch.zeros(num_classes, device=device)
                        sum_scores.scatter_add_(0, valid_labels, valid_sims)
                        class_counts.scatter_add_(0, valid_labels, torch.ones_like(valid_sims))
                        
                        # Accumulate similarities per class (sum strategy)
                        sum_scores = torch.zeros(num_classes, device=device)
                        sum_scores.scatter_add_(0, valid_labels, valid_sims)
                        sv_logits[b] = sum_scores
                    
                    all_logits_hard.append(hv_logits.cpu().numpy())
                    all_logits_soft.append(sv_logits.cpu().numpy())
                    all_targets.append(l_batch.cpu().numpy())

            # Combine Batches
            all_logits_hard = np.concatenate(all_logits_hard, axis=0)
            all_logits_soft = np.concatenate(all_logits_soft, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            # Compute Metrics
            metrics_hard = compute_all_metrics(all_logits_hard, all_targets, num_classes=num_classes)
            metrics_soft = compute_all_metrics(all_logits_soft, all_targets, num_classes=num_classes)
            
            print(f"\n[Epoch {epoch}] Retrieval Metrics:")
            print(f"  Hard Vote: ACC={metrics_hard['ACC']:.2f}, F1={metrics_hard['F1']:.2f}, Sensitivity={metrics_hard['Sensitivity']:.2f}")
            print(f"  Soft Vote: ACC={metrics_soft['ACC']:.2f}, F1={metrics_soft['F1']:.2f}, Sensitivity={metrics_soft['Sensitivity']:.2f}")
            
            # Save to JSON
            results_dict = {
                'epoch': epoch,
                'seed': args.seed,
                'hard_metrics': metrics_hard,
                'soft_metrics': metrics_soft
            }
            res_path = os.path.join(args.save_dir, f'results_seed{args.seed}_epoch{epoch}.json')
            with open(res_path, 'w') as f:
                json.dump(results_dict, f, indent=4)
            print(f"Saved metrics to {res_path}")
            
            # Save best model by validation accuracy
            current_val_acc = metrics_hard['ACC']
            if current_val_acc > best_val_acc:
                 best_val_acc = current_val_acc
                 trainer.save_checkpoint(os.path.join(args.save_dir, 'best_model.pt'))
                 print(f"🔥 New Best Model Saved! Acc: {best_val_acc:.2f} (Epoch {epoch})")
            print(f"Validation Done (Reward: {mean_val_reward:.4f}). Not saving checkpoint (Training-Reward Only).")
            
            # Memory cleanup
            del all_logits_hard, all_logits_soft, all_targets
            torch.cuda.empty_cache()
            import gc
            gc.collect()

    # === FINAL TEST SET EVALUATION ===
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET (Using Best Model)")
    print("="*80)
    
    # Load Best Model if exists
    best_model_path = os.path.join(args.save_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        print(f"Loading best model from {best_model_path}...")
        trainer.load_checkpoint(best_model_path)
    else:
        print("Best model not found, using latest state.")
    
    test_loader_for_eval = DataLoader(
        list(zip(test_img, test_lbl, test_knn_idx)),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: (
            torch.stack([item[0] for item in x]),
            torch.stack([item[1] for item in x]),
            torch.stack([item[2] for item in x])
        )
    )
    
    all_logits_hard = []
    all_logits_soft = []
    all_targets = []
    
    with torch.no_grad():
        for q_feat, l_batch, knn_idx_batch in tqdm(test_loader_for_eval, desc="Test Evaluation"):
            B_curr = q_feat.shape[0]
            
            # Gather candidates
            c_feat = train_img[knn_idx_batch]  # [B, Pool, D]
            c_lbl = train_lbl[knn_idx_batch]   # [B, Pool]
            c_txt = train_txt[knn_idx_batch]   # [B, Pool, D_txt]
            
            # Compute similarities
            q_feat_norm = F.normalize(q_feat, dim=1).unsqueeze(1)  # [B, 1, D]
            c_feat_norm = F.normalize(c_feat, dim=2)  # [B, Pool, D]
            sims = torch.bmm(q_feat_norm, c_feat_norm.transpose(1, 2)).squeeze(1)  # [B, Pool]
            
            # Initialize environment state
            state, _ = env.reset(
                query_features=q_feat,
                candidate_features=c_feat,
                candidate_text=c_txt,
                candidate_labels=c_lbl,
                train_kg=train_kg
            )
            
            # Run policy (no training, just inference)
            for step in range(args.max_steps):
                active_vis, active_stats, active_lbls, buffer_feat, active_text = trainer._refine_features_for_policy(
                    state, c_feat, q_feat, train_kg, query_labels=None
                )
                
                # Gather active labels
                active_labels_batch = torch.gather(state.candidate_labels, 1, state.active_indices)
                
                
                logits = trainer.policy_net(
                    active_features=active_vis,
                    active_stats=active_stats,
                    active_mask=state.active_mask,
                    text_features=active_text,
                )
                
                # Dynamic constraints
                min_steps = 0
                if args.initial_k >= 8:
                    min_steps = 3
                elif args.initial_k >= 4:
                    min_steps = 1
                    
                if step < min_steps:
                    logits[:, 0] = -1e9
                
                # Disable delete when only 1 item remains
                current_counts = state.active_mask.sum(dim=1)
                single_item_mask = (current_counts == 1)
                if single_item_mask.any():
                    for b_idx in range(logits.shape[0]):
                        if single_item_mask[b_idx]:
                            logits[b_idx, 1:args.initial_k+1] = -1e9
                
                probs = F.softmax(logits, dim=-1)
                actions = torch.argmax(probs, dim=-1)
                
                state = env.step(state, actions)
                done = state.done
                if done.all():
                    break
            
            # Final active set
            sims = torch.gather(state.candidate_sims, 1, state.active_indices)
            final_labels = torch.gather(state.candidate_labels, 1, state.active_indices)
            active_masks = state.active_mask
            
            num_classes = info.get('num_classes', 20)
            
            # Check for out-of-bounds labels
            if c_lbl.max() >= num_classes:
                print(f"WARNING: Found labels >= num_classes ({num_classes}). Max label: {c_lbl.max()}")
            
            # Hard Vote
            hv_logits = torch.zeros(B_curr, num_classes, device=device)
            for b in range(B_curr):
                valid_idx = torch.nonzero(active_masks[b]).squeeze(1)
                if len(valid_idx) == 0:
                    continue
                
                valid_labels = final_labels[b, valid_idx]
                valid_sims = sims[b, valid_idx]
                
                # Safety: Filter out-of-bounds labels
                mask_in_bounds = valid_labels < num_classes
                valid_labels = valid_labels[mask_in_bounds]
                valid_sims = valid_sims[mask_in_bounds]
                
                if len(valid_labels) == 0:
                    continue
                
                counts = torch.bincount(valid_labels, minlength=num_classes).float()
                if counts.shape[0] > num_classes:
                    counts = counts[:num_classes]
                    
                hv_logits[b] = counts
                
                # Tie-breaking
                avg_sims = torch.zeros(num_classes, device=device)
                class_counts = torch.zeros(num_classes, device=device)
                
                # Vectorized tie-breaking
                class_counts.scatter_add_(0, valid_labels, torch.ones_like(valid_labels, dtype=torch.float))
                avg_sims.scatter_add_(0, valid_labels, valid_sims)
                
                mask = class_counts > 0
                avg_sims[mask] = avg_sims[mask] / class_counts[mask]
                hv_logits[b] += (avg_sims + 2.0) * 1e-2 * mask.float()
            
            # Soft Vote
            sv_logits = torch.zeros(B_curr, num_classes, device=device)
            for b in range(B_curr):
                valid_idx = torch.nonzero(active_masks[b]).squeeze(1)
                if len(valid_idx) == 0:
                    continue
                
                valid_labels = final_labels[b, valid_idx]
                valid_sims = sims[b, valid_idx]
                
                # Safety: Filter out-of-bounds labels
                mask_in_bounds = valid_labels < num_classes
                valid_labels = valid_labels[mask_in_bounds]
                valid_sims = valid_sims[mask_in_bounds]
                
                if len(valid_labels) == 0:
                    continue
                
                sum_scores = torch.zeros(num_classes, device=device)
                # Sum of similarities (evidence accumulation)
                sum_scores.scatter_add_(0, valid_labels, valid_sims)
                sv_logits[b] = sum_scores
            
            all_logits_hard.append(hv_logits.cpu().numpy())
            all_logits_soft.append(sv_logits.cpu().numpy())
            all_targets.append(l_batch.cpu().numpy())
    
    # Combine and compute metrics
    all_logits_hard = np.concatenate(all_logits_hard, axis=0)
    all_logits_soft = np.concatenate(all_logits_soft, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    test_metrics_hard = compute_all_metrics(all_logits_hard, all_targets, num_classes=num_classes)
    test_metrics_soft = compute_all_metrics(all_logits_soft, all_targets, num_classes=num_classes)
    
    print("\n" + "="*80)
    print("FINAL TEST SET RESULTS")
    print("="*80)
    print(f"Hard Vote:")
    print(f"  ACC: {test_metrics_hard['ACC']:.2f}")
    print(f"  F1: {test_metrics_hard['F1']:.2f}")
    print(f"  Sensitivity: {test_metrics_hard['Sensitivity']:.2f}")
    print(f"\nSoft Vote:")
    print(f"  ACC: {test_metrics_soft['ACC']:.2f}")
    print(f"  F1: {test_metrics_soft['F1']:.2f}")
    print(f"  Sensitivity: {test_metrics_soft['Sensitivity']:.2f}")
    
    # Save final test results
    final_results = {
        'seed': args.seed,
        'test_hard_metrics': test_metrics_hard,
        'test_soft_metrics': test_metrics_soft
    }
    final_path = os.path.join(args.save_dir, f'final_test_results_seed{args.seed}.json')
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=4)
    print(f"\n✓ Saved final test results to {final_path}")
    print("="*80)
    
    # Save Checkpoint
    trainer.save_checkpoint(os.path.join(args.save_dir, 'latest.pt'))

if __name__ == '__main__':
    main()

