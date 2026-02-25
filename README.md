# Evo-RAD: Navigating Rare Retinal Disease Diagnosis via Self-Evolving Agentic Retrieval

A GRPO (Group Relative Policy Optimization) driven approach that uses a Graph Convolutional Network (GCN) to iteratively refine a retrieval set for fundus image classification. The policy learns to **delete** noisy neighbors and **insert** better candidates, evolving the retrieval graph until a purified set yields an accurate diagnosis via voting.

## Architecture

```
Query Image ──► Image Encoder ──► KNN Retrieval ──► Initial Graph G₀
                                                        │
                    ┌───────────────────────────────────┘
                    ▼
              ┌─────────────┐
              │  GCN Policy  │  ◄── Text Adjacency Matrix
              │  (K+1 nodes) │      (BioClinicalBERT embeddings)
              └──────┬──────┘
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
     DELETE       INSERT        STOP
   (remove node) (add node)  (terminate)
        │            │            │
        └────────────┼────────────┘
                     ▼
              Purified Graph G* ──► Voting ──► Predicted Label
```

**Key design**: The query image is included as **node 0** in the GCN graph (visual features only, no text label) so the policy can reason about query–candidate relationships during message passing.

## Project Structure

```
GCN_GRPO_Retrieval/
├── main.py                  # Entry point: training, validation, testing
├── run.sh                   # Universal shell script (all modes)
├── gnn_train.py             # Standalone GNN training (alternative)
│
├── models/
│   ├── dynamic_env.py       # Retrieval environment (state, actions)
│   ├── policy.py            # GCN-based policy network
│   ├── simple_gcn.py        # GCN layer implementation
│   └── gat.py               # GAT variant (optional)
│
├── training/
│   ├── grpo_trainer.py      # GRPO training loop + trajectory sampling
│   └── reward.py            # Reward engine (accuracy, purity, KG density)
│
├── data/
│   ├── dataset.py           # Unified dataset loader (train/val/test splits)
│   ├── feature_extractor.py # RetiZero / EyeCLIP feature extraction
│   ├── bioclinical_bert.py  # BioClinicalBERT text encoder + KG builder
│   └── disease_tags.py      # Curated clinical tags per disease
│
└── utils/
    ├── eval_utils.py         # Trajectory visualization
    ├── retrieval_metrics.py  # Hard/Soft vote + 5 standard metrics
    └── standard_metrics.py   # ACC, F1, AUC, Sensitivity, Specificity
```

## Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.12
- transformers (for BioClinicalBERT / CLIP)
- scikit-learn, numpy, tqdm

## Dataset Preparation

Organize your fundus image dataset as follows:

```
<data_root>/
├── train/
│   ├── <class_name_1>/
│   │   ├── image1.jpg
│   │   └── ...
│   └── <class_name_N>/
├── val/
│   └── ...
└── test/
    └── ...
```

Or provide a `data.csv` file with columns `image_path` and `label` under `<data_root>/`.

## Quick Start

### Single Training Run

```bash
bash run.sh --mode train \
    --data_root /path/to/dataset \
    --model_type retizero \
    --checkpoint /path/to/RetiZero.pth
```

### Multi-Seed Experiment (3 seeds + aggregation)

```bash
bash run.sh --mode multi_seed \
    --data_root /path/to/dataset \
    --model_type retizero \
    --checkpoint /path/to/RetiZero.pth \
    --save_dir ./checkpoints/experiment_1
```

### Multi-K Scale Study (K = 4, 8, 16, 32, 64)

```bash
bash run.sh --mode multi_k \
    --data_root /path/to/dataset \
    --model_type retizero \
    --checkpoint /path/to/RetiZero.pth
```

### Ablation Study

```bash
bash run.sh --mode ablation \
    --data_root /path/to/dataset \
    --model_type retizero \
    --checkpoint /path/to/RetiZero.pth
```

### Using EyeCLIP (no checkpoint needed)

```bash
bash run.sh --mode train \
    --model_type eyeclip \
    --data_root /path/to/dataset
```

## Configuration

All hyperparameters can be set via `run.sh` flags or directly in `main.py`:

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `train` | `train` / `multi_seed` / `multi_k` / `ablation` |
| `--model_type` | `retizero` | `retizero` or `eyeclip` |
| `--data_root` | — | Path to dataset root |
| `--checkpoint` | — | Path to model checkpoint (RetiZero only) |
| `--epochs` | 20 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--initial_k` | 8 | Initial retrieval set size |
| `--retrieval_pool_size` | 100 | Candidate pool size |
| `--max_steps` | 10 | Max evolution steps per episode |
| `--num_trajectories` | 8 | GRPO trajectories per sample |
| `--lr` | 1e-4 | Learning rate |
| `--seeds` | `"1 2 3"` | Seeds for multi-seed experiments |

## Evaluation Metrics

Results are reported with both **Hard Vote** and **Soft Vote** strategies across 5 metrics:

| Metric | Description |
|--------|-------------|
| ACC | Classification accuracy |
| F1 | Macro F1 score |
| Sensitivity | Macro recall |

Results for each seed are saved as JSON in the `--save_dir`. Multi-seed runs automatically aggregate to `aggregated_results.json` with mean ± std.

## Output Structure

```
checkpoints/experiment_1/
├── seed1/
│   ├── best_model.pt
│   ├── latest.pt
│   ├── final_test_results_seed1.json
│   ├── training.log
│   └── results_seed1_epoch*.json
├── seed2/ ...
├── seed3/ ...
└── aggregated_results.json
```
