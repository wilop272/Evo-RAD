# Evo-RAD: Navigating Rare Retinal Disease Diagnosis via Self-Evolving Agentic Retrieval

This repository contains the official implementation of **Evo-RAD**, a self-evolving agentic retrieval framework for rare retinal disease diagnosis. The system uses a GCN-based policy network trained with Group Relative Policy Optimization (GRPO) to dynamically refine retrieval sets, improving diagnostic accuracy through iterative insert/delete actions guided by multi-dimensional rewards.

## Project Structure

```
GCN_GRPO_Retrieval/
├── main.py                     # Main training & evaluation script
├── gnn_train.py                # Relational GNN training (knowledge graph)
├── run.sh                      # Universal run script (train/multi_seed/multi_k/ablation)
├── requirements.txt            # Python dependencies
├── models/
│   ├── policy.py               # GCN-based PolicyNetwork
│   ├── dynamic_env.py          # Retrieval environment (reset/step)
│   ├── gat.py                  # Graph Attention Network
│   └── simple_gcn.py           # Simple GCN layer
├── training/
│   ├── grpo_trainer.py         # GRPO trainer with multi-trajectory exploration
│   └── reward.py               # Multi-dimensional reward engine
├── data/
│   ├── feature_extractor.py    # RetiZero feature extraction with caching
│   ├── bioclinical_bert.py     # BioClinicalBERT for semantic KG construction
│   ├── disease_tags.py         # Clinical disease tag definitions
│   └── dataset.py              # Dataset loading and management
├── utils/
│   ├── retrieval_metrics.py    # Hard/Soft vote metrics (ACC, F1, Sensitivity)
│   ├── standard_metrics.py     # Standardized metrics with multi-seed support
│   └── eval_utils.py           # Trajectory evaluation utilities
└── dataset/
    ├── Rare-20/                # 20-class rare disease dataset
    │   ├── train.csv
    │   ├── val.csv
    │   └── test.csv
    └── Ritina-31/              # 31-class retinal disease dataset
        ├── train.csv
        ├── val.csv
        └── test.csv
```

## Installation

```bash
pip install -r requirements.txt
```

### Dependencies
- PyTorch ≥ 1.12.0
- Transformers ≥ 4.20.0
- scikit-learn ≥ 1.0.0
- NumPy ≥ 1.21.0
- tqdm ≥ 4.60.0
- Pillow ≥ 8.0.0
- TorchVision ≥ 0.13.0
- TensorBoard ≥ 2.10.0

### Additional Requirements
- **RetiZero checkpoint**: Download the pre-trained RetiZero model weights
- **RetiZero source code**: The `zeroshot.modeling.model.CLIPRModel` module must be importable

## Dataset Preparation

Organize your dataset with the following structure:

```
<data_root>/
├── train.csv       # Columns: image_path, label
├── val.csv
├── test.csv
└── images/         # Fundus image files
```

Two benchmark datasets are provided:
- **Rare-20**: 20 rare retinal disease categories
- **Ritina-31**: 31 retinal disease categories

## Usage

### Single Training Run

```bash
bash run.sh --mode train \
    --data_root /path/to/dataset \
    --checkpoint /path/to/retizero_model.pth \
    --save_dir ./checkpoints/exp1
```

### Multi-Seed Experiments (3 seeds)

```bash
bash run.sh --mode multi_seed \
    --data_root /path/to/dataset \
    --checkpoint /path/to/retizero_model.pth \
    --save_dir ./checkpoints/multi_seed
```

### Multi-K Scale Study

```bash
bash run.sh --mode multi_k \
    --data_root /path/to/dataset \
    --checkpoint /path/to/retizero_model.pth \
    --k_values "4 8 16 32 64" \
    --save_dir ./checkpoints/multi_k
```

### Ablation Study

```bash
bash run.sh --mode ablation \
    --data_root /path/to/dataset \
    --checkpoint /path/to/retizero_model.pth \
    --save_dir ./checkpoints/ablation
```

Ablation configurations:
- `no_acc`: Disable accuracy reward
- `no_purity`: Disable purity reward
- `no_density`: Disable KG density reward
- `no_step_insert`: Disable step insert reward
- `no_step_delete`: Disable step delete reward

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--data_root` | (required) | Path to dataset root directory |
| `--retizero_checkpoint` | None | Path to RetiZero model checkpoint |
| `--epochs` | 20 | Number of training epochs |
| `--batch_size` | 32 | Training batch size |
| `--lr` | 1e-4 | Learning rate |
| `--initial_k` | 8 | Initial active set size (top-K) |
| `--retrieval_pool_size` | 100 | Total candidate pool size |
| `--max_steps` | 10 | Maximum policy steps per episode |
| `--num_trajectories` | 8 | Number of GRPO trajectories |
| `--kl_beta` | 0.04 | KL divergence penalty coefficient |
| `--ablation_mode` | full | PolicyNetwork input mode (`full`/`no_stats`/`no_dev`/`no_ego`) |
| `--seed` | 42 | Random seed |

## Evaluation Metrics

Results are reported using **3 standard metrics**:

| Metric | Description |
|---|---|
| **ACC** | Classification accuracy (%) |
| **F1** | Macro F1-score (%) |
| **Sensitivity** | Macro sensitivity / recall (%) |

Results are computed for both **Hard Vote** and **Soft Vote** strategies:
- **Hard Vote**: Majority voting among active candidates with similarity-based tie-breaking
- **Soft Vote**: Similarity-weighted accumulation per class

### Output Format

Results are saved as JSON files:
```json
{
    "seed": 1,
    "test_hard_metrics": {"ACC": 85.00, "F1": 82.50, "Sensitivity": 80.00},
    "test_soft_metrics": {"ACC": 86.00, "F1": 83.00, "Sensitivity": 81.50}
}
```

Multi-seed aggregated results report `mean ± std` across seeds.

