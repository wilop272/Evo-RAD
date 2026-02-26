import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="Train relational GNN on multi-edge graph.")
    parser.add_argument("--graph-path", required=True, help="Path to graph_multirel_trainval.pt")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--path-column", default="impath")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--out-dim", type=int, default=256, help="Final embedding size")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--edge-dropout",
        type=float,
        default=0.0,
        help="Probability of dropping each edge during training (applied per relation).",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--undirected", action="store_true", help="Make edges undirected (default: False)")
    return parser.parse_args()


def load_csv_paths(csv_path: str, column: str) -> List[str]:
    import pandas as pd

    df = pd.read_csv(csv_path)
    return df[column].astype(str).tolist()


def build_index_map(paths: List[str]) -> Dict[str, int]:
    return {p: i for i, p in enumerate(paths)}


def edge_list_to_tensors(
    edges: List[Tuple[int, int, float]],
    num_nodes: int,
    device: torch.device,
    undirected: bool = False,
):
    if not edges:
        return (
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.float, device=device),
        )
    src = torch.tensor([e[0] for e in edges], dtype=torch.long, device=device)
    dst = torch.tensor([e[1] for e in edges], dtype=torch.long, device=device)
    weight = torch.tensor([e[2] for e in edges], dtype=torch.float, device=device)
    if undirected:
        src = torch.cat([src, dst])
        dst = torch.cat([dst, src[: len(edges)]])
        weight = torch.cat([weight, weight[: len(edges)]])
    return src, dst, weight


class RelGraphLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_relations: int,
        activation_dropout: float,
        edge_dropout: float,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_dim, out_dim))
        self.root = nn.Linear(in_dim, out_dim)
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.act_dropout = nn.Dropout(activation_dropout)
        self.edge_dropout = edge_dropout
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, rel_edges):
        out = self.root(x)
        agg = torch.zeros_like(out)
        for ridx, (src, dst, w) in enumerate(rel_edges):
            if src.numel() == 0:
                continue
            src_rel, dst_rel = src, dst
            w_rel = w
            if self.training and self.edge_dropout > 0.0 and src_rel.numel() > 0:
                keep_prob = 1.0 - self.edge_dropout
                mask = torch.rand(src_rel.shape[0], device=src_rel.device) < keep_prob
                if mask.sum() == 0:
                    continue
                src_rel = src_rel[mask]
                dst_rel = dst_rel[mask]
                if w_rel.numel() > 0:
                    w_rel = w_rel[mask]
            msg = torch.matmul(x[src_rel], self.weight[ridx])
            if w_rel.numel() > 0:
                msg = msg * w_rel.unsqueeze(-1)
            agg.index_add_(0, dst_rel, msg)
        out = out + agg + self.bias
        return self.act_dropout(F.relu(out))


class RelationalGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_classes: int, num_relations: int, num_layers: int, dropout: float):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            rel_layer = RelGraphLayer(
                dims[i],
                dims[i + 1],
                num_relations,
                activation_dropout=dropout,
                edge_dropout=0.0,  # default; updated after init
            )
            layers.append(rel_layer)
        self.layers = nn.ModuleList(layers)
        self.classifier = nn.Linear(out_dim, num_classes)

    def forward(self, x: torch.Tensor, rel_edges: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
        h = x
        for layer in self.layers:
            h = layer(h, rel_edges)
        logits = self.classifier(h)
        return h, logits


def get_masks(paths: List[str], path_to_idx: Dict[str, int], train_csv: str, val_csv: str, column: str):
    train_paths = load_csv_paths(train_csv, column)
    val_paths = load_csv_paths(val_csv, column)
    train_idx = torch.tensor([path_to_idx[p] for p in train_paths if p in path_to_idx], dtype=torch.long)
    val_idx = torch.tensor([path_to_idx[p] for p in val_paths if p in path_to_idx], dtype=torch.long)
    return train_idx, val_idx


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    graph = torch.load(args.graph_path, map_location="cpu")
    vis = graph["visual_features"].float()
    txt = graph["text_features"].float()
    base_nodes = vis.shape[0]
    features = torch.cat([vis, txt], dim=-1)

    hyper_meta = graph.get("hyper_meta") or {}
    num_hyper = int(hyper_meta.get("num_hyper_nodes", 0))
    if num_hyper > 0:
        hyper_feats = torch.zeros(num_hyper, features.shape[1])
        features = torch.cat([features, hyper_feats], dim=0)

    labels = graph["labels"].long()
    num_classes = int(labels.max().item()) + 1

    # Extend labels with dummy entries for hyper nodes (ignored during loss)
    if num_hyper > 0:
        pad = torch.full((num_hyper,), -1, dtype=torch.long)
        labels = torch.cat([labels, pad], dim=0)

    all_paths = graph["paths"]
    path_to_idx = build_index_map(all_paths)
    train_idx, val_idx = get_masks(all_paths, path_to_idx, args.train_csv, args.val_csv, args.path_column)

    rel_keys = ["text", "image"]
    if graph["edges"].get("hyper"):
        rel_keys.append("hyper")
    rel_edges = []
    for key in rel_keys:
        rel_edges.append(edge_list_to_tensors(graph["edges"].get(key, []), features.shape[0], device, undirected=args.undirected))

    x = features.to(device)
    labels = labels.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)

    model = RelationalGNN(
        in_dim=x.shape[1],
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_classes=num_classes,
        num_relations=len(rel_edges),
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    for layer in model.layers:
        layer.edge_dropout = args.edge_dropout

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val = 0.0
    best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        embeddings, logits = model(x, rel_edges)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx])
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            _, logits_eval = model(x, rel_edges)
            train_acc = (logits_eval[train_idx].argmax(dim=-1) == labels[train_idx]).float().mean().item()
            val_acc = (logits_eval[val_idx].argmax(dim=-1) == labels[val_idx]).float().mean().item()

        print(f"Epoch {epoch}/{args.epochs} | Loss {loss.item():.4f} | Train Acc {train_acc:.4f} | Val Acc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
            }
            torch.save(best_state, output_dir / "best_gnn.pt")

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    # Save final embeddings from best checkpoint
    model.load_state_dict(best_state["model_state"])
    model.eval()
    with torch.no_grad():
        final_embeddings, final_logits = model(x, rel_edges)

    torch.save(
        {
            "embeddings": final_embeddings.cpu(),
            "logits": final_logits.cpu(),
            "train_idx": train_idx.cpu(),
            "val_idx": val_idx.cpu(),
            "base_nodes": base_nodes,
            "num_classes": num_classes,
            "rel_keys": rel_keys,
            "graph_path": args.graph_path,
        },
        output_dir / "node_embeddings.pt",
    )

    with open(output_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump({"best_val_acc": best_val, "best_epoch": best_state["epoch"]}, f, indent=2)

    print(f"[GNN] Training complete. Best val acc={best_val:.4f}. Artifacts saved to {output_dir}.")


if __name__ == "__main__":
    main()
