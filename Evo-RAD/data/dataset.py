"""
Unified Data Layer for Evo-RAD
Handles consistent label mapping across splits and semantic graph construction.
"""
import os
import csv
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from collections import Counter
from tqdm import tqdm

try:
    from .bioclinical_bert import BioClinicalBERTExtractor
except ImportError:
    # Fallback for when running not as a module (e.g. debugging)
    from bioclinical_bert import BioClinicalBERTExtractor

class FundusUnifiedDataset(Dataset):
    """
    Dataset that enforces a global label mapping.
    """
    def __init__(
        self,
        csv_path: str,
        data_root: str,
        label_to_idx: Optional[Dict[str, int]] = None,
        transform = None,
        image_root: Optional[str] = None
    ):
        """
        Args:
            csv_path: Path to the CSV file (train, val, or test).
            data_root: Root directory of the dataset.
            label_to_idx: Global label mapping. If None, it will be computed (should only happen for Train).
            transform: Image augmentations/transforms.
            image_root: Override for image directory.
        """
        self.csv_path = csv_path
        self.data_root = data_root
        self.image_root = image_root or data_root
        self.transform = transform
        
        # Load Data
        self.samples = []
        self.labels_raw = []
        
        with open(csv_path, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            # Infer columns
            fieldnames = reader.fieldnames
            col_map = self._infer_columns(fieldnames)
            
            for row in reader:
                img_name = row[col_map['image']]
                label = row[col_map['label']]
                path = row[col_map['path']]
                
                self.samples.append({
                    'image': img_name,
                    'label': label,
                    'path': path
                })
                self.labels_raw.append(label)
                
        # Handle Label Mapping
        if label_to_idx is None:
            # this should be the training set
            unique_labels = sorted(list(set(self.labels_raw)))
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
            
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
        # Filter samples that have labels NOT in the map (e.g. unseen classes in test? usually shouldn't happen for closed set)
        # But for robustness, we keep them but maybe warn? Or mapping them to -1?
        # The prompt implies "Enforce this mapping on val and test loaders to prevent label drift."
        # We will assume all valid classes are in Train.
        
        self.valid_indices = []
        unknown_labels = set()
        for i, sample in enumerate(self.samples):
            if sample['label'] in self.label_to_idx:
                self.valid_indices.append(i)
            else:
                unknown_labels.add(sample['label'])
        
        if unknown_labels:
            print(f"Warning: {len(unknown_labels)} labels in {csv_path} are not in the global label map and will be skipped: {list(unknown_labels)[:5]}...")
            
    def _infer_columns(self, fieldnames):
        # reuse inference logic
        normalized = [name.lower().strip() for name in fieldnames]
        def match(candidates):
            for cand in candidates:
                if cand in normalized:
                    return fieldnames[normalized.index(cand)]
            return None
            
        col_map = {
            'image': match(['image', 'img', 'img_name', 'image_name']),
            'label': match(['label', 'labels', 'disease', 'condition']),
            'path': match(['impath', 'path', 'filepath', 'relpath'])
        }
        
        # Fallbacks
        if not col_map['image'] and col_map['path']: col_map['image'] = col_map['path']
        if not col_map['path'] and col_map['image']: col_map['path'] = col_map['image']
        
        if not col_map['label'] or not col_map['path']:
             raise ValueError(f"Could not infer columns from {fieldnames}")
             
        return col_map

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        item = self.samples[real_idx]
        
        # Load Image
        path = self._resolve_path(item['path'])
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224))
            
        if self.transform:
            image = self.transform(image)
            
        label_idx = self.label_to_idx[item['label']]
        
        return image, label_idx, path

    def _resolve_path(self, impath):
        if os.path.isabs(impath): return impath
        # Normalized handling
        norm_path = impath.replace("\\", "/").lstrip("./")
        # Logic to handle "images/" prefix if needed
        base = self.image_root
        if os.path.basename(base) == "images" and norm_path.startswith("images/"):
             norm_path = norm_path.replace("images/", "", 1)
        return os.path.join(base, norm_path)
        
    def get_label_name(self, idx):
        return self.idx_to_label.get(idx, "Unknown")

class DataManager:
    """
    Manager to handle Train/Val/Test loading with Unified Label Map.
    """
    def __init__(self, data_root: str, image_root: str = None):
        self.data_root = data_root
        
        # Smart detection: check if data_root IS the dataset folder
        if os.path.exists(os.path.join(data_root, "train.csv")):
            self.fundus_root = data_root
        else:
            # Try common subfolder patterns
            possible_subfolders = ["Fundus_Eval_Top20", "data", "dataset"]
            found = False
            for subfolder in possible_subfolders:
                candidate = os.path.join(data_root, subfolder)
                if os.path.exists(os.path.join(candidate, "train.csv")):
                    self.fundus_root = candidate
                    found = True
                    print(f"Found dataset in subfolder: {subfolder}")
                    break
            
            if not found:
                raise FileNotFoundError(
                    f"Could not find train.csv in {data_root} or common subfolders. "
                    f"Please ensure your dataset has train.csv, val.csv, and test.csv in the root directory."
                )
            
        self.image_root = image_root or self.fundus_root
        
        self.label_to_idx = None
        self.idx_to_label = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
    def load_datasets(self, transform=None):
        # 1. Load Train to establish Label Map
        train_csv = os.path.join(self.fundus_root, "train.csv")
        self.train_dataset = FundusUnifiedDataset(
            train_csv, self.data_root, label_to_idx=None, transform=transform, image_root=self.image_root
        )
        self.label_to_idx = self.train_dataset.label_to_idx
        self.idx_to_label = self.train_dataset.idx_to_label
        
        print(f"Global Label Map established with {len(self.label_to_idx)} classes.")
        
        # 2. Load Val and Test using the SAME map
        val_csv = os.path.join(self.fundus_root, "val.csv")
        test_csv = os.path.join(self.fundus_root, "test.csv")
        
        self.val_dataset = FundusUnifiedDataset(
            val_csv, self.data_root, label_to_idx=self.label_to_idx, transform=transform, image_root=self.image_root
        )
        self.test_dataset = FundusUnifiedDataset(
            test_csv, self.data_root, label_to_idx=self.label_to_idx, transform=transform, image_root=self.image_root
        )
        
        return self.train_dataset, self.val_dataset, self.test_dataset

    def build_knowledge_graph(self, device='cpu', threshold=0.5):
        """
        Build Semantic Prior Graph (Adjacency Matrix) using BioClinicalBERT on Train Set Labels.
        """
        print("Building Knowledge Graph from Training Set Labels...")
        extractor = BioClinicalBERTExtractor(device=device)
        
        # We need to compute similarity between ALL classes in the label map
        sorted_labels = [self.idx_to_label[i] for i in range(len(self.idx_to_label))]
        
        # Create a Class-to-Class Similarity Matrix
        sim_matrix = extractor.compute_disease_similarity_matrix(sorted_labels).cpu() # [C, C]
        
        # Thresholding
        adj = (sim_matrix > threshold).float()
        
        # Ensure self-loops
        adj.fill_diagonal_(1.0)
        
        print(f"Knowledge Graph Built. Shape: {adj.shape}")
        return adj, sim_matrix

def create_unified_dataloaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_root: Optional[str] = None
):
    manager = DataManager(data_root, image_root)
    manager.load_datasets() # No transforms here, usually transforms happen in feature extraction or training
    
    # Collate function
    def collate_fn(batch):
        images = [b[0] for b in batch]
        labels = torch.tensor([b[1] for b in batch])
        paths = [b[2] for b in batch]
        # Stack images if they are tensors, else list
        if len(images) > 0 and isinstance(images[0], torch.Tensor):
            images = torch.stack(images)
        return images, labels, paths

    train_loader = DataLoader(manager.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(manager.val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(manager.test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    
    info = {
        "num_classes": len(manager.label_to_idx),
        "label_to_idx": manager.label_to_idx,
        "idx_to_label": manager.idx_to_label,
        "train_size": len(manager.train_dataset),
        "val_size": len(manager.val_dataset),
        "test_size": len(manager.test_dataset),
        "manager": manager # Return manager to access graph building
    }
    
    return train_loader, val_loader, test_loader, info

