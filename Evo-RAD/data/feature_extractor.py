"""
Feature Extractor for GRPO-GNN System
Uses RetiZero (CLIPRModel + LoRA + BioClinicalBERT) as the vision-language backbone.
"""
from .disease_tags import DISEASE_CLINICAL_TAGS, get_disease_tags

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import pickle
from torchvision import transforms


class RetiZeroFeatureExtractor:
    """
    Extract frozen RetiZero features for images
    Uses CLIPRModel with LoRA adapters and BioClinicalBERT for text.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda', retizero_root: str = None, model_variant: str = 'openai/clip-vit-large-patch14'):
        self.device = device
        self.model_type = 'retizero'
        
        # Add RetiZero source to path if provided
        if retizero_root and retizero_root not in sys.path:
             if os.path.exists(retizero_root):
                 sys.path.insert(0, retizero_root)
             else:
                 print(f"Warning: RetiZero root {retizero_root} not found.")
        
        # Correct import path for CLIPRModel
        from zeroshot.modeling.model import CLIPRModel
        from transformers import CLIPImageProcessor, AutoTokenizer, AutoModel
        
        print(f"Loading RetiZero model from {checkpoint_path}")
        
        # Initialize model with BioClinicalBERT text encoder
        self.full_model = CLIPRModel(
            vision_type="lora",
            bert_type='emilyalsentzer/Bio_ClinicalBERT', 
            projection=True,
            norm_features=True,
            from_checkpoint=False,
            R=8
        )
        
        # Load checkpoint
        state_dict = torch.load(checkpoint_path, map_location=device)
        self.full_model.load_state_dict(state_dict, strict=False)
        self.full_model.to(device)
        self.full_model.eval()
        
        # Get vision model (includes LoRA adapters)
        self.vision_model = self.full_model.vision_model
        
        # Get text model (BioClinicalBERT)
        self.text_model = self.full_model.text_model
        
        # Preprocessor (CLIP image processor)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_variant)
        
        # Tokenizer (BioClinicalBERT)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
        except:
            print("Warning: Could not load BioClinicalBERT tokenizer. Text extraction might fail.")
            self.tokenizer = None

        print(f"✓ RetiZero loaded correctly (LoRA Enabled + BioClinicalBERT)")

    @torch.no_grad()
    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual features using LoRA-ViT"""
        images = images.to(self.device)
        
        # Forward pass through LoRA-adapted vision model
        features = self.vision_model(images)
        
        # Handle different output shapes
        if features.dim() == 3:
            # ViT output: [B, num_patches, D]
            # Take CLS token (first token)
            features = features[:, 0, :]  # [B, D]
        elif features.dim() == 2:
            # Already pooled: [B, D]
            pass
        else:
            raise ValueError(f"Unexpected vision model output shape: {features.shape}")
        
        features = features.float()
        features = F.normalize(features, dim=-1)
        return features
    
    @torch.no_grad()
    def extract_text_features(self, texts: list) -> torch.Tensor:
        """Extract text features using BioClinicalBERT"""
        if self.tokenizer is None:
            raise ValueError("BioClinicalBERT tokenizer not initialized")
            
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=77,
            return_tensors="pt",
            return_token_type_ids=False
        ).to(self.device)
        
        # BioBERT forward: extract the [CLS] token embedding
        # The text model may return different output formats depending on the wrapper
        
        outputs = self.text_model(**inputs)
        
        # Handle different output formats
        if isinstance(outputs, torch.Tensor):
            # Direct tensor output (already projected)
            embeddings = outputs
        elif hasattr(outputs, 'last_hidden_state'):
            # HuggingFace model output with last_hidden_state
            embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        elif hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            # HuggingFace model with pooler
            embeddings = outputs.pooler_output
        elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
            # Tuple/list output
            first_output = outputs[0]
            if first_output.dim() == 3:
                embeddings = first_output[:, 0, :]  # [B, seq_len, D] -> [B, D]
            elif first_output.dim() == 2:
                embeddings = first_output  # Already [B, D]
            else:
                raise ValueError(f"Unexpected output shape: {first_output.shape}")
        else:
            raise ValueError(f"Unexpected text_model output type: {type(outputs)}")
            
        # Project if RetiZero has a projection layer (it usually does for CLIP alignment)
        # If 'projection=True' was set, full_model has visual_projection and text_projection
        if hasattr(self.full_model, 'text_projection') and self.full_model.text_projection is not None:
            embeddings = embeddings @ self.full_model.text_projection
            
        embeddings = F.normalize(embeddings, dim=-1)
        return embeddings

    def extract_dataset_features(
        self, 
        dataloader, 
        cache_path: Optional[str] = None,
        description_template: str = "fundus image showing {}"
    ) -> Dict[str, torch.Tensor]:
        """
        Extract features for entire dataset with caching support
        
        Args:
            dataloader: PyTorch DataLoader
            cache_path: Optional path to cache features
            description_template: Template for text descriptions
        
        Returns:
            Dictionary with 'image_features', 'text_features', 'labels', 'image_paths'
        """
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached features from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        print("Extracting features...")
        all_image_features = []
        all_text_features = []
        all_labels = []
        all_paths = []
        
        for images, labels, paths in tqdm(dataloader, desc="Extracting features"):
            # Preprocess images using CLIP processor for consistency
            processed = []
            for img in images:
                # If img is a tensor, convert back to PIL for preprocessing
                if not isinstance(img, Image.Image):
                    # Convert tensor back to PIL if needed
                    # Convert tensor to PIL if needed
                        img = transforms.ToPILImage()(img)
                        
                inputs = self.image_processor(images=img, return_tensors="pt")
                processed.append(inputs["pixel_values"].squeeze(0))
            images = torch.stack(processed)
            
            img_features = self.extract_image_features(images)
            all_image_features.append(img_features.cpu())
            
            # Create text prompts: "<label>, with <tag1>, <tag2>, ..."
            label_names = [dataloader.dataset.get_label_name(label.item()) for label in labels]
            descriptions = []
            for name in label_names:
                tags = get_disease_tags(name)
                if tags:
                    descriptions.append(f"{name}, with {', '.join(tags)}")
                else:
                    descriptions.append(name)

            
            text_features = self.extract_text_features(descriptions)
            all_text_features.append(text_features.cpu())
            
            all_labels.append(labels)
            all_paths.extend(paths)
            
        features_dict = {
            'image_features': torch.cat(all_image_features, dim=0),
            'text_features': torch.cat(all_text_features, dim=0),
            'labels': torch.cat(all_labels, dim=0),
            'image_paths': all_paths
        }
        
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(features_dict, f)
                
        return features_dict


def extract_all_features(
    data_root: str, 
    cache_dir: str = 'retrieval_project/cache',
    retizero_checkpoint: Optional[str] = None
):
    """
    Extract and cache features for train/val/test sets using RetiZero.
    
    Args:
        data_root: Root directory containing dataset
        cache_dir: Directory to save cached features
        retizero_checkpoint: Path to RetiZero checkpoint
    
    Returns:
        train_features, val_features, test_features, info
    """
    from data.dataset import create_unified_dataloaders
    
    # Create dataloaders
    train_loader, val_loader, test_loader, info = create_unified_dataloaders(
        data_root, batch_size=32, num_workers=0
    )
    
    # Initialize feature extractor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Extract features for each split
    cache_dir = os.path.join(data_root, cache_dir)
    
    train_cache = os.path.join(cache_dir, 'train_features_retizero.pkl')
    val_cache = os.path.join(cache_dir, 'val_features_retizero.pkl')
    test_cache = os.path.join(cache_dir, 'test_features_retizero.pkl')
    
    # Check caches before loading the model
    if os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(test_cache):
        print("Found all cached features! Skipping model load.")
        with open(train_cache, 'rb') as f: train_features = pickle.load(f)
        with open(val_cache, 'rb') as f: val_features = pickle.load(f)
        with open(test_cache, 'rb') as f: test_features = pickle.load(f)
        return train_features, val_features, test_features, info

    if retizero_checkpoint is None:
        raise ValueError("RetiZero checkpoint path must be provided (--retizero_checkpoint)")
    extractor = RetiZeroFeatureExtractor(checkpoint_path=retizero_checkpoint, device=device)
    
    train_features = extractor.extract_dataset_features(
        train_loader,
        cache_path=train_cache
    )
    
    val_features = extractor.extract_dataset_features(
        val_loader,
        cache_path=val_cache
    )
    
    test_features = extractor.extract_dataset_features(
        test_loader,
        cache_path=test_cache
    )
    
    print(f"Test: {test_features['image_features'].shape}")
    
    # Clean up model to free GPU memory
    print("Cleaning up feature extractor model...")
    del extractor
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    return train_features, val_features, test_features, info

