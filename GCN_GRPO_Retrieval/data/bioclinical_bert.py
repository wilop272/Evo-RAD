"""
BioClinicalBERT-based Text Feature Extractor for Edge Construction
Uses clinical tags to compute semantic similarity between diseases
"""
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from .disease_tags import DISEASE_CLINICAL_TAGS, get_disease_tags


class BioClinicalBERTExtractor:
    """
    Extract text features using BioClinicalBERT
    Compute edge weights based on semantic similarity of clinical tags
    """
    
    def __init__(self, model_name: str = 'emilyalsentzer/Bio_ClinicalBERT', device: str = 'cuda'):
        """
        Args:
            model_name: BioClinicalBERT model from HuggingFace
            device: 'cuda' or 'cpu'
        """
        self.device = device
        
        print(f"Loading BioClinicalBERT: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(device)
            self.model.eval()
            print(f"✓ BioClinicalBERT loaded successfully on {device}")
        except Exception as e:
            print(f"⚠️ Failed to load BioClinicalBERT ({model_name}): {e}")
            print("Attempting fallback to 'bert-base-uncased'...")
            try:
                fallback_name = "bert-base-uncased"
                self.tokenizer = AutoTokenizer.from_pretrained(fallback_name)
                self.model = AutoModel.from_pretrained(fallback_name).to(device)
                self.model.eval()
                print(f"✓ Fallback model ({fallback_name}) loaded successfully.")
            except Exception as e2:
                print(f"❌ Critical Error: Could not load fallback model either: {e2}")
                raise e
    
    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text using BioClinicalBERT
        
        Args:
            texts: List of text strings
        
        Returns:
            embeddings: [N, D] tensor of text embeddings
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        ).to(self.device)
        
        # Get embeddings
        outputs = self.model(**inputs)
        
        # Use [CLS] token embedding
        embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Normalize
        embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings
    
    @staticmethod
    def build_disease_prompt(disease_name: str) -> str:
        """
        Build a single descriptive prompt for a disease:
          "<Disease Name>, with <tag1>, <tag2>, <tag3>, ..."
        Falls back to just the disease name if no tags are available.
        """
        tags = get_disease_tags(disease_name)
        if tags:
            tag_str = ", ".join(tags)
            return f"{disease_name}, with {tag_str}"
        return disease_name

    def encode_disease_tags(self, disease_name: str) -> torch.Tensor:
        """
        Encode a disease into a single embedding using a rich textual prompt:
          "<Disease Name>, with <tag1>, <tag2>, <tag3>, ..."

        Args:
            disease_name: Name of the disease

        Returns:
            embedding: [D] normalized embedding
        """
        prompt = self.build_disease_prompt(disease_name)
        embedding = self.encode_text([prompt])[0]   # [D]
        embedding = F.normalize(embedding, dim=-1)
        return embedding
    
    def build_disease_embeddings_dict(self) -> Dict[str, torch.Tensor]:
        """
        Build embeddings for all diseases
        
        Returns:
            disease_embeddings: Dict mapping disease name to embedding
        """
        disease_embeddings = {}
        
        print("\nEncoding clinical tags for all diseases...")
        for disease_name in tqdm(DISEASE_CLINICAL_TAGS.keys(), desc="Encoding diseases"):
            embedding = self.encode_disease_tags(disease_name)
            disease_embeddings[disease_name] = embedding.cpu()
        
        return disease_embeddings
    
    def compute_disease_similarity_matrix(
        self,
        disease_names: List[str]
    ) -> torch.Tensor:
        """
        Compute pairwise similarity matrix between diseases
        
        Args:
            disease_names: List of disease names
        
        Returns:
            similarity_matrix: [N, N] similarity matrix
        """
        # Encode all diseases
        embeddings = []
        for disease_name in tqdm(disease_names, desc="Encoding diseases"):
            emb = self.encode_disease_tags(disease_name)
            embeddings.append(emb.cpu())
        
        embeddings = torch.stack(embeddings)  # [N, D]
        
        # Compute similarity
        similarity = torch.mm(embeddings, embeddings.T)  # [N, N]
        
        return similarity
    
    def build_semantic_graph(
        self,
        labels: torch.Tensor,
        label_to_idx: Dict[str, int],
        idx_to_label: Dict[int, str],
        threshold: float = 0.5,
        use_weighted: bool = True
    ) -> torch.Tensor:
        """
        Build graph based on semantic similarity of clinical tags
        
        Args:
            labels: [N] tensor of label indices
            label_to_idx: Mapping from label name to index
            idx_to_label: Mapping from index to label name
            threshold: Similarity threshold for edge creation
            use_weighted: If True, use similarity as edge weight; else binary
        
        Returns:
            adj_matrix: [N, N] adjacency matrix
        """
        N = labels.shape[0]
        
        # Get unique diseases
        unique_labels = sorted(set(labels.tolist()))
        unique_disease_names = [idx_to_label[idx] for idx in unique_labels]
        
        # Compute disease-level similarity
        disease_similarity = self.compute_disease_similarity_matrix(unique_disease_names)
        
        # Map to image-level adjacency matrix
        adj_matrix = torch.zeros(N, N)
        
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                
                label_i = labels[i].item()
                label_j = labels[j].item()
                
                # Find indices in unique_labels
                idx_i = unique_labels.index(label_i)
                idx_j = unique_labels.index(label_j)
                
                # Get similarity
                sim = disease_similarity[idx_i, idx_j].item()
                
                if sim > threshold:
                    if use_weighted:
                        adj_matrix[i, j] = sim
                    else:
                        adj_matrix[i, j] = 1.0
        
        return adj_matrix


if __name__ == '__main__':
    # Test BioClinicalBERT extractor
    print("Testing BioClinicalBERT Extractor...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = BioClinicalBERTExtractor(device=device)
    
    # Test encoding
    print("\n" + "=" * 60)
    print("Testing Disease Encoding")
    print("=" * 60)
    
    rp_emb = extractor.encode_disease_tags("retinitis pigmentosa")
    stargardt_emb = extractor.encode_disease_tags("Stargardt Disease")
    
    print(f"RP embedding shape: {rp_emb.shape}")
    print(f"Stargardt embedding shape: {stargardt_emb.shape}")
    
    # Compute similarity
    sim = torch.dot(rp_emb, stargardt_emb).item()
    print(f"\nSimilarity (RP vs Stargardt): {sim:.4f}")
    
    # Test similarity matrix
    print("\n" + "=" * 60)
    print("Testing Disease Similarity Matrix")
    print("=" * 60)
    
    test_diseases = [
        "retinitis pigmentosa",
        "Stargardt Disease",
        "Coats' disease",
        "retinoblastoma"
    ]
    
    sim_matrix = extractor.compute_disease_similarity_matrix(test_diseases)
    
    print("\nSimilarity Matrix:")
    print("                    RP      Stargardt  Coats  Retinoblastoma")
    for i, disease in enumerate(test_diseases):
        row_str = f"{disease[:20]:20s}"
        for j in range(len(test_diseases)):
            row_str += f" {sim_matrix[i, j]:.3f}"
        print(row_str)
    
    print("\n✓ BioClinicalBERT extractor test passed!")
