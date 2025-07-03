"""
Two-Tower Model Implementation

This module implements the main Two-Tower model that combines customer and article towers
for similarity computation and training. The model supports both training and inference modes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, Union, Any
import pandas as pd

try:
    from .customer_tower import CustomerTower
    from .article_tower import ArticleTower
    from .tensor_converter import FeatureToTensorConverter
    from ..features.feature_config import get_customer_feature_config, get_article_feature_config
except ImportError:
    # For testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from models.customer_tower import CustomerTower
    from models.article_tower import ArticleTower
    from models.tensor_converter import FeatureToTensorConverter
    from features.feature_config import get_customer_feature_config, get_article_feature_config


class TwoTowerModel(nn.Module):
    """
    Main Two-Tower Model for recommendation system.
    
    This model combines customer and article towers to compute similarity scores
    between customers and articles for recommendation.
    """
    
    def __init__(
        self,
        customer_tower: CustomerTower,
        article_tower: ArticleTower,
        similarity_metric: str = 'cosine',
        temperature: float = 1.0
    ):
        """
        Initialize the Two-Tower Model.
        
        Args:
            customer_tower: The customer tower model
            article_tower: The article tower model
            similarity_metric: Similarity metric to use ('cosine' or 'dot')
            temperature: Temperature parameter for similarity scaling
        """
        super().__init__()
        
        self.customer_tower = customer_tower
        self.article_tower = article_tower
        self.similarity_metric = similarity_metric
        self.temperature = temperature
        
        # Ensure both towers have the same output dimension
        customer_dim = self.customer_tower.get_embedding_dim()
        article_dim = self.article_tower.get_embedding_dim()
        
        if customer_dim != article_dim:
            raise ValueError(
                f"Customer tower output dim ({customer_dim}) "
                f"must match article tower output dim ({article_dim})"
            )
        
        self.embedding_dim = customer_dim
        
    def forward(
        self,
        customer_features: Dict[str, torch.Tensor],
        article_features: Dict[str, torch.Tensor],
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the two-tower model.
        
        Args:
            customer_features: Dictionary of customer feature tensors
            article_features: Dictionary of article feature tensors
            return_embeddings: Whether to return individual embeddings
            
        Returns:
            If return_embeddings=False: similarity scores tensor
            If return_embeddings=True: (similarities, customer_embeddings, article_embeddings)
        """
        # Get embeddings from both towers
        customer_embeddings = self.customer_tower(
            customer_features.get('categorical', {}), 
            customer_features.get('numerical')
        )
        article_embeddings = self.article_tower(
            article_features.get('categorical', {}),
            article_features.get('numerical')
        )
        
        # Compute similarity scores
        similarities = self.compute_similarity(customer_embeddings, article_embeddings)
        
        if return_embeddings:
            return similarities, customer_embeddings, article_embeddings
        else:
            return similarities
    
    def compute_similarity(
        self,
        customer_embeddings: torch.Tensor,
        article_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity between customer and article embeddings.
        
        Args:
            customer_embeddings: Customer embedding tensor [batch_size, embedding_dim]
            article_embeddings: Article embedding tensor [batch_size, embedding_dim]
            
        Returns:
            Similarity scores tensor [batch_size, batch_size]
        """
        if self.similarity_metric == 'cosine':
            # Normalize embeddings
            customer_norm = F.normalize(customer_embeddings, p=2, dim=1)
            article_norm = F.normalize(article_embeddings, p=2, dim=1)
            
            # Compute cosine similarity
            similarities = torch.matmul(customer_norm, article_norm.transpose(0, 1))
            
        elif self.similarity_metric == 'dot':
            # Compute dot product similarity
            similarities = torch.matmul(customer_embeddings, article_embeddings.transpose(0, 1))
            
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        
        # Apply temperature scaling
        similarities = similarities / self.temperature
        
        return similarities
    
    def get_customer_embeddings(self, customer_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get customer embeddings only.
        
        Args:
            customer_features: Dictionary of customer feature tensors
            
        Returns:
            Customer embeddings tensor
        """
        return self.customer_tower(
            customer_features.get('categorical', {}), 
            customer_features.get('numerical')
        )
    
    def get_article_embeddings(self, article_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Get article embeddings only.
        
        Args:
            article_features: Dictionary of article feature tensors
            
        Returns:
            Article embeddings tensor
        """
        return self.article_tower(
            article_features.get('categorical', {}),
            article_features.get('numerical')
        )
    
    def predict_similarities(
        self,
        customer_data: pd.DataFrame,
        article_data: pd.DataFrame,
        batch_size: int = 1000
    ) -> torch.Tensor:
        """
        Predict similarities for pandas DataFrames (inference mode).
        
        Args:
            customer_data: Customer data DataFrame
            article_data: Article data DataFrame
            batch_size: Batch size for processing
            
        Returns:
            Similarity scores tensor
        """
        self.eval()
        
        # Convert dataframes to tensors
        customer_converter = FeatureToTensorConverter(get_customer_feature_config())
        article_converter = FeatureToTensorConverter(get_article_feature_config())
        
        customer_features = customer_converter.convert_dataframe(customer_data)
        article_features = article_converter.convert_dataframe(article_data)
        
        with torch.no_grad():
            similarities = self.forward(customer_features, article_features)
        
        return similarities
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'embedding_dim': self.embedding_dim,
            'similarity_metric': self.similarity_metric,
            'temperature': self.temperature,
            'customer_tower_params': sum(p.numel() for p in self.customer_tower.parameters()),
            'article_tower_params': sum(p.numel() for p in self.article_tower.parameters()),
            'total_params': sum(p.numel() for p in self.parameters())
        }


def create_two_tower_model(
    embedding_dim: int = 128,
    hidden_dims: Optional[list] = None,
    dropout_rate: float = 0.1,
    similarity_metric: str = 'cosine',
    temperature: float = 1.0
) -> TwoTowerModel:
    """
    Factory function to create a Two-Tower Model with default configurations.
    
    Args:
        embedding_dim: Dimension of the final embeddings (output_dim)
        hidden_dims: List of hidden layer dimensions
        dropout_rate: Dropout rate for regularization
        similarity_metric: Similarity metric to use ('cosine' or 'dot')
        temperature: Temperature parameter for similarity scaling
        
    Returns:
        Configured TwoTowerModel instance
    """
    if hidden_dims is None:
        hidden_dims = [512, 256]
    
    # Create customer tower
    customer_tower = CustomerTower(
        embedding_dim=256,  # Internal embedding dimension
        hidden_dims=hidden_dims,
        output_dim=embedding_dim,  # Final output dimension
        dropout_rate=dropout_rate
    )
    
    # Create article tower
    article_tower = ArticleTower(
        embedding_dim=256,  # Internal embedding dimension
        hidden_dims=hidden_dims,
        output_dim=embedding_dim,  # Final output dimension
        dropout_rate=dropout_rate
    )
    
    # Create two-tower model
    model = TwoTowerModel(
        customer_tower=customer_tower,
        article_tower=article_tower,
        similarity_metric=similarity_metric,
        temperature=temperature
    )
    
    return model


class TwoTowerLoss(nn.Module):
    """
    Loss function for Two-Tower Model training.
    
    Implements contrastive loss with in-batch negative sampling.
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Initialize the loss function.
        
        Args:
            margin: Margin for contrastive loss
        """
        super().__init__()
        self.margin = margin
        self.cross_entropy = nn.CrossEntropyLoss()
    
    def forward(self, similarities: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            similarities: Similarity scores tensor [batch_size, batch_size]
            labels: Binary labels tensor [batch_size, batch_size]
            
        Returns:
            Loss tensor
        """
        # Use diagonal as positive pairs (assuming batch contains positive pairs)
        batch_size = similarities.size(0)
        target = torch.arange(batch_size, device=similarities.device)
        
        # Compute cross-entropy loss
        loss = self.cross_entropy(similarities, target)
        
        return loss 