"""
Customer Tower Module for H&M Two-Tower Recommender System
Encodes customer features into dense embeddings
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional
import logging

from .embedding_layers import MultiFeatureEmbedding, TowerEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import feature config - handle both module and direct imports
try:
    from features.feature_config import CUSTOMER_CATEGORICAL_FEATURES, CUSTOMER_NUMERICAL_FEATURES
except ImportError:
    from ..features.feature_config import CUSTOMER_CATEGORICAL_FEATURES, CUSTOMER_NUMERICAL_FEATURES


class CustomerTower(nn.Module):
    """
    Customer tower of the two-tower model
    Processes customer features to generate customer embeddings
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 64,
                 dropout_rate: float = 0.1):
        """
        Initialize Customer Tower
        
        Args:
            embedding_dim: Dimension for feature embeddings
            hidden_dims: List of hidden layer dimensions
            output_dim: Final output embedding dimension
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Get feature configurations from shared config
        categorical_features = {}
        for name, config in CUSTOMER_CATEGORICAL_FEATURES.items():
            categorical_features[name] = config['num_categories']
        
        # Define categorical feature configurations for embeddings
        self.categorical_configs = {}
        for feature_name, num_categories in categorical_features.items():
            # Adaptive embedding dimension based on number of categories
            embed_dim = min(32, max(8, num_categories // 4))
            self.categorical_configs[feature_name] = (num_categories, embed_dim)
        
        # Count numerical features
        numerical_dim = len(CUSTOMER_NUMERICAL_FEATURES)
        
        # Multi-feature embedding layer
        self.feature_embedding = MultiFeatureEmbedding(
            categorical_configs=self.categorical_configs,
            numerical_dim=numerical_dim,
            embedding_dim=embedding_dim
        )
        
        # Tower encoder
        self.tower_encoder = TowerEncoder(
            input_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        )
        
        # L2 normalization for embedding
        self.normalize = True
        
        logger.info(f"CustomerTower initialized with output dimension: {output_dim}")
    
    def forward(self, 
                categorical_features: Dict[str, torch.Tensor],
                numerical_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through customer tower
        
        Args:
            categorical_features: Dict mapping feature names to tensors
            numerical_features: Optional tensor of numerical features
            
        Returns:
            Customer embeddings of shape (batch_size, output_dim)
        """
        # Get feature embeddings
        feature_embeddings = self.feature_embedding(
            categorical_features, numerical_features
        )
        
        # Pass through tower encoder
        customer_embeddings = self.tower_encoder(feature_embeddings)
        
        # L2 normalize embeddings for cosine similarity
        if self.normalize:
            customer_embeddings = nn.functional.normalize(
                customer_embeddings, p=2, dim=-1
            )
        
        return customer_embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the output embedding dimension"""
        return self.tower_encoder.encoder[-1].out_features


def create_customer_tower(config: Dict) -> CustomerTower:
    """
    Factory function to create customer tower from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CustomerTower instance
    """
    return CustomerTower(
        embedding_dim=config.get('embedding_dim', 128),
        hidden_dims=config.get('hidden_dims', [256, 128]),
        output_dim=config.get('output_dim', 64),
        dropout_rate=config.get('dropout_rate', 0.1)
    ) 