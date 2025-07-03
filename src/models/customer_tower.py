"""
Customer Tower Module for H&M Two-Tower Recommender System
Encodes customer features into dense embeddings
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

from .embedding_layers import MultiFeatureEmbedding, TowerEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerTower(nn.Module):
    """
    Customer tower of the two-tower model
    Processes customer features to generate customer embeddings
    """
    
    def __init__(self, 
                 categorical_features: Dict[str, int],
                 numerical_dim: int,
                 embedding_dim: int = 128,
                 hidden_dims: List[int] = [256, 128],
                 output_dim: int = 64,
                 dropout_rate: float = 0.1):
        """
        Initialize Customer Tower
        
        Args:
            categorical_features: Dict mapping feature names to number of categories
            numerical_dim: Number of numerical features
            embedding_dim: Dimension for feature embeddings
            hidden_dims: List of hidden layer dimensions
            output_dim: Final output embedding dimension
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Define categorical feature configurations
        self.categorical_configs = {}
        for feature_name, num_categories in categorical_features.items():
            # Adaptive embedding dimension based on number of categories
            embed_dim = min(32, max(8, num_categories // 4))
            self.categorical_configs[feature_name] = (num_categories, embed_dim)
        
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


class CustomerFeatureProcessor:
    """
    Preprocesses raw customer features for the customer tower
    """
    
    def __init__(self, feature_configs: Dict[str, Dict]):
        """
        Initialize feature processor
        
        Args:
            feature_configs: Configuration for each feature
        """
        self.feature_configs = feature_configs
        self.categorical_mappings = {}
        self._build_mappings()
    
    def _build_mappings(self):
        """Build mappings for categorical features"""
        for feature_name, config in self.feature_configs.items():
            if config.get('type') == 'categorical':
                # Create mapping from raw values to indices
                self.categorical_mappings[feature_name] = {
                    value: idx for idx, value in enumerate(config.get('values', []))
                }
    
    def process_features(self, 
                        raw_features: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Process raw features into model-ready format
        
        Args:
            raw_features: Dictionary of raw feature tensors
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        categorical_features = {}
        numerical_features = []
        
        for feature_name, feature_tensor in raw_features.items():
            config = self.feature_configs.get(feature_name, {})
            
            if config.get('type') == 'categorical':
                # Map categorical values to indices
                if feature_name in self.categorical_mappings:
                    # Apply mapping
                    categorical_features[feature_name] = feature_tensor
                else:
                    # If no mapping, assume already indices
                    categorical_features[feature_name] = feature_tensor
            
            elif config.get('type') == 'numerical':
                # Normalize numerical features if needed
                if config.get('normalize', True):
                    mean = config.get('mean', 0.0)
                    std = config.get('std', 1.0)
                    normalized = (feature_tensor - mean) / (std + 1e-6)
                    numerical_features.append(normalized)
                else:
                    numerical_features.append(feature_tensor)
        
        # Stack numerical features
        if numerical_features:
            numerical_tensor = torch.stack(numerical_features, dim=-1)
        else:
            numerical_tensor = None
        
        return categorical_features, numerical_tensor


def create_customer_tower(config: Dict) -> CustomerTower:
    """
    Factory function to create customer tower from configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        CustomerTower instance
    """
    # Extract H&M specific customer features
    categorical_features = {
        'age_group': config.get('age_groups', 10),
        'club_member_status': 2,  # Binary
        'fashion_news_active': 2,  # Binary
        'customer_lifecycle_stage': config.get('lifecycle_stages', 5),
        'favorite_department': config.get('num_departments', 20),
        'favorite_color': config.get('num_colors', 50)
    }
    
    # Number of numerical features
    numerical_dim = config.get('numerical_features', 15)
    
    return CustomerTower(
        categorical_features=categorical_features,
        numerical_dim=numerical_dim,
        embedding_dim=config.get('embedding_dim', 128),
        hidden_dims=config.get('hidden_dims', [256, 128]),
        output_dim=config.get('output_dim', 64),
        dropout_rate=config.get('dropout_rate', 0.1)
    ) 