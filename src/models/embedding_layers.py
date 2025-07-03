"""
Embedding Layers Module for H&M Two-Tower Recommender System
Contains shared embedding components used by both customer and article towers
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CategoricalEmbedding(nn.Module):
    """
    Embedding layer for categorical features with optional dimension reduction
    """
    
    def __init__(self, num_categories: int, embedding_dim: int, 
                 output_dim: Optional[int] = None):
        """
        Initialize categorical embedding layer
        
        Args:
            num_categories: Number of unique categories
            embedding_dim: Dimension of embeddings
            output_dim: Optional output dimension after linear projection
        """
        super().__init__()
        self.embedding = nn.Embedding(num_categories, embedding_dim)
        
        # Optional dimension reduction
        self.projection = None
        if output_dim and output_dim != embedding_dim:
            self.projection = nn.Linear(embedding_dim, output_dim)
            
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights"""
        nn.init.xavier_uniform_(self.embedding.weight)
        if self.projection:
            nn.init.xavier_uniform_(self.projection.weight)
            nn.init.zeros_(self.projection.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of category indices
            
        Returns:
            Embedded tensor
        """
        embedded = self.embedding(x)
        if self.projection:
            embedded = self.projection(embedded)
        return embedded


class NumericalEmbedding(nn.Module):
    """
    Embedding layer for numerical features with normalization
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 normalize: bool = True):
        """
        Initialize numerical embedding layer
        
        Args:
            input_dim: Number of numerical features
            output_dim: Output embedding dimension
            normalize: Whether to apply batch normalization
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.normalize = normalize
        
        if normalize:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize layer weights"""
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of numerical features
            
        Returns:
            Embedded tensor
        """
        embedded = self.linear(x)
        if self.normalize and embedded.shape[0] > 1:
            embedded = self.batch_norm(embedded)
        return embedded


class MultiFeatureEmbedding(nn.Module):
    """
    Combines multiple categorical and numerical embeddings
    """
    
    def __init__(self, categorical_configs: Dict[str, Tuple[int, int]], 
                 numerical_dim: int, embedding_dim: int):
        """
        Initialize multi-feature embedding layer
        
        Args:
            categorical_configs: Dict mapping feature names to (num_categories, embed_dim)
            numerical_dim: Number of numerical features
            embedding_dim: Final output embedding dimension
        """
        super().__init__()
        
        # Create categorical embeddings
        self.categorical_embeddings = nn.ModuleDict()
        total_cat_dim = 0
        
        for feature_name, (num_cats, embed_dim) in categorical_configs.items():
            self.categorical_embeddings[feature_name] = CategoricalEmbedding(
                num_cats, embed_dim
            )
            total_cat_dim += embed_dim
        
        # Create numerical embedding
        self.numerical_embedding = None
        if numerical_dim > 0:
            self.numerical_embedding = NumericalEmbedding(
                numerical_dim, embedding_dim // 2
            )
            total_cat_dim += embedding_dim // 2
        
        # Final projection to desired embedding dimension
        self.final_projection = nn.Linear(total_cat_dim, embedding_dim)
        self.dropout = nn.Dropout(0.1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection weights"""
        nn.init.xavier_uniform_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)
    
    def forward(self, categorical_features: Dict[str, torch.Tensor],
                numerical_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass combining all features
        
        Args:
            categorical_features: Dict mapping feature names to tensors
            numerical_features: Optional tensor of numerical features
            
        Returns:
            Combined embedding tensor
        """
        embeddings = []
        
        # Process categorical features
        for feature_name, feature_tensor in categorical_features.items():
            if feature_name in self.categorical_embeddings:
                embedded = self.categorical_embeddings[feature_name](feature_tensor)
                # Squeeze to remove extra dimension from (batch_size, 1, embed_dim) -> (batch_size, embed_dim)
                if embedded.dim() == 3:
                    embedded = embedded.squeeze(1)
                embeddings.append(embedded)
        
        # Process numerical features
        if numerical_features is not None and self.numerical_embedding is not None:
            num_embedded = self.numerical_embedding(numerical_features)
            embeddings.append(num_embedded)
        
        # Concatenate all embeddings
        combined = torch.cat(embeddings, dim=-1)
        
        # Final projection
        output = self.final_projection(combined)
        output = self.dropout(output)
        
        return output


class TowerEncoder(nn.Module):
    """
    Base encoder architecture for both customer and article towers
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dropout_rate: float = 0.1):
        """
        Initialize tower encoder
        
        Args:
            input_dim: Input embedding dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Final output dimension
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Build hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Final output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize encoder weights"""
        for module in self.encoder:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder
        
        Args:
            x: Input embedding tensor
            
        Returns:
            Encoded representation
        """
        return self.encoder(x) 