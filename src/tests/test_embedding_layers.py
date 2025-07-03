"""
Test cases for embedding layers module
"""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.embedding_layers import (
    CategoricalEmbedding, 
    NumericalEmbedding,
    MultiFeatureEmbedding,
    TowerEncoder
)


class TestCategoricalEmbedding:
    """Test CategoricalEmbedding layer"""
    
    def test_initialization(self):
        """Test layer initialization"""
        layer = CategoricalEmbedding(num_categories=100, embedding_dim=32)
        assert layer.embedding.num_embeddings == 100
        assert layer.embedding.embedding_dim == 32
        assert layer.projection is None
    
    def test_with_projection(self):
        """Test layer with projection"""
        layer = CategoricalEmbedding(num_categories=100, embedding_dim=32, output_dim=16)
        assert layer.projection is not None
        assert layer.projection.out_features == 16
    
    def test_forward_pass(self):
        """Test forward pass"""
        layer = CategoricalEmbedding(num_categories=100, embedding_dim=32)
        input_tensor = torch.randint(0, 100, (10,))
        output = layer(input_tensor)
        assert output.shape == (10, 32)


class TestNumericalEmbedding:
    """Test NumericalEmbedding layer"""
    
    def test_initialization(self):
        """Test layer initialization"""
        layer = NumericalEmbedding(input_dim=10, output_dim=32)
        assert layer.linear.in_features == 10
        assert layer.linear.out_features == 32
        assert layer.normalize is True
    
    def test_forward_pass(self):
        """Test forward pass"""
        layer = NumericalEmbedding(input_dim=10, output_dim=32)
        input_tensor = torch.randn(16, 10)
        output = layer(input_tensor)
        assert output.shape == (16, 32)


class TestMultiFeatureEmbedding:
    """Test MultiFeatureEmbedding layer"""
    
    def test_initialization(self):
        """Test layer initialization"""
        cat_configs = {
            'category_1': (50, 16),
            'category_2': (100, 32)
        }
        layer = MultiFeatureEmbedding(cat_configs, numerical_dim=10, embedding_dim=128)
        assert len(layer.categorical_embeddings) == 2
        assert layer.numerical_embedding is not None
    
    def test_forward_pass(self):
        """Test forward pass"""
        cat_configs = {
            'category_1': (50, 16),
            'category_2': (100, 32)
        }
        layer = MultiFeatureEmbedding(cat_configs, numerical_dim=10, embedding_dim=128)
        
        # Prepare inputs
        cat_features = {
            'category_1': torch.randint(0, 50, (16,)),
            'category_2': torch.randint(0, 100, (16,))
        }
        num_features = torch.randn(16, 10)
        
        output = layer(cat_features, num_features)
        assert output.shape == (16, 128)


class TestTowerEncoder:
    """Test TowerEncoder"""
    
    def test_initialization(self):
        """Test encoder initialization"""
        encoder = TowerEncoder(input_dim=128, hidden_dims=[256, 128], output_dim=64)
        # Check if encoder is properly constructed
        assert isinstance(encoder.encoder, nn.Sequential)
    
    def test_forward_pass(self):
        """Test forward pass"""
        encoder = TowerEncoder(input_dim=128, hidden_dims=[256, 128], output_dim=64)
        input_tensor = torch.randn(32, 128)
        output = encoder(input_tensor)
        assert output.shape == (32, 64)


if __name__ == '__main__':
    # Run basic tests
    print("Testing CategoricalEmbedding...")
    test_cat = TestCategoricalEmbedding()
    test_cat.test_initialization()
    test_cat.test_with_projection()
    test_cat.test_forward_pass()
    print("✓ CategoricalEmbedding tests passed")
    
    print("\nTesting NumericalEmbedding...")
    test_num = TestNumericalEmbedding()
    test_num.test_initialization()
    test_num.test_forward_pass()
    print("✓ NumericalEmbedding tests passed")
    
    print("\nTesting MultiFeatureEmbedding...")
    test_multi = TestMultiFeatureEmbedding()
    test_multi.test_initialization()
    test_multi.test_forward_pass()
    print("✓ MultiFeatureEmbedding tests passed")
    
    print("\nTesting TowerEncoder...")
    test_tower = TestTowerEncoder()
    test_tower.test_initialization()
    test_tower.test_forward_pass()
    print("✓ TowerEncoder tests passed")
    
    print("\n✅ All embedding layer tests passed!") 