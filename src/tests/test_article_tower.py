"""
Test cases for article tower module
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.article_tower import ArticleTower, create_article_tower
from models.tensor_converter import FeatureToTensorConverter
from features.feature_config import get_article_feature_config, ARTICLE_CATEGORICAL_FEATURES, ARTICLE_NUMERICAL_FEATURES


class TestArticleTower:
    """Test ArticleTower model"""
    
    def test_initialization(self):
        """Test tower initialization"""
        tower = ArticleTower(
            embedding_dim=64,
            hidden_dims=[128, 64],
            output_dim=32
        )
        
        assert tower.normalize is True
        assert tower.get_embedding_dim() == 32
    
    def test_forward_pass(self):
        """Test forward pass through tower"""
        tower = ArticleTower(
            embedding_dim=64,
            output_dim=32
        )
        
        # Create sample inputs matching the expected features
        batch_size = 16
        cat_inputs = {}
        for feature_name in ARTICLE_CATEGORICAL_FEATURES:
            num_categories = ARTICLE_CATEGORICAL_FEATURES[feature_name]['num_categories']
            cat_inputs[feature_name] = torch.randint(0, num_categories, (batch_size,))
        
        # Numerical features as defined in config
        num_features_count = len(ARTICLE_NUMERICAL_FEATURES)
        num_inputs = torch.randn(batch_size, num_features_count)
        
        # Forward pass
        embeddings = tower(cat_inputs, num_inputs)
        
        # Check output shape
        assert embeddings.shape == (batch_size, 32)
        
        # Check normalization (L2 norm should be ~1)
        norms = torch.norm(embeddings, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)
    
    def test_with_zero_numerical_features(self):
        """Test tower with zero numerical features"""
        tower = ArticleTower(output_dim=32)
        
        batch_size = 8
        cat_inputs = {}
        # Must provide all categorical features expected by the tower
        for feature_name in ARTICLE_CATEGORICAL_FEATURES:
            num_categories = ARTICLE_CATEGORICAL_FEATURES[feature_name]['num_categories']
            cat_inputs[feature_name] = torch.randint(0, num_categories, (batch_size,))
        
        # Zero numerical features
        num_features_count = len(ARTICLE_NUMERICAL_FEATURES)
        num_inputs = torch.zeros(batch_size, num_features_count)
        
        embeddings = tower(cat_inputs, num_inputs)
        assert embeddings.shape == (batch_size, 32)


class TestFeatureToTensorConverter:
    """Test FeatureToTensorConverter with article features"""
    
    def test_initialization(self):
        """Test converter initialization"""
        feature_config = {
            'department': {'type': 'categorical', 'num_categories': 20},
            'price': {'type': 'numerical', 'mean': 30.0, 'std': 15.0}
        }
        
        converter = FeatureToTensorConverter(feature_config)
        assert 'department' in converter.categorical_features
        assert 'price' in converter.numerical_features
    
    def test_convert_batch(self):
        """Test batch conversion"""
        converter = FeatureToTensorConverter(get_article_feature_config())
        
        # Create raw features dict with some features
        features_dict = {
            'department_name': [0, 1, 2, 3],
            'product_group_name': [5, 10, 15, 20],
            'colour_group_name': [10, 20, 30, 40],
            'popularity_score': [0.5, 0.8, 0.2, 0.9],
            'avg_selling_price': [0.02, 0.05, 0.08, 0.03]
        }
        
        cat_features, num_features = converter.convert_batch(features_dict)
        
        # Check categorical features
        assert len(cat_features) >= 3
        assert cat_features['department_name'].shape == (4,)
        
        # Check numerical features  
        assert num_features is not None
        assert num_features.shape[0] == 4


class TestFactoryFunction:
    """Test create_article_tower factory function"""
    
    def test_create_article_tower(self):
        """Test tower creation from config"""
        config = {
            'embedding_dim': 128,
            'hidden_dims': [256, 128],
            'output_dim': 64,
            'dropout_rate': 0.15
        }
        
        tower = create_article_tower(config)
        
        # Check tower is created correctly
        assert isinstance(tower, ArticleTower)
        assert tower.get_embedding_dim() == 64
        
        # Test forward pass with all expected features
        batch_size = 32
        cat_inputs = {}
        for feature_name in ARTICLE_CATEGORICAL_FEATURES:
            num_categories = ARTICLE_CATEGORICAL_FEATURES[feature_name]['num_categories']
            cat_inputs[feature_name] = torch.randint(0, num_categories, (batch_size,))
        
        # Numerical features
        num_features_count = len(ARTICLE_NUMERICAL_FEATURES)
        num_inputs = torch.randn(batch_size, num_features_count)
        
        embeddings = tower(cat_inputs, num_inputs)
        assert embeddings.shape == (batch_size, 64)


if __name__ == '__main__':
    # Run tests
    print("Testing ArticleTower...")
    test_tower = TestArticleTower()
    test_tower.test_initialization()
    test_tower.test_forward_pass()
    test_tower.test_with_zero_numerical_features()
    print("✓ ArticleTower tests passed")
    
    print("\nTesting FeatureToTensorConverter...")
    test_converter = TestFeatureToTensorConverter()
    test_converter.test_initialization()
    test_converter.test_convert_batch()
    print("✓ FeatureToTensorConverter tests passed")
    
    print("\nTesting Factory Function...")
    test_factory = TestFactoryFunction()
    test_factory.test_create_article_tower()
    print("✓ Factory function tests passed")
    
    print("\n✅ All article tower tests passed!") 