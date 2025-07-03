"""
Test cases for customer tower module
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.customer_tower import CustomerTower, create_customer_tower
from models.tensor_converter import FeatureToTensorConverter
from features.feature_config import get_customer_feature_config


class TestCustomerTower:
    """Test CustomerTower model"""
    
    def test_initialization(self):
        """Test tower initialization"""
        tower = CustomerTower(
            embedding_dim=64,
            hidden_dims=[128, 64],
            output_dim=32
        )
        
        assert tower.normalize is True
        assert tower.get_embedding_dim() == 32
    
    def test_forward_pass(self):
        """Test forward pass through tower"""
        tower = CustomerTower(
            embedding_dim=64,
            output_dim=32
        )
        
        # Create sample inputs matching the expected features
        batch_size = 16
        cat_inputs = {
            'age_group': torch.randint(0, 10, (batch_size,)),
            'club_member_status': torch.randint(0, 2, (batch_size,)),
            'fashion_news_active': torch.randint(0, 2, (batch_size,)),
            'customer_lifecycle_stage': torch.randint(0, 5, (batch_size,)),
            'favorite_department': torch.randint(0, 20, (batch_size,)),
            'favorite_color': torch.randint(0, 50, (batch_size,))
        }
        # 4 numerical features as defined in config
        num_inputs = torch.randn(batch_size, 4)
        
        # Forward pass
        embeddings = tower(cat_inputs, num_inputs)
        
        # Check output shape
        assert embeddings.shape == (batch_size, 32)
        
        # Check normalization (L2 norm should be ~1)
        norms = torch.norm(embeddings, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)
    
    def test_no_numerical_features(self):
        """Test tower with zero numerical features"""
        tower = CustomerTower(output_dim=32)
        
        batch_size = 8
        cat_inputs = {
            'age_group': torch.randint(0, 10, (batch_size,)),
            'club_member_status': torch.randint(0, 2, (batch_size,)),
            'fashion_news_active': torch.randint(0, 2, (batch_size,)),
            'customer_lifecycle_stage': torch.randint(0, 5, (batch_size,)),
            'favorite_department': torch.randint(0, 20, (batch_size,)),
            'favorite_color': torch.randint(0, 50, (batch_size,))
        }
        
        # Pass zeros for numerical features (4 features as per config)
        num_inputs = torch.zeros(batch_size, 4)
        
        embeddings = tower(cat_inputs, num_inputs)
        assert embeddings.shape == (batch_size, 32)


class TestFeatureToTensorConverter:
    """Test FeatureToTensorConverter with customer features"""
    
    def test_initialization(self):
        """Test converter initialization"""
        feature_config = {
            'age_group': {'type': 'categorical', 'num_categories': 10},
            'avg_price': {'type': 'numerical', 'mean': 50.0, 'std': 20.0}
        }
        
        converter = FeatureToTensorConverter(feature_config)
        assert 'age_group' in converter.categorical_features
        assert 'avg_price' in converter.numerical_features
    
    def test_convert_batch(self):
        """Test batch conversion"""
        converter = FeatureToTensorConverter(get_customer_feature_config())
        
        # Create raw features dict
        features_dict = {
            'age_group': [0, 1, 2, 3],
            'club_member_status': [0, 1, 0, 1],
            'fashion_news_active': [1, 1, 0, 0],
            'customer_lifecycle_stage': [0, 1, 2, 3],
            'favorite_department': [5, 10, 15, 20],
            'favorite_color': [10, 20, 30, 40],
            'days_since_last_purchase': [10.0, 20.0, 30.0, 40.0],
            'purchase_frequency': [0.5, 0.8, 0.2, 0.9]
        }
        
        cat_features, num_features = converter.convert_batch(features_dict)
        
        # Check categorical features
        assert len(cat_features) == 6
        assert cat_features['age_group'].shape == (4,)
        
        # Check numerical features  
        assert num_features is not None
        assert num_features.shape[0] == 4


class TestFactoryFunction:
    """Test create_customer_tower factory function"""
    
    def test_create_customer_tower(self):
        """Test tower creation from config"""
        config = {
            'embedding_dim': 128,
            'hidden_dims': [256, 128],
            'output_dim': 64,
            'dropout_rate': 0.2
        }
        
        tower = create_customer_tower(config)
        
        # Check tower is created correctly
        assert isinstance(tower, CustomerTower)
        assert tower.get_embedding_dim() == 64
        
        # Test forward pass with all expected features
        batch_size = 32
        cat_inputs = {
            'age_group': torch.randint(0, 10, (batch_size,)),
            'club_member_status': torch.randint(0, 2, (batch_size,)),
            'fashion_news_active': torch.randint(0, 2, (batch_size,)),
            'customer_lifecycle_stage': torch.randint(0, 5, (batch_size,)),
            'favorite_department': torch.randint(0, 20, (batch_size,)),
            'favorite_color': torch.randint(0, 50, (batch_size,))
        }
        # 4 numerical features
        num_inputs = torch.randn(batch_size, 4)
        
        embeddings = tower(cat_inputs, num_inputs)
        assert embeddings.shape == (batch_size, 64)


if __name__ == '__main__':
    # Run tests
    print("Testing CustomerTower...")
    test_tower = TestCustomerTower()
    test_tower.test_initialization()
    test_tower.test_forward_pass()
    test_tower.test_no_numerical_features()
    print("✓ CustomerTower tests passed")
    
    print("\nTesting FeatureToTensorConverter...")
    test_converter = TestFeatureToTensorConverter()
    test_converter.test_initialization()
    test_converter.test_convert_batch()
    print("✓ FeatureToTensorConverter tests passed")
    
    print("\nTesting Factory Function...")
    test_factory = TestFactoryFunction()
    test_factory.test_create_customer_tower()
    print("✓ Factory function tests passed")
    
    print("\n✅ All customer tower tests passed!") 