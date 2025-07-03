"""
Test cases for customer tower module
"""

import pytest
import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.customer_tower import (
    CustomerTower,
    CustomerFeatureProcessor,
    create_customer_tower
)


class TestCustomerTower:
    """Test CustomerTower model"""
    
    def test_initialization(self):
        """Test tower initialization"""
        categorical_features = {
            'age_group': 10,
            'club_member_status': 2,
            'fashion_news_active': 2
        }
        
        tower = CustomerTower(
            categorical_features=categorical_features,
            numerical_dim=5,
            embedding_dim=64,
            hidden_dims=[128, 64],
            output_dim=32
        )
        
        assert tower.normalize is True
        assert tower.get_embedding_dim() == 32
    
    def test_forward_pass(self):
        """Test forward pass through tower"""
        categorical_features = {
            'age_group': 10,
            'club_member_status': 2,
            'fashion_news_active': 2
        }
        
        tower = CustomerTower(
            categorical_features=categorical_features,
            numerical_dim=5,
            embedding_dim=64,
            output_dim=32
        )
        
        # Create sample inputs
        batch_size = 16
        cat_inputs = {
            'age_group': torch.randint(0, 10, (batch_size,)),
            'club_member_status': torch.randint(0, 2, (batch_size,)),
            'fashion_news_active': torch.randint(0, 2, (batch_size,))
        }
        num_inputs = torch.randn(batch_size, 5)
        
        # Forward pass
        embeddings = tower(cat_inputs, num_inputs)
        
        # Check output shape
        assert embeddings.shape == (batch_size, 32)
        
        # Check normalization (L2 norm should be ~1)
        norms = torch.norm(embeddings, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)
    
    def test_no_numerical_features(self):
        """Test tower with only categorical features"""
        categorical_features = {
            'age_group': 10,
            'department': 20
        }
        
        tower = CustomerTower(
            categorical_features=categorical_features,
            numerical_dim=0,
            output_dim=32
        )
        
        batch_size = 8
        cat_inputs = {
            'age_group': torch.randint(0, 10, (batch_size,)),
            'department': torch.randint(0, 20, (batch_size,))
        }
        
        embeddings = tower(cat_inputs, None)
        assert embeddings.shape == (batch_size, 32)


class TestCustomerFeatureProcessor:
    """Test CustomerFeatureProcessor"""
    
    def test_initialization(self):
        """Test processor initialization"""
        feature_configs = {
            'age_group': {'type': 'categorical', 'values': list(range(10))},
            'avg_price': {'type': 'numerical', 'mean': 50.0, 'std': 20.0}
        }
        
        processor = CustomerFeatureProcessor(feature_configs)
        assert 'age_group' in processor.categorical_mappings
        assert len(processor.categorical_mappings['age_group']) == 10
    
    def test_process_features(self):
        """Test feature processing"""
        feature_configs = {
            'age_group': {'type': 'categorical'},
            'avg_price': {'type': 'numerical', 'mean': 50.0, 'std': 20.0, 'normalize': True}
        }
        
        processor = CustomerFeatureProcessor(feature_configs)
        
        # Create raw features
        raw_features = {
            'age_group': torch.tensor([0, 1, 2, 3]),
            'avg_price': torch.tensor([40.0, 60.0, 80.0, 20.0])
        }
        
        cat_features, num_features = processor.process_features(raw_features)
        
        # Check categorical features
        assert 'age_group' in cat_features
        assert torch.equal(cat_features['age_group'], raw_features['age_group'])
        
        # Check numerical features are normalized
        assert num_features is not None
        assert num_features.shape == (4, 1)


class TestFactoryFunction:
    """Test create_customer_tower factory function"""
    
    def test_create_customer_tower(self):
        """Test tower creation from config"""
        config = {
            'age_groups': 8,
            'num_departments': 25,
            'num_colors': 40,
            'numerical_features': 10,
            'embedding_dim': 128,
            'hidden_dims': [256, 128],
            'output_dim': 64,
            'dropout_rate': 0.2
        }
        
        tower = create_customer_tower(config)
        
        # Check tower is created correctly
        assert isinstance(tower, CustomerTower)
        assert tower.get_embedding_dim() == 64
        
        # Test forward pass
        batch_size = 32
        cat_inputs = {
            'age_group': torch.randint(0, 8, (batch_size,)),
            'club_member_status': torch.randint(0, 2, (batch_size,)),
            'fashion_news_active': torch.randint(0, 2, (batch_size,)),
            'customer_lifecycle_stage': torch.randint(0, 5, (batch_size,)),
            'favorite_department': torch.randint(0, 25, (batch_size,)),
            'favorite_color': torch.randint(0, 40, (batch_size,))
        }
        num_inputs = torch.randn(batch_size, 10)
        
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
    
    print("\nTesting CustomerFeatureProcessor...")
    test_processor = TestCustomerFeatureProcessor()
    test_processor.test_initialization()
    test_processor.test_process_features()
    print("✓ CustomerFeatureProcessor tests passed")
    
    print("\nTesting Factory Function...")
    test_factory = TestFactoryFunction()
    test_factory.test_create_customer_tower()
    print("✓ Factory function tests passed")
    
    print("\n✅ All customer tower tests passed!") 