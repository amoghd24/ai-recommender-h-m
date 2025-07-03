"""
Test Training Data Loader and Dataset

Tests for PyTorch datasets and data loaders used in two-tower model training.
"""

import pytest
import torch
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.dataset import InteractionDataset, TwoTowerDataset, create_train_val_datasets
from training.data_loader import TwoTowerDataLoader, create_train_val_loaders, collate_fn
from training.data_loader import get_dataloader_stats, debug_dataloader_batch


def create_complete_test_data():
    """Create complete test data with all required columns for feature engineers"""
    customers_df = pd.DataFrame({
        'customer_id': ['c1', 'c2', 'c3'],
        'age': [25, 30, 35],
        'postal_code': ['12345', '67890', '11111'],
        'FN': [1.0, 0.5, 0.0],
        'Active': [1.0, 1.0, 0.0],
        'club_member_status': ['ACTIVE', 'ACTIVE', 'LEFT CLUB'],
        'fashion_news_frequency': ['Regularly', 'Monthly', 'NONE']
    })
    
    articles_df = pd.DataFrame({
        'article_id': [1, 2, 3, 4],
        'product_type_name': ['T-shirt', 'Jeans', 'Shoes', 'Hat'],
        'colour_group_name': ['Red', 'Blue', 'Black', 'White'],
        'product_code': [123456, 234567, 345678, 456789],
        'prod_name': ['Test T-shirt', 'Test Jeans', 'Test Shoes', 'Test Hat'],
        'product_type_no': [1, 2, 3, 4],
        'product_group_name': ['Garment Upper body', 'Garment Lower body', 'Shoes', 'Accessories'],
        'graphical_appearance_no': [1, 2, 3, 4],
        'graphical_appearance_name': ['Solid', 'Solid', 'Solid', 'Solid'],
        'colour_group_code': [1, 2, 3, 4],
        'perceived_colour_value_id': [1, 2, 3, 4],
        'perceived_colour_value_name': ['Light', 'Medium', 'Dark', 'Light'],
        'perceived_colour_master_id': [1, 2, 3, 4],
        'perceived_colour_master_name': ['Red', 'Blue', 'Black', 'White'],
        'department_no': [1, 2, 3, 4],
        'department_name': ['Menswear', 'Ladieswear', 'Sport', 'Accessories'],
        'index_code': ['A', 'B', 'C', 'D'],
        'index_name': ['Ladieswear', 'Menswear', 'Sport', 'Accessories'],
        'index_group_no': [1, 2, 3, 4],
        'index_group_name': ['Ladieswear', 'Menswear', 'Sport', 'Accessories'],
        'section_no': [1, 2, 3, 4],
        'section_name': ['Womens Everyday Basics', 'Mens Basics', 'Sport', 'Accessories'],
        'garment_group_no': [1, 2, 3, 4],
        'garment_group_name': ['Jersey Basic', 'Denim', 'Footwear', 'Hat/Cap'],
        'detail_desc': ['Jersey top', 'Blue jeans', 'Black shoes', 'White hat']
    })
    
    transactions_df = pd.DataFrame({
        'customer_id': ['c1', 'c2', 'c1', 'c3'],
        'article_id': [1, 2, 3, 1],
        't_dat': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']),
        'price': [10.0, 20.0, 15.0, 10.0]
    })
    
    return customers_df, articles_df, transactions_df


class TestInteractionDataset:
    """Test InteractionDataset class"""
    
    def test_dataset_creation(self):
        """Test basic dataset creation"""
        # Create sample data with complete schema
        customers_df, articles_df, transactions_df = create_complete_test_data()
        
        # Create dataset
        dataset = InteractionDataset(
            customers_df=customers_df,
            articles_df=articles_df,
            transactions_df=transactions_df,
            max_interactions_per_customer=10
        )
        
        # Basic assertions
        assert len(dataset) == 4  # 4 transactions
        assert hasattr(dataset, 'interactions')
        assert hasattr(dataset, 'customer_engineer')
        assert hasattr(dataset, 'article_engineer')
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method"""
        # Create complete sample data with all required columns
        customers_df = pd.DataFrame({
            'customer_id': ['c1'],
            'age': [25],
            'postal_code': ['12345'],
            'FN': [1.0],
            'Active': [1.0],
            'club_member_status': ['ACTIVE'],
            'fashion_news_frequency': ['Regularly']
        })
        
        articles_df = pd.DataFrame({
            'article_id': [1],
            'product_type_name': ['T-shirt'],
            'colour_group_name': ['Red'],
            'product_code': [123456],
            'prod_name': ['Test T-shirt'],
            'product_type_no': [1],
            'product_group_name': ['Garment Upper body'],
            'graphical_appearance_no': [1],
            'graphical_appearance_name': ['Solid'],
            'colour_group_code': [1],
            'perceived_colour_value_id': [1],
            'perceived_colour_value_name': ['Light'],
            'perceived_colour_master_id': [1],
            'perceived_colour_master_name': ['Red'],
            'department_no': [1],
            'department_name': ['Menswear'],
            'index_code': ['A'],
            'index_name': ['Ladieswear'],
            'index_group_no': [1],
            'index_group_name': ['Ladieswear'],
            'section_no': [1],
            'section_name': ['Womens Everyday Basics'],
            'garment_group_no': [1],
            'garment_group_name': ['Jersey Basic'],
            'detail_desc': ['Jersey top with short sleeves']
        })
        
        transactions_df = pd.DataFrame({
            'customer_id': ['c1'],
            'article_id': [1],
            't_dat': pd.to_datetime(['2023-01-01']),
            'price': [10.0]
        })
        
        dataset = InteractionDataset(
            customers_df=customers_df,
            articles_df=articles_df,
            transactions_df=transactions_df
        )
        
        # Get sample
        sample = dataset[0]
        
        # Check structure
        assert 'customer_features' in sample
        assert 'article_features' in sample
        assert 'customer_id' in sample
        assert 'article_id' in sample
        
        # Check tensor types
        assert isinstance(sample['customer_features']['categorical'], torch.Tensor)
        assert isinstance(sample['customer_features']['numerical'], torch.Tensor)
        assert isinstance(sample['article_features']['categorical'], torch.Tensor)
        assert isinstance(sample['article_features']['numerical'], torch.Tensor)


class TestTwoTowerDataset:
    """Test TwoTowerDataset class"""
    
    def test_dataset_with_negatives(self):
        """Test dataset creation with negative sampling"""
        # Create sample data
        customers_df = pd.DataFrame({
            'customer_id': ['c1', 'c2', 'c3'],
            'age': [25, 30, 35],
            'postal_code': ['12345', '67890', '11111']
        })
        
        articles_df = pd.DataFrame({
            'article_id': ['a1', 'a2', 'a3', 'a4'],
            'product_type_name': ['T-shirt', 'Jeans', 'Shoes', 'Hat'],
            'colour_group_name': ['Red', 'Blue', 'Black', 'White']
        })
        
        transactions_df = pd.DataFrame({
            'customer_id': ['c1', 'c2'],
            'article_id': ['a1', 'a2'],
            't_dat': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'price': [10.0, 20.0]
        })
        
        # Create dataset with negative sampling
        dataset = TwoTowerDataset(
            customers_df=customers_df,
            articles_df=articles_df,
            transactions_df=transactions_df,
            negative_samples_per_positive=2,
            random_seed=42
        )
        
        # Check dataset size (2 positive + 4 negative = 6 total)
        assert len(dataset) == 6
        
        # Check that we have both positive and negative samples
        labels = [dataset[i]['label'].item() for i in range(len(dataset))]
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        
        assert positive_count == 2  # 2 positive samples
        assert negative_count == 4  # 4 negative samples
    
    def test_dataset_sample_structure(self):
        """Test structure of dataset samples"""
        # Create minimal data
        customers_df = pd.DataFrame({
            'customer_id': ['c1'],
            'age': [25],
            'postal_code': ['12345']
        })
        
        articles_df = pd.DataFrame({
            'article_id': ['a1', 'a2'],
            'product_type_name': ['T-shirt', 'Jeans'],
            'colour_group_name': ['Red', 'Blue']
        })
        
        transactions_df = pd.DataFrame({
            'customer_id': ['c1'],
            'article_id': ['a1'],
            't_dat': pd.to_datetime(['2023-01-01']),
            'price': [10.0]
        })
        
        dataset = TwoTowerDataset(
            customers_df=customers_df,
            articles_df=articles_df,
            transactions_df=transactions_df,
            negative_samples_per_positive=1,
            random_seed=42
        )
        
        # Get samples
        sample = dataset[0]
        
        # Check sample structure
        assert 'customer_features' in sample
        assert 'article_features' in sample
        assert 'label' in sample
        assert 'customer_id' in sample
        assert 'article_id' in sample
        
        # Check tensor types
        assert isinstance(sample['customer_features']['categorical'], torch.Tensor)
        assert isinstance(sample['customer_features']['numerical'], torch.Tensor)
        assert isinstance(sample['article_features']['categorical'], torch.Tensor)
        assert isinstance(sample['article_features']['numerical'], torch.Tensor)
        assert isinstance(sample['label'], torch.Tensor)


class TestTwoTowerDataLoader:
    """Test TwoTowerDataLoader class"""
    
    def test_dataloader_creation(self):
        """Test basic data loader creation"""
        # Create minimal dataset
        customers_df = pd.DataFrame({
            'customer_id': ['c1', 'c2'],
            'age': [25, 30],
            'postal_code': ['12345', '67890']
        })
        
        articles_df = pd.DataFrame({
            'article_id': ['a1', 'a2'],
            'product_type_name': ['T-shirt', 'Jeans'],
            'colour_group_name': ['Red', 'Blue']
        })
        
        transactions_df = pd.DataFrame({
            'customer_id': ['c1', 'c2'],
            'article_id': ['a1', 'a2'],
            't_dat': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'price': [10.0, 20.0]
        })
        
        # Create dataset
        dataset = TwoTowerDataset(
            customers_df=customers_df,
            articles_df=articles_df,
            transactions_df=transactions_df,
            negative_samples_per_positive=1,
            random_seed=42
        )
        
        # Create data loader
        dataloader = TwoTowerDataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        
        # Test basic properties
        assert len(dataloader) == 2  # 4 samples / 2 batch_size = 2 batches
        assert dataloader.batch_size == 2
        assert dataloader.shuffle == False
        assert dataloader.num_workers == 0
    
    def test_dataloader_batching(self):
        """Test data loader batching functionality"""
        # Create minimal dataset
        customers_df = pd.DataFrame({
            'customer_id': ['c1', 'c2'],
            'age': [25, 30],
            'postal_code': ['12345', '67890']
        })
        
        articles_df = pd.DataFrame({
            'article_id': ['a1', 'a2'],
            'product_type_name': ['T-shirt', 'Jeans'],
            'colour_group_name': ['Red', 'Blue']
        })
        
        transactions_df = pd.DataFrame({
            'customer_id': ['c1', 'c2'],
            'article_id': ['a1', 'a2'],
            't_dat': pd.to_datetime(['2023-01-01', '2023-01-02']),
            'price': [10.0, 20.0]
        })
        
        # Create dataset
        dataset = TwoTowerDataset(
            customers_df=customers_df,
            articles_df=articles_df,
            transactions_df=transactions_df,
            negative_samples_per_positive=1,
            random_seed=42
        )
        
        # Create data loader
        dataloader = TwoTowerDataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,
            num_workers=0
        )
        
        # Get a batch
        batch = dataloader.get_sample_batch()
        
        # Check batch structure
        assert 'customer_features' in batch
        assert 'article_features' in batch
        assert 'labels' in batch
        assert 'customer_ids' in batch
        assert 'article_ids' in batch
        
        # Check batch dimensions
        assert batch['customer_features']['categorical'].shape[0] == 2  # batch size
        assert batch['customer_features']['numerical'].shape[0] == 2
        assert batch['article_features']['categorical'].shape[0] == 2
        assert batch['article_features']['numerical'].shape[0] == 2
        assert batch['labels'].shape[0] == 2
        assert len(batch['customer_ids']) == 2
        assert len(batch['article_ids']) == 2


class TestUtilityFunctions:
    """Test utility functions"""
    
    def test_create_train_val_datasets(self):
        """Test train/validation dataset creation"""
        # Create sample data
        customers_df = pd.DataFrame({
            'customer_id': ['c1', 'c2', 'c3'],
            'age': [25, 30, 35],
            'postal_code': ['12345', '67890', '11111']
        })
        
        articles_df = pd.DataFrame({
            'article_id': ['a1', 'a2', 'a3'],
            'product_type_name': ['T-shirt', 'Jeans', 'Shoes'],
            'colour_group_name': ['Red', 'Blue', 'Black']
        })
        
        transactions_df = pd.DataFrame({
            'customer_id': ['c1', 'c2', 'c1', 'c3', 'c2'],
            'article_id': ['a1', 'a2', 'a3', 'a1', 'a3'],
            't_dat': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'price': [10.0, 20.0, 15.0, 10.0, 15.0]
        })
        
        # Create datasets
        train_dataset, val_dataset = create_train_val_datasets(
            customers_df=customers_df,
            articles_df=articles_df,
            transactions_df=transactions_df,
            val_split=0.2,
            negative_samples_per_positive=1,
            random_seed=42
        )
        
        # Check that we have both datasets
        assert isinstance(train_dataset, TwoTowerDataset)
        assert isinstance(val_dataset, TwoTowerDataset)
        
        # Check that train dataset is larger
        assert len(train_dataset) > len(val_dataset)
    
    def test_create_train_val_loaders(self):
        """Test train/validation loader creation"""
        # Create sample data
        customers_df = pd.DataFrame({
            'customer_id': ['c1', 'c2', 'c3'],
            'age': [25, 30, 35],
            'postal_code': ['12345', '67890', '11111']
        })
        
        articles_df = pd.DataFrame({
            'article_id': ['a1', 'a2', 'a3'],
            'product_type_name': ['T-shirt', 'Jeans', 'Shoes'],
            'colour_group_name': ['Red', 'Blue', 'Black']
        })
        
        transactions_df = pd.DataFrame({
            'customer_id': ['c1', 'c2', 'c1', 'c3', 'c2'],
            'article_id': ['a1', 'a2', 'a3', 'a1', 'a3'],
            't_dat': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
            'price': [10.0, 20.0, 15.0, 10.0, 15.0]
        })
        
        # Create loaders
        train_loader, val_loader = create_train_val_loaders(
            customers_df=customers_df,
            articles_df=articles_df,
            transactions_df=transactions_df,
            train_batch_size=2,
            val_batch_size=2,
            val_split=0.2,
            negative_samples_per_positive=1,
            random_seed=42
        )
        
        # Check that we have both loaders
        assert isinstance(train_loader, TwoTowerDataLoader)
        assert isinstance(val_loader, TwoTowerDataLoader)
        
        # Check batch sizes
        assert train_loader.batch_size == 2
        assert val_loader.batch_size == 2
        
        # Check that we can get batches
        train_batch = train_loader.get_sample_batch()
        val_batch = val_loader.get_sample_batch()
        
        assert 'customer_features' in train_batch
        assert 'article_features' in train_batch
        assert 'labels' in train_batch
        
        assert 'customer_features' in val_batch
        assert 'article_features' in val_batch
        assert 'labels' in val_batch


class TestCollateFn:
    """Test custom collate function"""
    
    def test_collate_function(self):
        """Test custom collate function for batching"""
        # Create mock samples
        sample1 = {
            'customer_features': {
                'categorical': torch.tensor([1, 2, 3]),
                'numerical': torch.tensor([0.1, 0.2, 0.3, 0.4])
            },
            'article_features': {
                'categorical': torch.tensor([4, 5, 6]),
                'numerical': torch.tensor([0.5, 0.6, 0.7, 0.8])
            },
            'label': torch.tensor(1.0),
            'customer_id': 'c1',
            'article_id': 'a1'
        }
        
        sample2 = {
            'customer_features': {
                'categorical': torch.tensor([7, 8, 9]),
                'numerical': torch.tensor([0.9, 1.0, 1.1, 1.2])
            },
            'article_features': {
                'categorical': torch.tensor([10, 11, 12]),
                'numerical': torch.tensor([1.3, 1.4, 1.5, 1.6])
            },
            'label': torch.tensor(0.0),
            'customer_id': 'c2',
            'article_id': 'a2'
        }
        
        batch = [sample1, sample2]
        
        # Test collate function
        collated = collate_fn(batch)
        
        # Check structure
        assert 'customer_features' in collated
        assert 'article_features' in collated
        assert 'labels' in collated
        assert 'customer_ids' in collated
        assert 'article_ids' in collated
        
        # Check dimensions
        assert collated['customer_features']['categorical'].shape == (2, 3)
        assert collated['customer_features']['numerical'].shape == (2, 4)
        assert collated['article_features']['categorical'].shape == (2, 3)
        assert collated['article_features']['numerical'].shape == (2, 4)
        assert collated['labels'].shape == (2,)
        assert len(collated['customer_ids']) == 2
        assert len(collated['article_ids']) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 