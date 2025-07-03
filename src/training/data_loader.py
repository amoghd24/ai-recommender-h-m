"""
PyTorch DataLoader for Two-Tower Model Training

This module provides DataLoader wrappers and utilities for efficiently 
loading batches of training data for the two-tower model.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging
import pandas as pd
from .dataset import TwoTowerDataset, create_train_val_datasets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching two-tower training data.
    
    Args:
        batch: List of samples from TwoTowerDataset
        
    Returns:
        Dictionary with batched tensors
    """
    # Extract customer features
    customer_categorical = []
    customer_numerical = []
    
    # Extract article features  
    article_categorical = []
    article_numerical = []
    
    # Extract labels
    labels = []
    
    # Extract IDs for debugging
    customer_ids = []
    article_ids = []
    
    for sample in batch:
        # Customer features
        customer_features = sample['customer_features']
        customer_categorical.append(customer_features['categorical'])
        customer_numerical.append(customer_features['numerical'])
        
        # Article features
        article_features = sample['article_features']
        article_categorical.append(article_features['categorical'])
        article_numerical.append(article_features['numerical'])
        
        # Labels and IDs
        labels.append(sample['label'])
        customer_ids.append(sample['customer_id'])
        article_ids.append(sample['article_id'])
    
    # Stack tensors into batches
    try:
        customer_categorical_batch = torch.stack(customer_categorical, dim=0)
        customer_numerical_batch = torch.stack(customer_numerical, dim=0)
        article_categorical_batch = torch.stack(article_categorical, dim=0)
        article_numerical_batch = torch.stack(article_numerical, dim=0)
        labels_batch = torch.stack(labels, dim=0)
        
        return {
            'customer_features': {
                'categorical': customer_categorical_batch,
                'numerical': customer_numerical_batch
            },
            'article_features': {
                'categorical': article_categorical_batch,
                'numerical': article_numerical_batch
            },
            'labels': labels_batch,
            'customer_ids': customer_ids,
            'article_ids': article_ids
        }
    except Exception as e:
        logger.error(f"Error in collate_fn: {e}")
        logger.error(f"Customer categorical shapes: {[t.shape for t in customer_categorical]}")
        logger.error(f"Customer numerical shapes: {[t.shape for t in customer_numerical]}")
        logger.error(f"Article categorical shapes: {[t.shape for t in article_categorical]}")
        logger.error(f"Article numerical shapes: {[t.shape for t in article_numerical]}")
        raise


class TwoTowerDataLoader:
    """
    DataLoader wrapper for Two-Tower model training with configuration management.
    """
    
    def __init__(
        self,
        dataset: TwoTowerDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = False
    ):
        """
        Initialize two-tower data loader.
        
        Args:
            dataset: TwoTowerDataset instance
            batch_size: Batch size for training
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            drop_last: Whether to drop incomplete last batch
            pin_memory: Whether to pin memory for faster GPU transfer
            persistent_workers: Whether to keep workers alive between epochs
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        
        # Create PyTorch DataLoader
        self.dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn
        )
        
        logger.info(f"TwoTowerDataLoader initialized:")
        logger.info(f"  - Dataset size: {len(dataset):,}")
        logger.info(f"  - Batch size: {batch_size}")
        logger.info(f"  - Batches per epoch: {len(self.dataloader):,}")
        logger.info(f"  - Shuffle: {shuffle}")
        logger.info(f"  - Num workers: {num_workers}")
        
    def __iter__(self):
        """Return iterator over batches"""
        return iter(self.dataloader)
    
    def __len__(self):
        """Return number of batches"""
        return len(self.dataloader)
    
    def get_batch_info(self) -> Dict[str, Any]:
        """
        Get information about batch structure.
        
        Returns:
            Dictionary with batch information
        """
        return {
            'dataset_size': len(self.dataset),
            'batch_size': self.batch_size,
            'num_batches': len(self.dataloader),
            'shuffle': self.shuffle,
            'num_workers': self.num_workers,
            'drop_last': self.drop_last,
            'pin_memory': self.pin_memory,
            'persistent_workers': self.persistent_workers
        }
    
    def get_sample_batch(self) -> Dict[str, torch.Tensor]:
        """
        Get a sample batch for debugging or model testing.
        
        Returns:
            Single batch of data
        """
        return next(iter(self.dataloader))


def create_train_val_loaders(
    customers_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    train_batch_size: int = 32,
    val_batch_size: int = 64,
    val_split: float = 0.2,
    negative_samples_per_positive: int = 4,
    num_workers: int = 0,
    random_seed: int = 42,
    shuffle_train: bool = True,
    shuffle_val: bool = False,
    drop_last: bool = False,
    pin_memory: bool = True
) -> Tuple[TwoTowerDataLoader, TwoTowerDataLoader]:
    """
    Create train and validation data loaders.
    
    Args:
        customers_df: Customer data
        articles_df: Article data
        transactions_df: Transaction data
        train_batch_size: Training batch size
        val_batch_size: Validation batch size
        val_split: Fraction of data for validation
        negative_samples_per_positive: Negative samples per positive
        num_workers: Number of worker processes
        random_seed: Random seed for reproducibility
        shuffle_train: Whether to shuffle training data
        shuffle_val: Whether to shuffle validation data
        drop_last: Whether to drop incomplete last batch
        pin_memory: Whether to pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info("Creating train and validation data loaders...")
    
    # Create datasets
    train_dataset, val_dataset = create_train_val_datasets(
        customers_df=customers_df,
        articles_df=articles_df,
        transactions_df=transactions_df,
        val_split=val_split,
        negative_samples_per_positive=negative_samples_per_positive,
        random_seed=random_seed
    )
    
    # Create train loader
    train_loader = TwoTowerDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Create validation loader  
    val_loader = TwoTowerDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=shuffle_val,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    logger.info("Data loaders created successfully!")
    logger.info(f"Train loader: {len(train_loader):,} batches")
    logger.info(f"Validation loader: {len(val_loader):,} batches")
    
    return train_loader, val_loader


def create_inference_loader(
    customers_df: pd.DataFrame,
    articles_df: pd.DataFrame,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = True
) -> DataLoader:
    """
    Create data loader for inference (computing embeddings).
    
    Args:
        customers_df: Customer data
        articles_df: Article data
        batch_size: Batch size for inference
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory
        
    Returns:
        DataLoader for inference
    """
    # This is a placeholder - in real implementation, you'd create
    # separate datasets for just computing embeddings without labels
    pass


def get_dataloader_stats(dataloader: TwoTowerDataLoader) -> Dict[str, Any]:
    """
    Get detailed statistics about a data loader.
    
    Args:
        dataloader: TwoTowerDataLoader instance
        
    Returns:
        Dictionary with statistics
    """
    dataset = dataloader.dataset
    
    # Get label distribution
    all_labels = [dataset[i]['label'].item() for i in range(len(dataset))]
    positive_count = sum(all_labels)
    negative_count = len(all_labels) - positive_count
    
    stats = {
        'dataset_size': len(dataset),
        'positive_samples': positive_count,
        'negative_samples': negative_count,
        'positive_ratio': positive_count / len(all_labels),
        'negative_ratio': negative_count / len(all_labels),
        'batch_size': dataloader.batch_size,
        'num_batches': len(dataloader),
        'shuffle': dataloader.shuffle,
        'num_workers': dataloader.num_workers,
        'samples_per_epoch': len(dataset)
    }
    
    return stats


def debug_dataloader_batch(dataloader: TwoTowerDataLoader) -> None:
    """
    Test a data loader by fetching and examining a batch.
    
    Args:
        dataloader: TwoTowerDataLoader to test
    """
    logger.info("Testing data loader batch...")
    
    try:
        batch = dataloader.get_sample_batch()
        
        logger.info("Batch structure:")
        logger.info(f"  Customer features:")
        logger.info(f"    Categorical: {batch['customer_features']['categorical'].shape}")
        logger.info(f"    Numerical: {batch['customer_features']['numerical'].shape}")
        logger.info(f"  Article features:")
        logger.info(f"    Categorical: {batch['article_features']['categorical'].shape}")
        logger.info(f"    Numerical: {batch['article_features']['numerical'].shape}")
        logger.info(f"  Labels: {batch['labels'].shape}")
        logger.info(f"  Sample customer IDs: {batch['customer_ids'][:5]}")
        logger.info(f"  Sample article IDs: {batch['article_ids'][:5]}")
        
        # Check label distribution in batch
        positive_labels = (batch['labels'] == 1).sum().item()
        negative_labels = (batch['labels'] == 0).sum().item()
        logger.info(f"  Positive labels in batch: {positive_labels}")
        logger.info(f"  Negative labels in batch: {negative_labels}")
        
        logger.info("Data loader test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing data loader: {e}")
        raise 