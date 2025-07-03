"""
PyTorch Dataset Classes for Two-Tower Model Training

This module provides dataset classes that convert H&M data into 
PyTorch-compatible training data with positive/negative sampling.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union
import logging
from sklearn.model_selection import train_test_split

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import feature processing components
try:
    from ..features.customer_features import CustomerFeatureEngineer
    from ..features.article_features import ArticleFeatureEngineer  
    from ..models.tensor_converter import FeatureToTensorConverter
    from ..features.feature_config import get_customer_feature_config, get_article_feature_config
except ImportError:
    # For standalone testing
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from features.customer_features import CustomerFeatureEngineer
    from features.article_features import ArticleFeatureEngineer
    from models.tensor_converter import FeatureToTensorConverter
    from features.feature_config import get_customer_feature_config, get_article_feature_config


class InteractionDataset(Dataset):
    """
    Basic dataset for customer-article interactions.
    Contains positive interactions from transaction data.
    """
    
    def __init__(
        self, 
        customers_df: pd.DataFrame,
        articles_df: pd.DataFrame, 
        transactions_df: pd.DataFrame,
        max_interactions_per_customer: int = 50
    ):
        """
        Initialize interaction dataset.
        
        Args:
            customers_df: Customer data
            articles_df: Article data  
            transactions_df: Transaction data (positive interactions)
            max_interactions_per_customer: Limit interactions per customer
        """
        self.customers_df = customers_df
        self.articles_df = articles_df
        self.max_interactions_per_customer = max_interactions_per_customer
        
        # Process interactions
        self.interactions = self._process_interactions(transactions_df)
        
        # Initialize feature processors
        self.customer_engineer = CustomerFeatureEngineer()
        self.article_engineer = ArticleFeatureEngineer()
        
        # Initialize tensor converters
        self.customer_converter = FeatureToTensorConverter(get_customer_feature_config())
        self.article_converter = FeatureToTensorConverter(get_article_feature_config())
        
        logger.info(f"InteractionDataset initialized with {len(self.interactions):,} interactions")
    
    def _process_interactions(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw transactions into interaction pairs.
        
        Args:
            transactions_df: Raw transaction data
            
        Returns:
            Processed interactions DataFrame
        """
        # Group by customer and limit interactions
        interactions = []
        
        for customer_id, group in transactions_df.groupby('customer_id'):
            # Sort by date and take most recent interactions
            recent_interactions = group.sort_values('t_dat').tail(
                self.max_interactions_per_customer
            )
            
            for _, row in recent_interactions.iterrows():
                interactions.append({
                    'customer_id': customer_id,
                    'article_id': row['article_id'],
                    'interaction_date': row['t_dat'],
                    'price': row['price']
                })
        
        interactions_df = pd.DataFrame(interactions)
        logger.info(f"Processed {len(interactions_df):,} interactions from {len(transactions_df):,} transactions")
        
        return interactions_df
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single interaction sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with customer and article features as tensors
        """
        interaction = self.interactions.iloc[idx]
        
        # Get customer data
        customer_data = self.customers_df[
            self.customers_df['customer_id'] == interaction['customer_id']
        ].iloc[0]
        
        # Get article data  
        article_data = self.articles_df[
            self.articles_df['article_id'] == interaction['article_id']
        ].iloc[0]
        
        # Process features
        customer_features = self.customer_engineer.extract_basic_features(
            pd.DataFrame([customer_data])
        ).iloc[0]
        
        article_features = self.article_engineer.extract_basic_features(
            pd.DataFrame([article_data])
        ).iloc[0]
        
        # Convert to tensors
        customer_tensors = self.customer_converter.convert_series(customer_features)
        article_tensors = self.article_converter.convert_series(article_features)
        
        return {
            'customer_features': customer_tensors,
            'article_features': article_tensors,
            'customer_id': interaction['customer_id'],
            'article_id': interaction['article_id']
        }


class TwoTowerDataset(Dataset):
    """
    Dataset for Two-Tower model training with negative sampling.
    
    Generates positive and negative customer-article pairs for contrastive learning.
    """
    
    def __init__(
        self,
        customers_df: pd.DataFrame,
        articles_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        negative_samples_per_positive: int = 4,
        max_interactions_per_customer: int = 50,
        random_seed: int = 42
    ):
        """
        Initialize two-tower training dataset.
        
        Args:
            customers_df: Customer data
            articles_df: Article data
            transactions_df: Transaction data
            negative_samples_per_positive: Number of negative samples per positive
            max_interactions_per_customer: Limit interactions per customer
            random_seed: Random seed for reproducibility
        """
        self.customers_df = customers_df
        self.articles_df = articles_df
        self.negative_samples_per_positive = negative_samples_per_positive
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        # Get base interactions
        base_dataset = InteractionDataset(
            customers_df, articles_df, transactions_df, max_interactions_per_customer
        )
        self.positive_interactions = base_dataset.interactions
        
        # Create negative samples
        self.negative_interactions = self._generate_negative_samples()
        
        # Combine positive and negative samples
        self.all_samples = self._combine_samples()
        
        # Initialize feature processors (reuse from base dataset)
        self.customer_engineer = base_dataset.customer_engineer
        self.article_engineer = base_dataset.article_engineer
        self.customer_converter = base_dataset.customer_converter
        self.article_converter = base_dataset.article_converter
        
        # Cache processed customer and article features for efficiency
        self._cache_features()
        
        logger.info(f"TwoTowerDataset initialized:")
        logger.info(f"  - Positive samples: {len(self.positive_interactions):,}")
        logger.info(f"  - Negative samples: {len(self.negative_interactions):,}")
        logger.info(f"  - Total samples: {len(self.all_samples):,}")
    
    def _generate_negative_samples(self) -> pd.DataFrame:
        """
        Generate negative samples by randomly pairing customers with articles 
        they haven't interacted with.
        
        Returns:
            DataFrame with negative interactions
        """
        logger.info("Generating negative samples...")
        
        # Get sets of existing interactions
        existing_pairs = set(zip(
            self.positive_interactions['customer_id'],
            self.positive_interactions['article_id']
        ))
        
        # Get unique customers and articles
        unique_customers = self.positive_interactions['customer_id'].unique()
        unique_articles = self.articles_df['article_id'].unique()
        
        negative_samples = []
        target_negatives = len(self.positive_interactions) * self.negative_samples_per_positive
        
        while len(negative_samples) < target_negatives:
            # Randomly sample customer and article
            customer_id = np.random.choice(unique_customers)
            article_id = np.random.choice(unique_articles)
            
            # Check if this is a valid negative sample
            if (customer_id, article_id) not in existing_pairs:
                negative_samples.append({
                    'customer_id': customer_id,
                    'article_id': article_id,
                    'label': 0  # Negative sample
                })
                
                # Add to existing pairs to avoid duplicates
                existing_pairs.add((customer_id, article_id))
        
        return pd.DataFrame(negative_samples)
    
    def _combine_samples(self) -> pd.DataFrame:
        """
        Combine positive and negative samples into training dataset.
        
        Returns:
            Combined DataFrame with labels
        """
        # Add labels to positive samples
        positive_labeled = self.positive_interactions.copy()
        positive_labeled['label'] = 1
        
        # Combine and shuffle
        all_samples = pd.concat([
            positive_labeled[['customer_id', 'article_id', 'label']],
            self.negative_interactions[['customer_id', 'article_id', 'label']]
        ], ignore_index=True)
        
        # Shuffle the dataset
        all_samples = all_samples.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        return all_samples
    
    def _cache_features(self):
        """Cache processed features for efficiency during training."""
        logger.info("Caching processed features...")
        
        # Process and cache customer features
        customer_features = self.customer_engineer.extract_basic_features(self.customers_df)
        self.customer_feature_cache = {}
        
        for _, row in customer_features.iterrows():
            customer_id = row.name if 'customer_id' not in row.index else row['customer_id']
            if customer_id in self.customers_df['customer_id'].values:
                tensors = self.customer_converter.convert_series(row)
                self.customer_feature_cache[customer_id] = tensors
        
        # Process and cache article features  
        article_features = self.article_engineer.extract_basic_features(self.articles_df)
        self.article_feature_cache = {}
        
        for _, row in article_features.iterrows():
            article_id = row.name if 'article_id' not in row.index else row['article_id'] 
            if article_id in self.articles_df['article_id'].values:
                tensors = self.article_converter.convert_series(row)
                self.article_feature_cache[article_id] = tensors
        
        logger.info(f"Cached features for {len(self.customer_feature_cache)} customers and {len(self.article_feature_cache)} articles")
    
    def __len__(self) -> int:
        """Return dataset size"""
        return len(self.all_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, int]]:
        """
        Get a training sample with customer features, article features, and label.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with features and label
        """
        sample = self.all_samples.iloc[idx]
        
        # Get cached features
        customer_features = self.customer_feature_cache[sample['customer_id']]
        article_features = self.article_feature_cache[sample['article_id']]
        
        return {
            'customer_features': customer_features,
            'article_features': article_features,
            'label': torch.tensor(sample['label'], dtype=torch.float32),
            'customer_id': sample['customer_id'],
            'article_id': sample['article_id']
        }


def create_train_val_datasets(
    customers_df: pd.DataFrame,
    articles_df: pd.DataFrame, 
    transactions_df: pd.DataFrame,
    val_split: float = 0.2,
    negative_samples_per_positive: int = 4,
    random_seed: int = 42
) -> Tuple[TwoTowerDataset, TwoTowerDataset]:
    """
    Create train and validation datasets with temporal or random split.
    
    Args:
        customers_df: Customer data
        articles_df: Article data
        transactions_df: Transaction data
        val_split: Fraction of data for validation
        negative_samples_per_positive: Negative samples per positive
        random_seed: Random seed
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Sort transactions by date for temporal split
    transactions_sorted = transactions_df.sort_values('t_dat')
    
    # Split transactions temporally
    split_idx = int(len(transactions_sorted) * (1 - val_split))
    train_transactions = transactions_sorted.iloc[:split_idx]
    val_transactions = transactions_sorted.iloc[split_idx:]
    
    # Create datasets
    train_dataset = TwoTowerDataset(
        customers_df=customers_df,
        articles_df=articles_df,
        transactions_df=train_transactions,
        negative_samples_per_positive=negative_samples_per_positive,
        random_seed=random_seed
    )
    
    val_dataset = TwoTowerDataset(
        customers_df=customers_df,
        articles_df=articles_df, 
        transactions_df=val_transactions,
        negative_samples_per_positive=negative_samples_per_positive,
        random_seed=random_seed + 1  # Different seed for validation
    )
    
    logger.info(f"Created train dataset: {len(train_dataset):,} samples")
    logger.info(f"Created validation dataset: {len(val_dataset):,} samples")
    
    return train_dataset, val_dataset 