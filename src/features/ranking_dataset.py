"""
Ranking Dataset Creation with Negative Sampling
Implements 1:10 positive:negative ratio as specified in Lesson 2
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.utils import shuffle

logger = logging.getLogger(__name__)


class RankingDatasetCreator:
    """
    Creates ranking dataset with negative sampling for Stage 3 ranking model
    Follows Lesson 2 specifications for balanced positive/negative samples
    """
    
    def __init__(self, negative_ratio: int = 10, random_state: int = 42):
        """
        Initialize ranking dataset creator
        
        Args:
            negative_ratio: Number of negative samples per positive sample
            random_state: Random seed for reproducibility
        """
        self.negative_ratio = negative_ratio
        self.random_state = random_state
        logger.info(f"RankingDatasetCreator initialized with {negative_ratio}:1 negative:positive ratio")
    
    def create_ranking_dataset(self, 
                             transactions_df: pd.DataFrame,
                             customer_features: pd.DataFrame,
                             article_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create ranking dataset with positive and negative samples
        
        Args:
            transactions_df: Transaction data (positive interactions)
            customer_features: Customer features from comprehensive pipeline
            article_features: Article features from comprehensive pipeline
            
        Returns:
            Balanced ranking dataset with labels
        """
        logger.info("Creating ranking dataset with negative sampling...")
        
        # 1. Create positive samples from transactions
        positive_samples = self._create_positive_samples(transactions_df)
        logger.info(f"Created {len(positive_samples)} positive samples")
        
        # 2. Create negative samples
        negative_samples = self._create_negative_samples(
            transactions_df, positive_samples, customer_features, article_features
        )
        logger.info(f"Created {len(negative_samples)} negative samples")
        
        # 3. Combine positive and negative samples
        ranking_dataset = pd.concat([positive_samples, negative_samples], ignore_index=True)
        
        # 4. Join with customer features
        ranking_dataset = ranking_dataset.merge(
            customer_features, on='customer_id', how='left'
        )
        
        # 5. Join with article features
        ranking_dataset = ranking_dataset.merge(
            article_features, on='article_id', how='left'
        )
        
        # 6. Shuffle the dataset
        ranking_dataset = shuffle(ranking_dataset, random_state=self.random_state)
        ranking_dataset = ranking_dataset.reset_index(drop=True)
        
        # 7. Validate the dataset
        self._validate_dataset(ranking_dataset)
        
        logger.info(f"Ranking dataset created: {len(ranking_dataset)} samples")
        logger.info(f"Label distribution: {ranking_dataset['label'].value_counts().to_dict()}")
        
        return ranking_dataset
    
    def _create_positive_samples(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create positive samples from actual transactions
        
        Args:
            transactions_df: Transaction data
            
        Returns:
            DataFrame with positive samples (label=1)
        """
        positive_samples = transactions_df[['customer_id', 'article_id']].copy()
        positive_samples['label'] = 1
        positive_samples = positive_samples.drop_duplicates()
        
        return positive_samples
    
    def _create_negative_samples(self, 
                               transactions_df: pd.DataFrame,
                               positive_samples: pd.DataFrame,
                               customer_features: pd.DataFrame,
                               article_features: pd.DataFrame) -> pd.DataFrame:
        """
        Create negative samples using random sampling strategy
        
        Args:
            transactions_df: Transaction data
            positive_samples: Positive interaction pairs
            customer_features: Available customers
            article_features: Available articles
            
        Returns:
            DataFrame with negative samples (label=0)
        """
        np.random.seed(self.random_state)
        
        # Get unique customers and articles
        unique_customers = customer_features['customer_id'].unique()
        unique_articles = article_features['article_id'].unique()
        
        # Create set of positive pairs for fast lookup
        positive_pairs = set(
            zip(positive_samples['customer_id'], positive_samples['article_id'])
        )
        
        # Calculate number of negative samples needed
        n_negative = len(positive_samples) * self.negative_ratio
        
        negative_samples = []
        attempts = 0
        max_attempts = n_negative * 100  # Prevent infinite loop
        
        while len(negative_samples) < n_negative and attempts < max_attempts:
            # Randomly sample customer and article
            customer_id = np.random.choice(unique_customers)
            article_id = np.random.choice(unique_articles)
            
            # Check if this is not a positive pair
            if (customer_id, article_id) not in positive_pairs:
                negative_samples.append({
                    'customer_id': customer_id,
                    'article_id': article_id,
                    'label': 0
                })
            
            attempts += 1
        
        if len(negative_samples) < n_negative:
            logger.warning(f"Only generated {len(negative_samples)} negative samples out of {n_negative} requested")
        
        return pd.DataFrame(negative_samples)
    
    def _validate_dataset(self, ranking_dataset: pd.DataFrame) -> None:
        """
        Validate the ranking dataset
        
        Args:
            ranking_dataset: Created ranking dataset
        """
        # Check label distribution
        label_counts = ranking_dataset['label'].value_counts()
        positive_count = label_counts.get(1, 0)
        negative_count = label_counts.get(0, 0)
        
        if positive_count == 0:
            raise ValueError("No positive samples in ranking dataset")
        
        actual_ratio = negative_count / positive_count if positive_count > 0 else 0
        expected_ratio = self.negative_ratio
        
        if abs(actual_ratio - expected_ratio) > 1.0:  # Allow some tolerance
            logger.warning(f"Negative ratio {actual_ratio:.1f} differs from expected {expected_ratio}")
        
        # Check for missing values in key columns
        required_columns = ['customer_id', 'article_id', 'label']
        for col in required_columns:
            if ranking_dataset[col].isna().any():
                raise ValueError(f"Missing values found in {col}")
        
        logger.info("✅ Ranking dataset validation passed")
    
    def get_dataset_statistics(self, ranking_dataset: pd.DataFrame) -> Dict[str, any]:
        """
        Get statistics about the ranking dataset
        
        Args:
            ranking_dataset: Ranking dataset
            
        Returns:
            Dictionary with dataset statistics
        """
        label_counts = ranking_dataset['label'].value_counts()
        
        stats = {
            'total_samples': len(ranking_dataset),
            'positive_samples': label_counts.get(1, 0),
            'negative_samples': label_counts.get(0, 0),
            'negative_ratio': label_counts.get(0, 0) / label_counts.get(1, 1),
            'unique_customers': ranking_dataset['customer_id'].nunique(),
            'unique_articles': ranking_dataset['article_id'].nunique(),
            'feature_count': len(ranking_dataset.columns) - 3  # Exclude customer_id, article_id, label
        }
        
        return stats