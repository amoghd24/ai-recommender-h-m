"""
Two-Tower Model Feature Configuration for H&M Recommender System
Minimal feature set as specified in Lesson 2 for retrieval model
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Two-tower model feature specifications from Lesson 2
TWO_TOWER_CUSTOMER_FEATURES = {
    'customer_id': {
        'type': 'categorical',
        'source_column': 'customer_id'
    },
    'age': {
        'type': 'numerical',
        'source_column': 'age',
        'normalize': True,
        'fillna': 30
    },
    'month_sin': {
        'type': 'numerical',
        'source_column': 'month_sin',
        'normalize': False,
        'fillna': 0
    },
    'month_cos': {
        'type': 'numerical',
        'source_column': 'month_cos',
        'normalize': False,
        'fillna': 0
    }
}

TWO_TOWER_ARTICLE_FEATURES = {
    'article_id': {
        'type': 'categorical',
        'source_column': 'article_id'
    },
    'garment_group_name': {
        'type': 'categorical',
        'source_column': 'garment_group_name',
        'num_categories': 30
    },
    'index_group_name': {
        'type': 'categorical',
        'source_column': 'index_group_name',
        'num_categories': 10
    }
}


class TwoTowerFeatureExtractor:
    """
    Extract minimal feature set for two-tower retrieval model
    Following Lesson 2 specifications
    """
    
    def __init__(self, reference_date: str = '2020-09-22'):
        """
        Initialize extractor with reference date for temporal features
        
        Args:
            reference_date: Reference date for cyclic encoding
        """
        self.reference_date = pd.to_datetime(reference_date)
        logger.info(f"TwoTowerFeatureExtractor initialized with reference date: {self.reference_date}")
    
    def extract_customer_features(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract minimal customer features for two-tower model
        
        Args:
            customers_df: Customer data with basic demographic info
            
        Returns:
            DataFrame with customer_id, age, month_sin, month_cos
        """
        logger.info("Extracting two-tower customer features...")
        
        df = customers_df.copy()
        
        # Select required features
        result_df = pd.DataFrame()
        result_df['customer_id'] = df['customer_id']
        
        # Age feature (numerical)
        result_df['age'] = df['age'].fillna(30)
        
        # Cyclic temporal encoding based on reference date
        current_month = self.reference_date.month
        result_df['month_sin'] = np.sin(2 * np.pi * current_month / 12)
        result_df['month_cos'] = np.cos(2 * np.pi * current_month / 12)
        
        logger.info(f"Extracted two-tower customer features for {len(result_df)} customers")
        return result_df
    
    def extract_article_features(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract minimal article features for two-tower model
        
        Args:
            articles_df: Article data with product information
            
        Returns:
            DataFrame with article_id, garment_group_name, index_group_name
        """
        logger.info("Extracting two-tower article features...")
        
        df = articles_df.copy()
        
        # Select required features
        result_df = pd.DataFrame()
        result_df['article_id'] = df['article_id']
        
        # Handle categorical features with missing values
        result_df['garment_group_name'] = df['garment_group_name'].astype(str).fillna('Unknown')
        result_df['index_group_name'] = df['index_group_name'].astype(str).fillna('Unknown')
        
        logger.info(f"Extracted two-tower article features for {len(result_df)} articles")
        return result_df
    
    def get_customer_feature_names(self) -> List[str]:
        """Get list of customer feature names for two-tower model"""
        return list(TWO_TOWER_CUSTOMER_FEATURES.keys())
    
    def get_article_feature_names(self) -> List[str]:
        """Get list of article feature names for two-tower model"""
        return list(TWO_TOWER_ARTICLE_FEATURES.keys())
    
    def get_categorical_sizes(self) -> Dict[str, int]:
        """
        Get categorical feature sizes for embedding layers
        This needs to be computed from actual data
        """
        # These will be computed dynamically from the data
        return {
            'customer_id': 1_500_000,  # Max customers in H&M dataset
            'article_id': 110_000,     # Max articles in H&M dataset
            'garment_group_name': 30,
            'index_group_name': 10
        }


def get_two_tower_feature_config() -> Dict[str, Dict[str, Any]]:
    """Get complete two-tower feature configuration"""
    return {
        'customer': TWO_TOWER_CUSTOMER_FEATURES,
        'article': TWO_TOWER_ARTICLE_FEATURES
    }