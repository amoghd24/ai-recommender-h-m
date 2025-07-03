"""
Customer Feature Engineering Module for H&M Recommender System
Extracts demographic and behavioral features from customer data
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CustomerFeatureEngineer:
    """
    Extract and engineer features from customer data
    """
    
    def __init__(self, age_bins: Optional[list] = None):
        """
        Initialize CustomerFeatureEngineer
        
        Args:
            age_bins: List of age bin edges for age segmentation
        """
        if age_bins is None:
            age_bins = [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 100]
        
        self.age_bins = age_bins
        self.age_labels = [f"{age_bins[i]}-{age_bins[i+1]}" for i in range(len(age_bins)-1)]
        logger.info("CustomerFeatureEngineer initialized")
    
    def extract_basic_features(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic demographic features from customer data
        
        Args:
            customers_df: Raw customers DataFrame
            
        Returns:
            DataFrame with engineered customer features
        """
        logger.info("Extracting basic customer features...")
        
        # Create a copy to avoid modifying original
        df = customers_df.copy()
        
        # 1. Age-based features
        df['age_group'] = pd.cut(
            df['age'], 
            bins=self.age_bins, 
            labels=self.age_labels,
            include_lowest=True
        )
        
        # Fill missing ages with median before binning
        median_age = df['age'].median()
        df['age_filled'] = df['age'].fillna(median_age)
        df['is_age_missing'] = df['age'].isna().astype(int)
        
        # 2. Club member features
        df['is_club_member'] = (df['club_member_status'] == 'ACTIVE').astype(int)
        # Handle category dtype properly
        if df['club_member_status'].dtype.name == 'category':
            df['club_member_status_filled'] = df['club_member_status'].astype(str).fillna('UNKNOWN')
        else:
            df['club_member_status_filled'] = df['club_member_status'].fillna('UNKNOWN')
        
        # 3. Fashion news features
        df['receives_fashion_news'] = (~df['fashion_news_frequency'].isna()).astype(int)
        # Handle category dtype properly
        if df['fashion_news_frequency'].dtype.name == 'category':
            df['fashion_news_frequency_filled'] = df['fashion_news_frequency'].astype(str).fillna('NONE')
        else:
            df['fashion_news_frequency_filled'] = df['fashion_news_frequency'].fillna('NONE')
        
        # Map frequency to numeric scale
        news_freq_map = {
            'NONE': 0,
            'Monthly': 1,
            'Regularly': 2
        }
        df['fashion_news_score'] = df['fashion_news_frequency_filled'].map(news_freq_map).fillna(0)
        
        # 4. Customer activity features
        df['is_active'] = df['Active'].fillna(0).astype(int)
        df['has_FN'] = df['FN'].fillna(0).astype(int)
        
        # 5. Create customer lifecycle stage based on multiple factors
        df['lifecycle_stage'] = self._determine_lifecycle_stage(df)
        
        logger.info(f"Basic features extracted for {len(df)} customers")
        return df
    
    def _determine_lifecycle_stage(self, df: pd.DataFrame) -> pd.Series:
        """
        Determine customer lifecycle stage based on available features
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            Series with lifecycle stage labels
        """
        conditions = [
            (df['is_active'] == 0),
            (df['is_club_member'] == 1) & (df['fashion_news_score'] >= 2),
            (df['is_club_member'] == 1) & (df['fashion_news_score'] < 2),
            (df['is_club_member'] == 0) & (df['receives_fashion_news'] == 1),
        ]
        
        choices = ['inactive', 'loyal', 'engaged', 'casual']
        
        return pd.Series(
            np.select(conditions, choices, default='new'),
            index=df.index
        ) 