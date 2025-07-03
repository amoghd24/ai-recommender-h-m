"""
Feature Engineering Pipeline for H&M Recommender System
Combines all feature engineering components into a unified pipeline
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from datetime import datetime

from features.customer_features import CustomerFeatureEngineer
from features.article_features import ArticleFeatureEngineer
from features.temporal_features import TemporalFeatureEngineer
from features.interaction_features import InteractionFeatureEngineer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeaturePipeline:
    """
    Orchestrates all feature engineering components into a unified pipeline
    """
    
    def __init__(self, reference_date: Optional[str] = None):
        """
        Initialize FeaturePipeline with all feature engineers
        
        Args:
            reference_date: Reference date for temporal features (default: 2020-09-22)
        """
        self.customer_engineer = CustomerFeatureEngineer()
        self.article_engineer = ArticleFeatureEngineer()
        self.temporal_engineer = TemporalFeatureEngineer(reference_date)
        self.interaction_engineer = InteractionFeatureEngineer()
        
        logger.info("FeaturePipeline initialized with all feature engineers")
    
    def create_customer_features(self, customers_df: pd.DataFrame, 
                               transactions_df: pd.DataFrame,
                               articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive customer features
        
        Args:
            customers_df: Customer data
            transactions_df: Transaction data
            articles_df: Article data
            
        Returns:
            DataFrame with all customer features
        """
        logger.info("Creating comprehensive customer features...")
        
        # 1. Basic customer features
        customer_features = self.customer_engineer.extract_basic_features(customers_df)
        
        # 2. Customer temporal features
        customer_temporal = self.temporal_engineer.extract_customer_temporal_features(
            transactions_df
        )
        
        # 3. Customer preferences from interactions
        customer_prefs = self.interaction_engineer.extract_customer_preferences(
            transactions_df, articles_df
        )
        
        # Merge all customer features
        result = customer_features
        
        if len(customer_temporal) > 0:
            result = result.merge(customer_temporal, on='customer_id', how='left')
            
        if len(customer_prefs) > 0:
            result = result.merge(customer_prefs, on='customer_id', how='left')
        
        logger.info(f"Created features for {len(result)} customers with {len(result.columns)} features")
        return result
    
    def create_article_features(self, articles_df: pd.DataFrame,
                              transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive article features
        
        Args:
            articles_df: Article data
            transactions_df: Transaction data
            
        Returns:
            DataFrame with all article features
        """
        logger.info("Creating comprehensive article features...")
        
        # 1. Basic article features
        article_features = self.article_engineer.extract_basic_features(articles_df)
        
        # 2. Article temporal features
        article_temporal = self.temporal_engineer.extract_article_temporal_features(
            transactions_df
        )
        
        # 3. Article popularity features
        article_popularity = self.interaction_engineer.extract_article_popularity(transactions_df)
        
        # Merge all article features
        result = article_features
        
        if len(article_temporal) > 0:
            result = result.merge(article_temporal, on='article_id', how='left')
            
        if len(article_popularity) > 0:
            result = result.merge(article_popularity, on='article_id', how='left')
        
        logger.info(f"Created features for {len(result)} articles with {len(result.columns)} features")
        return result
    
    def create_training_features(self, customers_df: pd.DataFrame,
                               articles_df: pd.DataFrame,
                               transactions_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create all features for model training
        
        Args:
            customers_df: Customer data
            articles_df: Article data  
            transactions_df: Transaction data
            
        Returns:
            Tuple of (customer_features, article_features)
        """
        logger.info("Creating comprehensive feature set for training...")
        
        # Create customer features
        customer_features = self.create_customer_features(
            customers_df, transactions_df, articles_df
        )
        
        # Create article features
        article_features = self.create_article_features(
            articles_df, transactions_df
        )
        
        # Log feature summary
        logger.info(f"\nFeature Summary:")
        logger.info(f"- Customer features: {len(customer_features)} customers, {len(customer_features.columns)} features")
        logger.info(f"- Article features: {len(article_features)} articles, {len(article_features.columns)} features")
        
        return customer_features, article_features
    
    def get_feature_names(self) -> Dict[str, list]:
        """
        Get names of all features by category
        
        Returns:
            Dictionary mapping feature categories to feature names
        """
        return {
            'customer_demographic': [
                'age', 'age_group', 'club_member_status', 'fashion_news_active',
                'customer_lifecycle_stage'
            ],
            'customer_temporal': [
                'days_since_last_purchase', 'purchase_frequency', 'avg_days_between_purchases',
                'is_active_last_30_days', 'seasonal_shopper'
            ],
            'customer_preference': [
                'favorite_department', 'department_diversity', 'favorite_color',
                'avg_purchase_price', 'price_range'
            ],
            'article_product': [
                'department_name', 'product_group_name', 'is_menswear', 'is_kidswear',
                'is_womenswear', 'garment_type', 'price_bin'
            ],
            'article_temporal': [
                'days_since_first_sale', 'sales_velocity', 'is_trending',
                'product_lifecycle_stage'
            ],
            'article_popularity': [
                'total_purchases', 'unique_customers', 'repurchase_rate',
                'popularity_score', 'is_bestseller'
            ]
        } 