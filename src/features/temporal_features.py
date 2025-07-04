"""
Temporal Feature Engineering Module for H&M Recommender System
Extracts time-based and seasonal patterns from transaction data
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """
    Extract temporal and seasonal features from transaction data
    """
    
    def __init__(self, reference_date: Optional[str] = None):
        """
        Initialize TemporalFeatureEngineer
        
        Args:
            reference_date: Reference date for recency calculations (YYYY-MM-DD)
        """
        if reference_date is None:
            reference_date = '2020-09-22'  # Default to last date in H&M dataset
        
        self.reference_date = pd.to_datetime(reference_date)
        logger.info(f"TemporalFeatureEngineer initialized with reference date: {self.reference_date}")
    
    def extract_customer_temporal_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features for customers based on their transaction history
        
        Args:
            transactions_df: Transaction data with customer_id and t_dat columns
            
        Returns:
            DataFrame with customer temporal features
        """
        logger.info("Extracting customer temporal features...")
        
        # Ensure date column is datetime
        df = transactions_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['t_dat']):
            df['t_dat'] = pd.to_datetime(df['t_dat'])
        
        # Group by customer
        customer_features = []
        
        for customer_id, customer_data in df.groupby('customer_id'):
            features = {'customer_id': customer_id}
            
            # 1. Recency features
            last_purchase_date = customer_data['t_dat'].max()
            first_purchase_date = customer_data['t_dat'].min()
            
            features['days_since_last_purchase'] = (self.reference_date - last_purchase_date).days
            features['days_since_first_purchase'] = (self.reference_date - first_purchase_date).days
            features['customer_lifespan_days'] = (last_purchase_date - first_purchase_date).days
            
            # 2. Purchase frequency patterns
            features['total_transactions'] = len(customer_data)
            features['avg_days_between_purchases'] = (
                features['customer_lifespan_days'] / features['total_transactions'] 
                if features['total_transactions'] > 1 else 0
            )
            
            # 3. Seasonal patterns
            customer_data['month'] = customer_data['t_dat'].dt.month
            customer_data['weekday'] = customer_data['t_dat'].dt.dayofweek
            customer_data['is_weekend'] = customer_data['weekday'].isin([5, 6])
            
            features['favorite_month'] = customer_data['month'].mode().iloc[0] if len(customer_data) > 0 else 0
            features['weekend_purchase_ratio'] = customer_data['is_weekend'].mean()
            
            # 4. Activity patterns
            features['purchases_last_30_days'] = len(customer_data[customer_data['t_dat'] >= self.reference_date - timedelta(days=30)])
            features['purchases_last_90_days'] = len(customer_data[customer_data['t_dat'] >= self.reference_date - timedelta(days=90)])
            features['is_active_customer'] = features['purchases_last_90_days'] > 0
            
            customer_features.append(features)
        
        result_df = pd.DataFrame(customer_features)
        logger.info(f"Temporal features extracted for {len(result_df)} customers")
        return result_df
    
    def extract_article_temporal_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features for articles based on sales patterns
        
        Args:
            transactions_df: Transaction data with article_id and t_dat columns
            
        Returns:
            DataFrame with article temporal features
        """
        logger.info("Extracting article temporal features...")
        
        # Ensure date column is datetime
        df = transactions_df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['t_dat']):
            df['t_dat'] = pd.to_datetime(df['t_dat'])
        
        # Group by article
        article_features = []
        
        for article_id, article_data in df.groupby('article_id'):
            features = {'article_id': article_id}
            
            # 1. Sales velocity
            first_sale_date = article_data['t_dat'].min()
            last_sale_date = article_data['t_dat'].max()
            
            features['days_since_first_sale'] = (self.reference_date - first_sale_date).days
            features['days_since_last_sale'] = (self.reference_date - last_sale_date).days
            features['product_lifespan_days'] = (last_sale_date - first_sale_date).days
            
            # 2. Sales patterns
            features['total_sales'] = len(article_data)
            features['unique_customers'] = article_data['customer_id'].nunique()
            features['avg_sales_per_day'] = (
                features['total_sales'] / (features['product_lifespan_days'] + 1)
            )
            
            # 3. Trend features
            features['sales_last_7_days'] = len(article_data[article_data['t_dat'] >= self.reference_date - timedelta(days=7)])
            features['sales_last_30_days'] = len(article_data[article_data['t_dat'] >= self.reference_date - timedelta(days=30)])
            features['is_trending'] = features['sales_last_7_days'] > features['avg_sales_per_day'] * 7
            features['is_new_product'] = features['days_since_first_sale'] <= 30
            
            article_features.append(features)
        
        result_df = pd.DataFrame(article_features)
        logger.info(f"Temporal features extracted for {len(result_df)} articles")
        return result_df 