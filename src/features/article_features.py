"""
Article Feature Engineering Module for H&M Recommender System
Extracts content-based and categorical features from article data
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArticleFeatureEngineer:
    """
    Extract and engineer features from article/product data
    """
    
    def __init__(self, price_bins: Optional[List[float]] = None):
        """
        Initialize ArticleFeatureEngineer
        
        Args:
            price_bins: List of price bin edges for price segmentation
        """
        if price_bins is None:
            # Price bins in the normalized scale (0-1)
            price_bins = [0, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
        
        self.price_bins = price_bins
        self.price_labels = [f"{price_bins[i]:.2f}-{price_bins[i+1]:.2f}" for i in range(len(price_bins)-1)]
        logger.info("ArticleFeatureEngineer initialized")
    
    def extract_basic_features(self, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract basic product features from article data
        
        Args:
            articles_df: Raw articles DataFrame
            
        Returns:
            DataFrame with engineered article features
        """
        logger.info("Extracting basic article features...")
        
        # Create a copy to avoid modifying original
        df = articles_df.copy()
        
        # 1. Hierarchical category features
        df['department_idx'] = pd.Categorical(df['department_name']).codes
        df['product_group_idx'] = pd.Categorical(df['product_group_name']).codes
        df['product_type_idx'] = pd.Categorical(df['product_type_name']).codes
        
        # Create hierarchical category path
        df['category_path'] = (
            df['department_name'].astype(str) + '/' + 
            df['product_group_name'].astype(str) + '/' + 
            df['product_type_name'].astype(str)
        )
        
        # 2. Color features
        df['is_black'] = (df['colour_group_name'] == 'Black').astype(int)
        df['is_white'] = (df['colour_group_name'] == 'White').astype(int)
        df['is_neutral_color'] = df['colour_group_name'].isin(['Black', 'White', 'Grey', 'Beige']).astype(int)
        
        # 3. Style features
        df['has_graphic'] = (df['graphical_appearance_name'] != 'Solid').astype(int)
        df['is_patterned'] = df['graphical_appearance_name'].isin(['Striped', 'Spotted', 'Checked']).astype(int)
        
        # 4. Index features (product categorization)
        df['is_menswear'] = df['index_name'].str.contains('Men', case=False, na=False).astype(int)
        df['is_kidswear'] = df['index_name'].str.contains('Children|Baby', case=False, na=False).astype(int)
        df['is_womenswear'] = ((df['is_menswear'] == 0) & (df['is_kidswear'] == 0)).astype(int)
        
        # 5. Section features (store section)
        df['is_divided'] = (df['section_name'] == 'Divided').astype(int)
        df['section_idx'] = pd.Categorical(df['section_name']).codes
        
        # 6. Garment type features
        df['is_upper_body'] = df['garment_group_name'].isin(['Jersey Basic', 'Jersey Fancy', 'Knitwear']).astype(int)
        df['is_lower_body'] = df['garment_group_name'].isin(['Trousers', 'Shorts', 'Skirts']).astype(int)
        df['is_full_body'] = df['garment_group_name'].isin(['Dresses', 'Jumpsuit']).astype(int)
        
        # 7. Description features
        df['has_description'] = (~df['detail_desc'].isna()).astype(int)
        df['description_length'] = df['detail_desc'].fillna('').str.len()
        
        logger.info(f"Basic features extracted for {len(df)} articles")
        return df
    
    def extract_price_features(self, articles_df: pd.DataFrame, transactions_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract price-related features
        
        Args:
            articles_df: Articles DataFrame with basic features
            transactions_df: Optional transactions data for actual price statistics
            
        Returns:
            DataFrame with price features added
        """
        logger.info("Extracting price features...")
        
        df = articles_df.copy()
        
        # If we have transaction data, calculate actual price statistics
        if transactions_df is not None:
            price_stats = transactions_df.groupby('article_id')['price'].agg(['mean', 'std', 'min', 'max'])
            price_stats.columns = ['avg_selling_price', 'price_std', 'min_price', 'max_price']
            
            # Merge with articles
            df = df.merge(price_stats, left_on='article_id', right_index=True, how='left')
            
            # Price volatility
            df['price_volatility'] = df['price_std'] / df['avg_selling_price']
            df['price_volatility'] = df['price_volatility'].fillna(0)
            
            # Use average selling price for binning
            price_col = 'avg_selling_price'
        else:
            # Use a dummy price column if no transaction data
            df['avg_selling_price'] = 0.05  # Default middle price
            price_col = 'avg_selling_price'
        
        # Price binning
        df['price_range'] = pd.cut(
            df[price_col], 
            bins=self.price_bins,
            labels=self.price_labels,
            include_lowest=True
        )
        
        logger.info(f"Price features extracted for {len(df)} articles")
        return df 