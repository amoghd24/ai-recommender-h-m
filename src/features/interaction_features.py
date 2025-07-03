"""
Interaction Feature Engineering Module for H&M Recommender System
Extracts features that capture the relationship between customers and articles
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InteractionFeatureEngineer:
    """
    Extract features that capture customer-article interactions
    """
    
    def __init__(self):
        """
        Initialize InteractionFeatureEngineer
        """
        logger.info("InteractionFeatureEngineer initialized")
    
    def extract_customer_preferences(self, transactions_df: pd.DataFrame, articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract customer preference features based on their purchase history
        
        Args:
            transactions_df: Transaction data
            articles_df: Article data with product attributes
            
        Returns:
            DataFrame with customer preference features
        """
        logger.info("Extracting customer preference features...")
        
        # Merge transactions with article attributes
        trans_with_articles = transactions_df.merge(
            articles_df[['article_id', 'product_group_name', 'department_name', 
                        'colour_group_name', 'index_name']], 
            on='article_id', 
            how='left'
        )
        
        customer_prefs = []
        
        for customer_id, customer_data in trans_with_articles.groupby('customer_id'):
            prefs = {'customer_id': customer_id}
            
            # 1. Department preferences
            dept_counts = customer_data['department_name'].value_counts()
            prefs['favorite_department'] = dept_counts.index[0] if len(dept_counts) > 0 else 'Unknown'
            prefs['department_diversity'] = len(dept_counts)
            prefs['department_concentration'] = dept_counts.iloc[0] / len(customer_data) if len(customer_data) > 0 else 0
            
            # 2. Product group preferences  
            prod_counts = customer_data['product_group_name'].value_counts()
            prefs['favorite_product_group'] = prod_counts.index[0] if len(prod_counts) > 0 else 'Unknown'
            prefs['product_group_diversity'] = len(prod_counts)
            
            # 3. Color preferences
            color_counts = customer_data['colour_group_name'].value_counts()
            prefs['favorite_color'] = color_counts.index[0] if len(color_counts) > 0 else 'Unknown'
            prefs['color_diversity'] = len(color_counts)
            prefs['prefers_black'] = (prefs['favorite_color'] == 'Black')
            
            # 4. Price preferences
            prefs['avg_purchase_price'] = customer_data['price'].mean()
            prefs['price_std'] = customer_data['price'].std()
            prefs['min_purchase_price'] = customer_data['price'].min()
            prefs['max_purchase_price'] = customer_data['price'].max()
            prefs['price_range'] = prefs['max_purchase_price'] - prefs['min_purchase_price']
            
            # 5. Gender preferences (based on index_name)
            prefs['menswear_ratio'] = customer_data['index_name'].str.contains('Men', case=False, na=False).mean()
            prefs['kidswear_ratio'] = customer_data['index_name'].str.contains('Children|Baby', case=False, na=False).mean()
            prefs['womenswear_ratio'] = 1 - prefs['menswear_ratio'] - prefs['kidswear_ratio']
            
            customer_prefs.append(prefs)
        
        result_df = pd.DataFrame(customer_prefs)
        logger.info(f"Customer preferences extracted for {len(result_df)} customers")
        return result_df
    
    def extract_article_popularity(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract article popularity features based on sales patterns
        
        Args:
            transactions_df: Transaction data
            
        Returns:
            DataFrame with article popularity features
        """
        logger.info("Extracting article popularity features...")
        
        # Basic popularity metrics
        article_stats = transactions_df.groupby('article_id').agg({
            'customer_id': ['count', 'nunique'],
            'price': ['mean', 'std'],
            't_dat': ['min', 'max']
        })
        
        # Flatten column names
        article_stats.columns = ['_'.join(col).strip() for col in article_stats.columns.values]
        article_stats = article_stats.reset_index()
        
        # Rename columns for clarity
        article_stats.rename(columns={
            'customer_id_count': 'total_purchases',
            'customer_id_nunique': 'unique_customers',
            'price_mean': 'avg_selling_price',
            'price_std': 'price_volatility',
            't_dat_min': 'first_purchase_date',
            't_dat_max': 'last_purchase_date'
        }, inplace=True)
        
        # Calculate derived metrics
        article_stats['repurchase_rate'] = (
            article_stats['total_purchases'] / article_stats['unique_customers']
        )
        article_stats['popularity_score'] = (
            article_stats['total_purchases'] * article_stats['unique_customers']
        ) ** 0.5
        
        # Popularity rank
        article_stats['popularity_rank'] = article_stats['popularity_score'].rank(ascending=False, method='dense')
        article_stats['is_bestseller'] = article_stats['popularity_rank'] <= 100
        
        logger.info(f"Article popularity features extracted for {len(article_stats)} articles")
        return article_stats
    
    def create_customer_article_affinity(self, transactions_df: pd.DataFrame, 
                                       customer_prefs_df: pd.DataFrame,
                                       articles_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create affinity scores between customers and articles based on preferences
        
        Args:
            transactions_df: Transaction data
            customer_prefs_df: Customer preference features
            articles_df: Article data with attributes
            
        Returns:
            DataFrame with customer-article affinity scores
        """
        logger.info("Creating customer-article affinity scores...")
        
        # For demonstration, calculate simple affinity based on department match
        # In practice, this would be more sophisticated
        
        # Get customer IDs that exist in both dataframes
        common_customers = set(transactions_df['customer_id'].unique()) & set(customer_prefs_df['customer_id'].unique())
        customers = list(common_customers)[:100]  # Limit for demo
        
        # Get article IDs from transactions that exist in articles_df
        common_articles = set(transactions_df['article_id'].unique()) & set(articles_df['article_id'].unique())
        articles = list(common_articles)[:50]      # Limit for demo
        
        affinities = []
        
        for customer_id in customers:
            # Get customer's favorite department
            cust_pref = customer_prefs_df[customer_prefs_df['customer_id'] == customer_id]
            if len(cust_pref) == 0:
                continue
                
            fav_dept = cust_pref.iloc[0]['favorite_department']
            
            for article_id in articles:
                # Get article's department
                art_info = articles_df[articles_df['article_id'] == article_id]
                if len(art_info) == 0:
                    continue
                    
                art_dept = art_info.iloc[0]['department_name']
                
                # Simple affinity: 1 if departments match, 0.5 otherwise
                affinity_score = 1.0 if fav_dept == art_dept else 0.5
                
                affinities.append({
                    'customer_id': customer_id,
                    'article_id': article_id,
                    'department_affinity': affinity_score
                })
        
        # If no affinities were created, return empty DataFrame with proper columns
        if not affinities:
            result_df = pd.DataFrame(columns=['customer_id', 'article_id', 'department_affinity'])
        else:
            result_df = pd.DataFrame(affinities)
            
        logger.info(f"Created {len(result_df)} customer-article affinity scores")
        return result_df 