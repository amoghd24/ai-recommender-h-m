"""
Stage 2: Filtering Module for 4-Stage Recommender
Filters retrieved candidates based on business rules and user history
"""

import numpy as np
import pandas as pd
from typing import List, Set, Dict, Optional
from pybloom_live import BloomFilter
import logging

logger = logging.getLogger(__name__)


class FilteringStage:
    """
    Implements Stage 2 filtering with Bloom filters for efficient filtering
    """
    
    def __init__(
        self,
        bloom_capacity: int = 1_000_000,
        bloom_error_rate: float = 0.001
    ):
        """
        Initialize filtering stage
        
        Args:
            bloom_capacity: Expected number of items in Bloom filter
            bloom_error_rate: False positive probability
        """
        self.bloom_capacity = bloom_capacity
        self.bloom_error_rate = bloom_error_rate
        self.user_bloom_filters: Dict[str, BloomFilter] = {}
        
        logger.info(f"FilteringStage initialized with capacity={bloom_capacity}, error_rate={bloom_error_rate}")
    
    def update_user_history(self, customer_id: str, article_ids: List[str]) -> None:
        """
        Update user's purchase/view history in Bloom filter
        
        Args:
            customer_id: Customer identifier
            article_ids: List of article IDs to add to history
        """
        if customer_id not in self.user_bloom_filters:
            self.user_bloom_filters[customer_id] = BloomFilter(
                capacity=self.bloom_capacity,
                error_rate=self.bloom_error_rate
            )
        
        bloom = self.user_bloom_filters[customer_id]
        for article_id in article_ids:
            bloom.add(article_id)
    
    def filter_candidates(
        self,
        candidates: pd.DataFrame,
        customer_id: str,
        stock_data: Optional[pd.DataFrame] = None,
        business_rules: Optional[Dict] = None
    ) -> pd.DataFrame:
        """
        Filter candidate articles based on multiple criteria
        
        Args:
            candidates: DataFrame with candidate articles
            customer_id: Customer identifier
            stock_data: Optional stock availability data
            business_rules: Optional business filtering rules
            
        Returns:
            Filtered DataFrame of candidates
        """
        initial_count = len(candidates)
        filtered = candidates.copy()
        
        # 1. Filter seen/purchased items using Bloom filter
        if customer_id in self.user_bloom_filters:
            bloom = self.user_bloom_filters[customer_id]
            seen_mask = filtered['article_id'].apply(lambda x: x not in bloom)
            filtered = filtered[seen_mask]
            logger.debug(f"Filtered {initial_count - len(filtered)} seen items")
        
        # 2. Filter out-of-stock items
        if stock_data is not None:
            in_stock = stock_data[stock_data['in_stock'] == True]['article_id'].tolist()
            filtered = filtered[filtered['article_id'].isin(in_stock)]
            logger.debug(f"Filtered to {len(filtered)} in-stock items")
        
        # 3. Apply business rules
        if business_rules:
            filtered = self._apply_business_rules(filtered, customer_id, business_rules)
        
        logger.info(f"Filtered {initial_count} candidates to {len(filtered)} for customer {customer_id}")
        return filtered
    
    def _apply_business_rules(
        self,
        candidates: pd.DataFrame,
        customer_id: str,
        rules: Dict
    ) -> pd.DataFrame:
        """
        Apply custom business filtering rules
        
        Args:
            candidates: Candidate articles
            customer_id: Customer identifier
            rules: Business rules dictionary
            
        Returns:
            Filtered candidates
        """
        filtered = candidates.copy()
        
        # Price range filtering
        if 'price_range' in rules:
            min_price, max_price = rules['price_range']
            filtered = filtered[
                (filtered['price'] >= min_price) & 
                (filtered['price'] <= max_price)
            ]
        
        # Department filtering
        if 'allowed_departments' in rules:
            filtered = filtered[
                filtered['department_name'].isin(rules['allowed_departments'])
            ]
        
        # Age-appropriate filtering
        if 'customer_age_group' in rules:
            age_group = rules['customer_age_group']
            if age_group == 'kids':
                filtered = filtered[filtered['is_kidswear'] == 1]
            elif age_group == 'adult':
                filtered = filtered[filtered['is_kidswear'] == 0]
        
        # Regional availability
        if 'region' in rules and 'available_regions' in filtered.columns:
            region = rules['region']
            filtered = filtered[
                filtered['available_regions'].apply(lambda x: region in x if pd.notna(x) else True)
            ]
        
        return filtered
    
    def batch_update_history(self, transactions_df: pd.DataFrame) -> None:
        """
        Batch update user histories from transaction data
        
        Args:
            transactions_df: DataFrame with customer_id and article_id columns
        """
        logger.info("Batch updating user histories...")
        
        for customer_id, group in transactions_df.groupby('customer_id'):
            article_ids = group['article_id'].tolist()
            self.update_user_history(str(customer_id), article_ids)
        
        logger.info(f"Updated histories for {len(self.user_bloom_filters)} customers")
    
    def get_filter_stats(self, customer_id: str) -> Dict[str, int]:
        """
        Get filtering statistics for a customer
        
        Args:
            customer_id: Customer identifier
            
        Returns:
            Dictionary with filter statistics
        """
        stats = {
            'has_history': customer_id in self.user_bloom_filters,
            'bloom_item_count': 0
        }
        
        if customer_id in self.user_bloom_filters:
            # Bloom filters don't have exact counts, but we can estimate
            bloom = self.user_bloom_filters[customer_id]
            stats['bloom_item_count'] = bloom.count
        
        return stats 