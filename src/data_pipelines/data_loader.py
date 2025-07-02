"""
Data Loader Module for H&M Recommender System
Provides efficient data loading with memory optimization and validation
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Efficient data loader with memory optimization and validation
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize DataLoader with configuration
        
        Args:
            config_path: Path to configuration YAML file
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "data_config.yaml"
        
        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent.parent  # Get project root from config location
        self.config = self._load_config()
        self.data_paths = self._resolve_data_paths(self.config['data_paths'])
        
        # Define optimized data types for memory efficiency
        self.dtypes = self._get_optimized_dtypes()
        
        logger.info(f"DataLoader initialized with config: {self.config_path}")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Configuration loaded from {self.config_path}")
                return config
        except Exception as e:
            logger.error(f"Failed to load config from {self.config_path}: {e}")
            raise
    
    def _resolve_data_paths(self, data_paths: Dict) -> Dict:
        """
        Resolve data paths relative to project root
        
        Args:
            data_paths: Dictionary with potentially relative paths
            
        Returns:
            Dictionary with resolved absolute paths
        """
        resolved_paths = {}
        
        for key, path in data_paths.items():
            if isinstance(path, str):
                path_obj = Path(path)
                if not path_obj.is_absolute():
                    # Resolve relative to project root
                    resolved_path = self.project_root / path
                else:
                    resolved_path = path_obj
                resolved_paths[key] = str(resolved_path)
            else:
                resolved_paths[key] = path
        
        return resolved_paths
    
    def _get_optimized_dtypes(self) -> Dict:
        """
        Define optimized data types for memory efficiency
        
        Returns:
            Dictionary mapping dataset names to their optimal dtypes
        """
        return {
            'customers': {
                'customer_id': 'string',
                'FN': 'float32',
                'Active': 'float32', 
                'club_member_status': 'category',
                'fashion_news_frequency': 'category',
                'age': 'float32',
                'postal_code': 'string'
            },
            'articles': {
                'article_id': 'int32',
                'product_code': 'int32',
                'prod_name': 'string',
                'product_type_no': 'int16',
                'product_type_name': 'category',
                'product_group_name': 'category',
                'graphical_appearance_no': 'int16',
                'graphical_appearance_name': 'category',
                'colour_group_code': 'int8',
                'colour_group_name': 'category',
                'perceived_colour_value_id': 'int8',
                'perceived_colour_value_name': 'category',
                'perceived_colour_master_id': 'int8', 
                'perceived_colour_master_name': 'category',
                'department_no': 'int16',
                'department_name': 'category',
                'index_code': 'category',
                'index_name': 'category',
                'index_group_no': 'int8',
                'index_group_name': 'category',
                'section_no': 'int16',
                'section_name': 'category',
                'garment_group_no': 'int16',
                'garment_group_name': 'category',
                'detail_desc': 'string'
            },
            'transactions': {
                't_dat': 'string',  # Will convert to datetime after loading
                'customer_id': 'string',
                'article_id': 'int32',
                'price': 'float32',
                'sales_channel_id': 'int8'
            }
        }
    
    def load_customers(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load customers data with memory optimization
        
        Args:
            sample_size: Number of rows to sample (None for full dataset)
            
        Returns:
            DataFrame with customers data
        """
        logger.info("Loading customers data...")
        
        try:
            file_path = self.data_paths['customers_file']
            
            # Load with optimized dtypes
            df = pd.read_csv(
                file_path,
                dtype=self.dtypes['customers'],
                nrows=sample_size
            )
            
            # Memory usage info
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            logger.info(f"Customers data loaded: {len(df):,} rows, {memory_mb:.1f} MB")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load customers data: {e}")
            raise
    
    def load_articles(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load articles data with memory optimization
        
        Args:
            sample_size: Number of rows to sample (None for full dataset)
            
        Returns:
            DataFrame with articles data
        """
        logger.info("Loading articles data...")
        
        try:
            file_path = self.data_paths['articles_file']
            
            # Load with optimized dtypes
            df = pd.read_csv(
                file_path,
                dtype=self.dtypes['articles'],
                nrows=sample_size
            )
            
            # Memory usage info
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            logger.info(f"Articles data loaded: {len(df):,} rows, {memory_mb:.1f} MB")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load articles data: {e}")
            raise
    
    def load_transactions(self, sample_size: Optional[int] = None, 
                         date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Load transactions data with memory optimization and optional date filtering
        
        Args:
            sample_size: Number of rows to sample (None for full dataset)
            date_range: Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with transactions data
        """
        logger.info("Loading transactions data...")
        
        try:
            file_path = self.data_paths['transactions_file']
            
            # Load with optimized dtypes
            df = pd.read_csv(
                file_path,
                dtype=self.dtypes['transactions'],
                nrows=sample_size
            )
            
            # Convert date column to datetime
            df['t_dat'] = pd.to_datetime(df['t_dat'])
            
            # Apply date filtering if specified
            if date_range:
                start_date, end_date = date_range
                mask = (df['t_dat'] >= start_date) & (df['t_dat'] <= end_date)
                df = df[mask].copy()
                logger.info(f"Date filtering applied: {start_date} to {end_date}")
            
            # Memory usage info
            memory_mb = df.memory_usage(deep=True).sum() / (1024**2)
            logger.info(f"Transactions data loaded: {len(df):,} rows, {memory_mb:.1f} MB")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load transactions data: {e}")
            raise
    
    def load_all(self, sample_sizes: Optional[Dict[str, int]] = None,
                 date_range: Optional[Tuple[str, str]] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load all datasets efficiently
        
        Args:
            sample_sizes: Dictionary with sample sizes for each dataset
            date_range: Date range for transactions filtering
            
        Returns:
            Tuple of (customers, articles, transactions) DataFrames
        """
        logger.info("Loading all datasets...")
        
        if sample_sizes is None:
            sample_sizes = {}
        
        # Load all datasets
        customers = self.load_customers(sample_sizes.get('customers'))
        articles = self.load_articles(sample_sizes.get('articles'))
        transactions = self.load_transactions(sample_sizes.get('transactions'), date_range)
        
        # Summary statistics
        total_memory = (
            customers.memory_usage(deep=True).sum() + 
            articles.memory_usage(deep=True).sum() + 
            transactions.memory_usage(deep=True).sum()
        ) / (1024**2)
        
        logger.info(f"All datasets loaded successfully. Total memory: {total_memory:.1f} MB")
        
        return customers, articles, transactions
    
    def get_sample_config(self) -> Dict[str, int]:
        """Get sample sizes from configuration"""
        return self.config.get('development', {})
    
    def validate_data_files(self) -> Dict[str, bool]:
        """
        Validate that all required data files exist
        
        Returns:
            Dictionary with validation results for each file
        """
        results = {}
        
        for key, file_path in self.data_paths.items():
            if key.endswith('_file'):
                exists = Path(file_path).exists()
                results[key] = exists
                if exists:
                    logger.info(f"✅ {key}: {file_path}")
                else:
                    logger.error(f"❌ {key}: {file_path} not found")
        
        all_valid = all(results.values())
        logger.info(f"Data validation {'passed' if all_valid else 'failed'}")
        
        return results 