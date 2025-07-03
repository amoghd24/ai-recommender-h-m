"""
Tensor Converter Utility for H&M Two-Tower Recommender System
Converts pandas DataFrames to PyTorch tensors for neural network input
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureToTensorConverter:
    """
    Converts engineered features from pandas DataFrame to PyTorch tensors
    Maintains separation between data engineering and deep learning layers
    """
    
    def __init__(self, feature_config: Dict[str, Dict]):
        """
        Initialize converter with feature configuration
        
        Args:
            feature_config: Configuration for each feature
        """
        self.feature_config = feature_config
        self.categorical_features = []
        self.numerical_features = []
        
        # Separate features by type
        for feature_name, config in feature_config.items():
            if config.get('type') == 'categorical':
                self.categorical_features.append(feature_name)
            elif config.get('type') == 'numerical':
                self.numerical_features.append(feature_name)
    
    def convert_to_tensors(
        self, 
        features_df: pd.DataFrame,
        normalize_numerical: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Convert DataFrame features to PyTorch tensors
        
        Args:
            features_df: DataFrame with engineered features
            normalize_numerical: Whether to normalize numerical features
            
        Returns:
            Tuple of (categorical_tensors, numerical_tensor)
        """
        categorical_tensors = {}
        numerical_list = []
        
        # Convert categorical features
        for feature_name in self.categorical_features:
            if feature_name in features_df.columns:
                # Convert to tensor, handling categorical dtype
                values = features_df[feature_name]
                if hasattr(values, 'cat'):
                    # If it's a categorical column, use the codes
                    tensor_values = values.cat.codes.values
                else:
                    # Otherwise, assume it's already numeric
                    tensor_values = values.values
                
                # Convert to LongTensor for embedding layers
                categorical_tensors[feature_name] = torch.LongTensor(tensor_values)
        
        # Convert numerical features
        for feature_name in self.numerical_features:
            if feature_name in features_df.columns:
                values = features_df[feature_name].values
                
                if normalize_numerical:
                    config = self.feature_config.get(feature_name, {})
                    mean = config.get('mean', np.mean(values))
                    std = config.get('std', np.std(values))
                    if std > 0:
                        values = (values - mean) / std
                
                numerical_list.append(values)
        
        # Stack numerical features
        if numerical_list:
            numerical_tensor = torch.FloatTensor(np.column_stack(numerical_list))
        else:
            numerical_tensor = None
        
        return categorical_tensors, numerical_tensor 
    
    def convert_series(
        self,
        feature_series: pd.Series,
        normalize_numerical: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Convert a single pandas Series with features to tensors
        
        Args:
            feature_series: Pandas Series with feature values
            normalize_numerical: Whether to normalize numerical features
            
        Returns:
            Dictionary with 'categorical' and 'numerical' tensors
        """
        categorical_values = []
        numerical_values = []
        
        # Process categorical features
        for feature_name in self.categorical_features:
            if feature_name in feature_series.index:
                value = feature_series[feature_name]
                
                # Handle categorical dtype
                if hasattr(value, 'cat'):
                    categorical_values.append(value.cat.codes)
                elif isinstance(value, str):
                    # Convert string categories to numeric codes (simple hash)
                    categorical_values.append(hash(value) % 10000)
                else:
                    categorical_values.append(int(value))
        
        # Process numerical features
        for feature_name in self.numerical_features:
            if feature_name in feature_series.index:
                value = float(feature_series[feature_name])
                
                if normalize_numerical:
                    config = self.feature_config.get(feature_name, {})
                    mean = config.get('mean', 0.0)
                    std = config.get('std', 1.0)
                    if std > 0:
                        value = (value - mean) / std
                
                numerical_values.append(value)
        
        # Convert to tensors
        categorical_tensor = torch.LongTensor(categorical_values) if categorical_values else torch.LongTensor([])
        numerical_tensor = torch.FloatTensor(numerical_values) if numerical_values else torch.FloatTensor([])
        
        return {
            'categorical': categorical_tensor,
            'numerical': numerical_tensor
        }
    
    def convert_batch(
        self,
        features_dict: Dict[str, Union[List, np.ndarray]],
        normalize_numerical: bool = True
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Convert a batch of features from dict format to tensors
        Useful for real-time inference
        
        Args:
            features_dict: Dictionary of feature arrays
            normalize_numerical: Whether to normalize numerical features
            
        Returns:
            Tuple of (categorical_tensors, numerical_tensor)
        """
        categorical_tensors = {}
        numerical_list = []
        
        # Convert categorical features
        for feature_name in self.categorical_features:
            if feature_name in features_dict:
                values = np.array(features_dict[feature_name])
                categorical_tensors[feature_name] = torch.LongTensor(values)
        
        # Convert numerical features  
        for feature_name in self.numerical_features:
            if feature_name in features_dict:
                values = np.array(features_dict[feature_name], dtype=np.float32)
                
                if normalize_numerical:
                    config = self.feature_config.get(feature_name, {})
                    mean = config.get('mean', 0.0)
                    std = config.get('std', 1.0) 
                    if std > 0:
                        values = (values - mean) / std
                
                numerical_list.append(values)
        
        # Stack numerical features
        if numerical_list:
            numerical_tensor = torch.FloatTensor(np.column_stack(numerical_list))
        else:
            numerical_tensor = None
        
        return categorical_tensors, numerical_tensor 