"""
Data Quality Assessment Module for H&M Recommender System
Provides comprehensive data quality analysis and reporting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityAssessor:
    """
    Comprehensive data quality assessment and reporting
    """
    
    def __init__(self, quality_thresholds: Optional[Dict] = None):
        """
        Initialize DataQualityAssessor
        
        Args:
            quality_thresholds: Dictionary with quality threshold configurations
        """
        if quality_thresholds is None:
            quality_thresholds = {
                'max_missing_rate': 0.5,
                'outlier_std_threshold': 3,
                'min_unique_values': 2,
                'max_cardinality_rate': 0.8
            }
        
        self.thresholds = quality_thresholds
        logger.info("DataQualityAssessor initialized")
    
    def assess_missing_values(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Comprehensive missing value analysis
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Dictionary with missing value analysis results
        """
        logger.info(f"Analyzing missing values for {dataset_name}")
        
        missing_info = {}
        missing_counts = df.isnull().sum()
        missing_rates = missing_counts / len(df)
        
        # Overall statistics
        total_missing = missing_counts.sum()
        total_values = df.size
        overall_missing_rate = total_missing / total_values
        
        missing_info['overall'] = {
            'total_missing': int(total_missing),
            'total_values': int(total_values),
            'missing_rate': float(overall_missing_rate),
            'quality_score': max(0, 1 - overall_missing_rate)
        }
        
        # Per-column analysis
        missing_info['by_column'] = {}
        for col in df.columns:
            missing_count = int(missing_counts[col])
            missing_rate = float(missing_rates[col])
            
            missing_info['by_column'][col] = {
                'missing_count': missing_count,
                'missing_rate': missing_rate,
                'quality_flag': 'good' if missing_rate <= self.thresholds['max_missing_rate'] else 'poor'
            }
        
        # Columns with high missing rates
        high_missing_cols = missing_rates[missing_rates > self.thresholds['max_missing_rate']].index.tolist()
        missing_info['high_missing_columns'] = high_missing_cols
        
        # Summary statistics
        missing_info['summary'] = {
            'columns_with_missing': int((missing_counts > 0).sum()),
            'columns_with_high_missing': len(high_missing_cols),
            'worst_missing_rate': float(missing_rates.max()),
            'avg_missing_rate': float(missing_rates.mean())
        }
        
        logger.info(f"Missing value analysis complete for {dataset_name}")
        return missing_info
    
    def detect_outliers(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Detect outliers in numerical columns using IQR and Z-score methods
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Dictionary with outlier detection results
        """
        logger.info(f"Detecting outliers for {dataset_name}")
        
        outlier_info = {}
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if df[col].notna().sum() < 10:  # Skip columns with very few values
                continue
                
            col_data = df[col].dropna()
            outlier_info[col] = {}
            
            # IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            iqr_outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            
            # Z-score method
            z_scores = np.abs((col_data - col_data.mean()) / col_data.std())
            zscore_outliers = col_data[z_scores > self.thresholds['outlier_std_threshold']]
            
            outlier_info[col] = {
                'iqr_outliers': {
                    'count': len(iqr_outliers),
                    'rate': len(iqr_outliers) / len(col_data),
                    'bounds': (float(lower_bound), float(upper_bound))
                },
                'zscore_outliers': {
                    'count': len(zscore_outliers),
                    'rate': len(zscore_outliers) / len(col_data),
                    'threshold': self.thresholds['outlier_std_threshold']
                },
                'stats': {
                    'mean': float(col_data.mean()),
                    'std': float(col_data.std()),
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'median': float(col_data.median())
                }
            }
        
        logger.info(f"Outlier detection complete for {dataset_name}")
        return outlier_info
    
    def analyze_distributions(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Analyze data distributions and characteristics
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Dictionary with distribution analysis results
        """
        logger.info(f"Analyzing distributions for {dataset_name}")
        
        distribution_info = {}
        
        # Numerical columns analysis
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        distribution_info['numerical'] = {}
        
        for col in numerical_cols:
            if df[col].notna().sum() < 2:
                continue
                
            col_data = df[col].dropna()
            
            distribution_info['numerical'][col] = {
                'count': len(col_data),
                'unique_values': int(col_data.nunique()),
                'mean': float(col_data.mean()),
                'std': float(col_data.std()),
                'min': float(col_data.min()),
                'max': float(col_data.max()),
                'median': float(col_data.median()),
                'skewness': float(col_data.skew()),
                'kurtosis': float(col_data.kurtosis()),
                'quartiles': {
                    'q25': float(col_data.quantile(0.25)),
                    'q75': float(col_data.quantile(0.75))
                }
            }
        
        # Categorical columns analysis
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        distribution_info['categorical'] = {}
        
        for col in categorical_cols:
            col_data = df[col].dropna()
            value_counts = col_data.value_counts()
            
            distribution_info['categorical'][col] = {
                'count': len(col_data),
                'unique_values': int(col_data.nunique()),
                'cardinality_rate': col_data.nunique() / len(col_data) if len(col_data) > 0 else 0,
                'most_frequent': str(value_counts.iloc[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'least_frequent': str(value_counts.iloc[-1]) if len(value_counts) > 0 else None,
                'least_frequent_count': int(value_counts.iloc[-1]) if len(value_counts) > 0 else 0,
                'top_5_values': value_counts.head(5).to_dict()
            }
        
        logger.info(f"Distribution analysis complete for {dataset_name}")
        return distribution_info
    
    def assess_data_consistency(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Assess data consistency and identify potential issues
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Dictionary with consistency analysis results
        """
        logger.info(f"Assessing data consistency for {dataset_name}")
        
        consistency_info = {}
        
        # Duplicate rows analysis
        duplicate_count = df.duplicated().sum()
        consistency_info['duplicates'] = {
            'count': int(duplicate_count),
            'rate': float(duplicate_count / len(df)) if len(df) > 0 else 0
        }
        
        # Data type consistency
        consistency_info['data_types'] = {}
        for col in df.columns:
            dtype_info = {
                'current_dtype': str(df[col].dtype),
                'null_count': int(df[col].isnull().sum()),
                'non_null_count': int(df[col].notna().sum())
            }
            
            # Check for mixed types in object columns
            if df[col].dtype == 'object':
                sample_values = df[col].dropna().head(100)
                unique_types = set(type(val).__name__ for val in sample_values)
                dtype_info['unique_types_in_sample'] = list(unique_types)
                dtype_info['type_consistency'] = len(unique_types) <= 1
            
            consistency_info['data_types'][col] = dtype_info
        
        # Value range consistency (for numerical columns)
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        consistency_info['value_ranges'] = {}
        
        for col in numerical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                consistency_info['value_ranges'][col] = {
                    'has_negative': bool((col_data < 0).any()),
                    'has_zero': bool((col_data == 0).any()),
                    'has_positive': bool((col_data > 0).any()),
                    'range_span': float(col_data.max() - col_data.min()),
                    'coefficient_of_variation': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else np.inf
                }
        
        logger.info(f"Consistency analysis complete for {dataset_name}")
        return consistency_info
    
    def generate_quality_report(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
        """
        Generate comprehensive data quality report
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset for reporting
            
        Returns:
            Complete data quality report dictionary
        """
        logger.info(f"Generating comprehensive quality report for {dataset_name}")
        
        report = {
            'dataset_info': {
                'name': dataset_name,
                'shape': df.shape,
                'memory_usage_mb': float(df.memory_usage(deep=True).sum() / (1024**2)),
                'analysis_timestamp': datetime.now().isoformat()
            },
            'missing_values': self.assess_missing_values(df, dataset_name),
            'outliers': self.detect_outliers(df, dataset_name),
            'distributions': self.analyze_distributions(df, dataset_name),
            'consistency': self.assess_data_consistency(df, dataset_name)
        }
        
        # Calculate overall quality score
        missing_score = report['missing_values']['overall']['quality_score']
        consistency_score = 1 - report['consistency']['duplicates']['rate']
        
        # Simple quality scoring (can be enhanced)
        overall_score = (missing_score + consistency_score) / 2
        
        report['overall_quality'] = {
            'score': float(overall_score),
            'grade': 'A' if overall_score >= 0.9 else 'B' if overall_score >= 0.7 else 'C' if overall_score >= 0.5 else 'D',
            'recommendations': self._generate_recommendations(report)
        }
        
        logger.info(f"Quality report generated for {dataset_name}. Overall score: {overall_score:.3f}")
        return report
    
    def _generate_recommendations(self, report: Dict) -> List[str]:
        """Generate actionable recommendations based on quality analysis"""
        recommendations = []
        
        # Missing values recommendations
        high_missing_cols = report['missing_values']['high_missing_columns']
        if high_missing_cols:
            recommendations.append(f"Consider handling missing values in columns: {', '.join(high_missing_cols)}")
        
        # Duplicates recommendations
        if report['consistency']['duplicates']['rate'] > 0.01:  # More than 1% duplicates
            recommendations.append("Consider removing or investigating duplicate rows")
        
        # Data type recommendations
        inconsistent_types = [
            col for col, info in report['consistency']['data_types'].items()
            if info.get('type_consistency') is False
        ]
        if inconsistent_types:
            recommendations.append(f"Review data type consistency in columns: {', '.join(inconsistent_types)}")
        
        # Memory optimization recommendations
        if report['dataset_info']['memory_usage_mb'] > 100:
            recommendations.append("Consider data type optimization to reduce memory usage")
        
        return recommendations 