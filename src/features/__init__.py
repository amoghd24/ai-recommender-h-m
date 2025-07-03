"""
Feature engineering modules for the H&M recommender system.
"""

from .customer_features import CustomerFeatureEngineer
from .article_features import ArticleFeatureEngineer
from .temporal_features import TemporalFeatureEngineer
from .interaction_features import InteractionFeatureEngineer
from .feature_config import (
    get_customer_feature_config,
    get_article_feature_config,
    get_categorical_sizes,
    CUSTOMER_CATEGORICAL_FEATURES,
    CUSTOMER_NUMERICAL_FEATURES,
    ARTICLE_CATEGORICAL_FEATURES,
    ARTICLE_NUMERICAL_FEATURES
)

__all__ = [
    'CustomerFeatureEngineer',
    'ArticleFeatureEngineer',
    'TemporalFeatureEngineer',
    'InteractionFeatureEngineer',
    'get_customer_feature_config',
    'get_article_feature_config',
    'get_categorical_sizes',
    'CUSTOMER_CATEGORICAL_FEATURES',
    'CUSTOMER_NUMERICAL_FEATURES',
    'ARTICLE_CATEGORICAL_FEATURES',
    'ARTICLE_NUMERICAL_FEATURES'
] 