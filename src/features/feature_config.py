"""Feature configuration wrapper.
The code below simply re-exports the
loader utilities and dictionaries so that existing imports continue to work
unchanged.
"""

from utils.config_loader import (  # type: ignore
    load_feature_config,
    get_customer_feature_config,
    get_article_feature_config,
    get_categorical_sizes,
    CUSTOMER_CATEGORICAL_FEATURES,
    CUSTOMER_NUMERICAL_FEATURES,
    ARTICLE_CATEGORICAL_FEATURES,
    ARTICLE_NUMERICAL_FEATURES,
)

__all__ = [
    "load_feature_config",
    "get_customer_feature_config",
    "get_article_feature_config",
    "get_categorical_sizes",
    "CUSTOMER_CATEGORICAL_FEATURES",
    "CUSTOMER_NUMERICAL_FEATURES",
    "ARTICLE_CATEGORICAL_FEATURES",
    "ARTICLE_NUMERICAL_FEATURES",
] 