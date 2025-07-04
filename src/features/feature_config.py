"""
Shared Feature Configuration for H&M Recommender System
Central configuration for all feature definitions
"""

from typing import Dict, List, Any

# Customer feature configurations
CUSTOMER_CATEGORICAL_FEATURES = {
    'age_group': {
        'type': 'categorical',
        'num_categories': 10,
        'source_column': 'age',
        'transform': 'age_binning'
    },
    'club_member_status': {
        'type': 'categorical', 
        'num_categories': 2,
        'source_column': 'club_member_status',
        'transform': 'binary'
    },
    'fashion_news_active': {
        'type': 'categorical',
        'num_categories': 2,
        'source_column': 'fashion_news_frequency',
        'transform': 'binary'
    },
    'customer_lifecycle_stage': {
        'type': 'categorical',
        'num_categories': 5,
        'source_column': 'computed',
        'transform': 'lifecycle'
    },
    'favorite_department': {
        'type': 'categorical',
        'num_categories': 20,
        'source_column': 'computed',
        'transform': 'mode_from_transactions'
    },
    'favorite_color': {
        'type': 'categorical',
        'num_categories': 50,
        'source_column': 'computed',
        'transform': 'mode_from_transactions'
    }
}

CUSTOMER_NUMERICAL_FEATURES = {
    'days_since_last_purchase': {
        'type': 'numerical',
        'normalize': True,
        'fillna': 365
    },
    'purchase_frequency': {
        'type': 'numerical',
        'normalize': True,
        'fillna': 0
    },
    'avg_purchase_price': {
        'type': 'numerical',
        'normalize': True,
        'fillna': 0.05
    },
    'total_transactions': {
        'type': 'numerical',
        'normalize': True,
        'fillna': 0
    },
    'month_sin': {
        'type': 'numerical',
        'normalize': False,
        'fillna': 0
    },
    'month_cos': {
        'type': 'numerical',
        'normalize': False,
        'fillna': 0
    }
}

# Article feature configurations
ARTICLE_CATEGORICAL_FEATURES = {
    'department_name': {
        'type': 'categorical',
        'num_categories': 20,
        'source_column': 'department_name'
    },
    'product_group_name': {
        'type': 'categorical',
        'num_categories': 50,
        'source_column': 'product_group_name'
    },
    'garment_group_name': {
        'type': 'categorical',
        'num_categories': 30,
        'source_column': 'garment_group_name'
    },
    'colour_group_name': {
        'type': 'categorical',
        'num_categories': 50,
        'source_column': 'colour_group_name'
    },
    'perceived_colour_value_name': {
        'type': 'categorical',
        'num_categories': 20,
        'source_column': 'perceived_colour_value_name'
    },
    'index_name': {
        'type': 'categorical',
        'num_categories': 10,
        'source_column': 'index_name'
    },
    'section_name': {
        'type': 'categorical',
        'num_categories': 50,
        'source_column': 'section_name'
    },
    'graphical_appearance_name': {
        'type': 'categorical',
        'num_categories': 30,
        'source_column': 'graphical_appearance_name'
    }
}

ARTICLE_NUMERICAL_FEATURES = {
    'popularity_score': {
        'type': 'numerical',
        'normalize': True,
        'fillna': 0
    },
    'avg_selling_price': {
        'type': 'numerical',
        'normalize': True,
        'fillna': 0.05
    },
    'sales_velocity': {
        'type': 'numerical',
        'normalize': True,
        'fillna': 0
    },
    'days_since_first_sale': {
        'type': 'numerical',
        'normalize': True,
        'fillna': 365
    }
}

def get_customer_feature_config() -> Dict[str, Dict[str, Any]]:
    """Get complete customer feature configuration"""
    return {
        **CUSTOMER_CATEGORICAL_FEATURES,
        **CUSTOMER_NUMERICAL_FEATURES
    }

def get_article_feature_config() -> Dict[str, Dict[str, Any]]:
    """Get complete article feature configuration"""
    return {
        **ARTICLE_CATEGORICAL_FEATURES,
        **ARTICLE_NUMERICAL_FEATURES
    }

def get_categorical_sizes() -> Dict[str, int]:
    """Get sizes of all categorical features for embedding layers"""
    sizes = {}
    
    # Customer categoricals
    for name, config in CUSTOMER_CATEGORICAL_FEATURES.items():
        sizes[f"customer_{name}"] = config['num_categories']
    
    # Article categoricals  
    for name, config in ARTICLE_CATEGORICAL_FEATURES.items():
        sizes[f"article_{name}"] = config['num_categories']
        
    return sizes 