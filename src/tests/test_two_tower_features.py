"""
Test two-tower feature extraction for retrieval model
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from features.two_tower_features import TwoTowerFeatureExtractor
from data_pipelines.data_loader import DataLoader


def test_two_tower_features():
    """Test two-tower feature extraction following Lesson 2 specs"""
    print("🧪 Testing two-tower features...")
    
    # Load sample data
    loader = DataLoader()
    customers_df = loader.load_customers(sample_size=100)
    articles_df = loader.load_articles(sample_size=100)
    print(f"✅ Loaded {len(customers_df)} customers and {len(articles_df)} articles")
    
    # Initialize feature extractor
    extractor = TwoTowerFeatureExtractor(reference_date='2020-09-22')
    
    # Test customer features
    customer_features = extractor.extract_customer_features(customers_df)
    print(f"✅ Extracted customer features: {customer_features.shape}")
    
    # Test 1: Check customer feature columns
    expected_customer_features = ['customer_id', 'age', 'month_sin', 'month_cos']
    for feature in expected_customer_features:
        assert feature in customer_features.columns, f"Missing customer feature: {feature}"
    print("✅ All required customer features present")
    
    # Test 2: Validate customer feature values
    assert customer_features['customer_id'].notna().all(), "customer_id should not have NaN"
    assert customer_features['age'].between(0, 120).all(), "Invalid age values"
    assert customer_features['month_sin'].between(-1, 1).all(), "Invalid month_sin values"
    assert customer_features['month_cos'].between(-1, 1).all(), "Invalid month_cos values"
    
    # Test 3: Check cyclic encoding consistency
    assert customer_features['month_sin'].nunique() == 1, "month_sin should be consistent"
    assert customer_features['month_cos'].nunique() == 1, "month_cos should be consistent"
    print("✅ Customer feature values are valid")
    
    # Test article features
    article_features = extractor.extract_article_features(articles_df)
    print(f"✅ Extracted article features: {article_features.shape}")
    
    # Test 4: Check article feature columns
    expected_article_features = ['article_id', 'garment_group_name', 'index_group_name']
    for feature in expected_article_features:
        assert feature in article_features.columns, f"Missing article feature: {feature}"
    print("✅ All required article features present")
    
    # Test 5: Validate article feature values
    assert article_features['article_id'].notna().all(), "article_id should not have NaN"
    assert article_features['garment_group_name'].notna().all(), "garment_group_name should not have NaN"
    assert article_features['index_group_name'].notna().all(), "index_group_name should not have NaN"
    print("✅ Article feature values are valid")
    
    # Test 6: Feature minimalism check
    assert len(customer_features.columns) == 4, f"Customer features should be minimal (4), got {len(customer_features.columns)}"
    assert len(article_features.columns) == 3, f"Article features should be minimal (3), got {len(article_features.columns)}"
    print("✅ Feature set is minimal as per Lesson 2")
    
    # Test 7: Check feature name getters
    customer_names = extractor.get_customer_feature_names()
    article_names = extractor.get_article_feature_names()
    
    assert customer_names == expected_customer_features, "Customer feature names mismatch"
    assert article_names == expected_article_features, "Article feature names mismatch"
    print("✅ Feature name getters work correctly")
    
    # Test 8: Verify cyclic encoding values for September (month 9)
    expected_sin = np.sin(2 * np.pi * 9 / 12)  # September = -1.0
    expected_cos = np.cos(2 * np.pi * 9 / 12)  # September = 0.0
    
    actual_sin = customer_features['month_sin'].iloc[0]
    actual_cos = customer_features['month_cos'].iloc[0]
    
    assert abs(actual_sin - expected_sin) < 0.0001, f"month_sin mismatch: {actual_sin} vs {expected_sin}"
    assert abs(actual_cos - expected_cos) < 0.0001, f"month_cos mismatch: {actual_cos} vs {expected_cos}"
    print("✅ Cyclic encoding values are mathematically correct")
    
    print("\n🎉 All two-tower feature tests passed!")
    print(f"Customer features: {list(customer_features.columns)}")
    print(f"Article features: {list(article_features.columns)}")


if __name__ == "__main__":
    test_two_tower_features()