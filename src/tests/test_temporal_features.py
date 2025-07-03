"""
Test temporal feature engineering
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from features.temporal_features import TemporalFeatureEngineer
from data_pipelines.data_loader import DataLoader

def test_temporal_features():
    """Test temporal feature extraction"""
    print("🧪 Testing temporal features...")
    
    # Load sample transaction data
    loader = DataLoader()
    transactions_df = loader.load_transactions(sample_size=5000)
    print(f"✅ Loaded {len(transactions_df)} sample transactions")
    
    # Initialize temporal feature engineer
    temporal_engineer = TemporalFeatureEngineer(reference_date='2020-09-22')
    
    # Test customer temporal features
    customer_temporal = temporal_engineer.extract_customer_temporal_features(transactions_df)
    print(f"✅ Extracted temporal features for {len(customer_temporal)} customers")
    
    # Test 1: Check customer temporal features
    expected_customer_features = [
        'customer_id', 'days_since_last_purchase', 'days_since_first_purchase',
        'customer_lifespan_days', 'total_transactions', 'avg_days_between_purchases',
        'favorite_month', 'weekend_purchase_ratio', 'purchases_last_30_days',
        'purchases_last_90_days', 'is_active_customer'
    ]
    
    for feature in expected_customer_features:
        assert feature in customer_temporal.columns, f"Missing customer feature: {feature}"
    print("✅ All customer temporal features created")
    
    # Test 2: Verify feature validity
    assert (customer_temporal['days_since_last_purchase'] >= 0).all(), "Invalid recency values"
    assert customer_temporal['weekend_purchase_ratio'].between(0, 1).all(), "Invalid weekend ratio"
    assert customer_temporal['favorite_month'].between(1, 12).all(), "Invalid favorite month"
    print("✅ Customer temporal features are valid")
    
    # Test 3: Check article temporal features
    article_temporal = temporal_engineer.extract_article_temporal_features(transactions_df)
    print(f"✅ Extracted temporal features for {len(article_temporal)} articles")
    
    assert 'is_trending' in article_temporal.columns, "Missing trending feature"
    assert 'is_new_product' in article_temporal.columns, "Missing new product feature"
    
    print("\n🎉 All temporal feature tests passed!")

if __name__ == "__main__":
    test_temporal_features() 