"""
Test basic customer feature engineering
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from features.customer_features import CustomerFeatureEngineer
from data_pipelines.data_loader import DataLoader

def test_basic_customer_features():
    """Test basic customer feature extraction"""
    print("🧪 Testing basic customer features...")
    
    # Load sample customer data
    loader = DataLoader()
    customers_df = loader.load_customers(sample_size=1000)
    print(f"✅ Loaded {len(customers_df)} sample customers")
    
    # Initialize feature engineer
    feature_engineer = CustomerFeatureEngineer()
    
    # Extract features
    features_df = feature_engineer.extract_basic_features(customers_df)
    print(f"✅ Extracted features for {len(features_df)} customers")
    
    # Test 1: Check all expected columns are created
    expected_features = [
        'age_group', 'age_filled', 'is_age_missing',
        'is_club_member', 'club_member_status_filled',
        'receives_fashion_news', 'fashion_news_frequency_filled', 
        'fashion_news_score', 'is_active', 'has_FN', 'lifecycle_stage'
    ]
    
    for feature in expected_features:
        assert feature in features_df.columns, f"Missing feature: {feature}"
    print("✅ All expected features created")
    
    # Test 2: Verify no null values in filled columns
    filled_columns = ['age_filled', 'club_member_status_filled', 'fashion_news_frequency_filled']
    for col in filled_columns:
        assert features_df[col].isna().sum() == 0, f"Found nulls in {col}"
    print("✅ No nulls in filled columns")
    
    # Test 3: Verify lifecycle stages are valid
    valid_stages = ['inactive', 'loyal', 'engaged', 'casual', 'new']
    assert features_df['lifecycle_stage'].isin(valid_stages).all(), "Invalid lifecycle stages found"
    print("✅ All lifecycle stages are valid")
    
    print("\n🎉 All basic customer feature tests passed!")

if __name__ == "__main__":
    test_basic_customer_features() 