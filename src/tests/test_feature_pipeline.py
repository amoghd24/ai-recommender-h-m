"""
Test feature pipeline module
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pipelines.feature_pipeline import FeaturePipeline
from data_pipelines.data_loader import DataLoader

def test_feature_pipeline():
    """Test feature pipeline functionality"""
    print("🧪 Testing feature pipeline...")
    
    # Load sample data
    loader = DataLoader()
    customers_df = loader.load_customers(sample_size=1000)
    articles_df = loader.load_articles(sample_size=1000)
    transactions_df = loader.load_transactions(sample_size=5000)
    print(f"✅ Loaded data: {len(customers_df)} customers, {len(articles_df)} articles, {len(transactions_df)} transactions")
    
    # Initialize feature pipeline
    pipeline = FeaturePipeline()
    
    # Test 1: Create comprehensive features
    customer_features, article_features = pipeline.create_training_features(
        customers_df, articles_df, transactions_df
    )
    
    print(f"✅ Created customer features: {customer_features.shape}")
    print(f"✅ Created article features: {article_features.shape}")
    
    # Test 2: Verify feature columns exist
    expected_customer_cols = ['customer_id', 'age', 'age_group', 'club_member_status', 
                             'days_since_last_purchase', 'favorite_department']
    missing_cols = [col for col in expected_customer_cols if col not in customer_features.columns]
    assert len(missing_cols) == 0, f"Missing customer columns: {missing_cols}"
    print("✅ All expected customer feature columns present")
    
    # Print a sample of available columns for debugging
    print(f"Available article columns: {list(article_features.columns[:10])}...")
    
    expected_article_cols = ['article_id', 'department_name', 'product_group_name', 
                            'is_menswear', 'total_purchases', 'popularity_score']
    missing_cols = [col for col in expected_article_cols if col not in article_features.columns]
    assert len(missing_cols) == 0, f"Missing article columns: {missing_cols}"
    print("✅ All expected article feature columns present")
    
    # Test 3: Get feature names
    feature_names = pipeline.get_feature_names()
    assert len(feature_names) == 6, f"Expected 6 feature categories, got {len(feature_names)}"
    print(f"✅ Feature categories: {list(feature_names.keys())}")
    
    print("\n🎉 All feature pipeline tests passed!")

if __name__ == "__main__":
    test_feature_pipeline() 