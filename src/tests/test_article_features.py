"""
Test basic article feature engineering
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from features.article_features import ArticleFeatureEngineer
from data_pipelines.data_loader import DataLoader

def test_basic_article_features():
    """Test basic article feature extraction"""
    print("🧪 Testing basic article features...")
    
    # Load sample article data
    loader = DataLoader()
    articles_df = loader.load_articles(sample_size=500)
    print(f"✅ Loaded {len(articles_df)} sample articles")
    
    # Initialize feature engineer
    feature_engineer = ArticleFeatureEngineer()
    
    # Extract basic features
    features_df = feature_engineer.extract_basic_features(articles_df)
    print(f"✅ Extracted basic features for {len(features_df)} articles")
    
    # Test 1: Check all expected columns are created
    expected_features = [
        'department_idx', 'product_group_idx', 'product_type_idx', 'category_path',
        'is_black', 'is_white', 'is_neutral_color',
        'has_graphic', 'is_patterned',
        'is_menswear', 'is_kidswear', 'is_womenswear',
        'is_divided', 'section_idx',
        'is_upper_body', 'is_lower_body', 'is_full_body',
        'has_description', 'description_length'
    ]
    
    for feature in expected_features:
        assert feature in features_df.columns, f"Missing feature: {feature}"
    print("✅ All expected features created")
    
    # Test 2: Verify binary features are binary
    binary_features = [col for col in features_df.columns if col.startswith(('is_', 'has_'))]
    for col in binary_features:
        assert features_df[col].isin([0, 1]).all(), f"{col} contains non-binary values"
    print("✅ All binary features are valid")
    
    # Test 3: Test price features without transaction data
    price_features_df = feature_engineer.extract_price_features(features_df)
    assert 'price_range' in price_features_df.columns, "Price range feature missing"
    assert 'avg_selling_price' in price_features_df.columns, "Average price feature missing"
    print("✅ Price features created successfully")
    
    print("\n🎉 All basic article feature tests passed!")

if __name__ == "__main__":
    test_basic_article_features() 