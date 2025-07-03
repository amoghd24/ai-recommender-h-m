"""
Test interaction feature engineering
"""

import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from features.interaction_features import InteractionFeatureEngineer
from data_pipelines.data_loader import DataLoader

def test_interaction_features():
    """Test interaction feature extraction"""
    print("🧪 Testing interaction features...")
    
    # Load sample data
    loader = DataLoader()
    transactions_df = loader.load_transactions(sample_size=5000)
    articles_df = loader.load_articles(sample_size=1000)
    print(f"✅ Loaded {len(transactions_df)} transactions and {len(articles_df)} articles")
    
    # Initialize interaction feature engineer
    interaction_engineer = InteractionFeatureEngineer()
    
    # Test 1: Customer preferences
    customer_prefs = interaction_engineer.extract_customer_preferences(transactions_df, articles_df)
    print(f"✅ Extracted preferences for {len(customer_prefs)} customers")
    
    expected_pref_features = [
        'customer_id', 'favorite_department', 'department_diversity',
        'favorite_color', 'avg_purchase_price', 'menswear_ratio', 'prefers_black'
    ]
    
    for feature in expected_pref_features:
        assert feature in customer_prefs.columns, f"Missing preference feature: {feature}"
    print("✅ All customer preference features created")
    
    # Test 2: Article popularity
    article_popularity = interaction_engineer.extract_article_popularity(transactions_df)
    print(f"✅ Extracted popularity for {len(article_popularity)} articles")
    
    assert 'popularity_score' in article_popularity.columns, "Missing popularity score"
    assert 'is_bestseller' in article_popularity.columns, "Missing bestseller flag"
    assert (article_popularity['repurchase_rate'] >= 1).all(), "Invalid repurchase rates"
    print("✅ Article popularity features are valid")
    
    # Test 3: Customer-article affinity
    affinity_scores = interaction_engineer.create_customer_article_affinity(
        transactions_df, customer_prefs, articles_df
    )
    print(f"✅ Created {len(affinity_scores)} affinity scores")
    
    # Verify affinity DataFrame structure
    assert 'department_affinity' in affinity_scores.columns, "Missing department_affinity column"
    
    # Only check affinity values if there are any scores
    if len(affinity_scores) > 0:
        assert affinity_scores['department_affinity'].between(0, 1).all(), "Invalid affinity scores"
        print("✅ Affinity scores are within valid range")
    else:
        print("⚠️  No affinity scores created (likely due to sample size)")
    
    print("\n🎉 All interaction feature tests passed!")

if __name__ == "__main__":
    test_interaction_features() 