"""
Unit tests for the CatBoost ranking model
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from models.ranking_model import RankingModel


def create_sample_ranking_data():
    """Create sample ranking dataset for testing"""
    np.random.seed(42)
    
    # Create customer features
    customer_features = ['age', 'total_transactions', 'avg_purchase_price']
    # Create article features  
    article_features = ['popularity_score', 'avg_selling_price', 'sales_velocity']
    # Create categorical features
    categorical_features = ['department_name', 'favorite_department']
    
    # Generate sample data
    n_samples = 1000
    data = {}
    
    # Numerical features
    for feature in customer_features + article_features:
        data[feature] = np.random.uniform(0, 1, n_samples)
    
    # Categorical features
    departments = ['Men', 'Women', 'Kids', 'Sport', 'Accessories']
    data['department_name'] = np.random.choice(departments, n_samples)
    data['favorite_department'] = np.random.choice(departments, n_samples)
    
    # Create labels (purchase = 1, no purchase = 0)
    # Make it slightly predictable based on department match
    labels = []
    for i in range(n_samples):
        if data['department_name'][i] == data['favorite_department'][i]:
            # Higher probability of purchase if departments match
            label = np.random.choice([0, 1], p=[0.3, 0.7])
        else:
            # Lower probability if departments don't match
            label = np.random.choice([0, 1], p=[0.8, 0.2])
        labels.append(label)
    
    data['label'] = labels
    data['customer_id'] = [f"customer_{i}" for i in range(n_samples)]
    data['article_id'] = [f"article_{i}" for i in range(n_samples)]
    
    return pd.DataFrame(data)


def test_ranking_model_initialization():
    """Test that ranking model initializes correctly"""
    model = RankingModel(iterations=10, learning_rate=0.1, depth=3)
    
    assert model.model is not None
    assert model.categorical_features is None
    assert model.is_fitted == False


def test_ranking_model_training():
    """Test that ranking model can train on sample data"""
    # Create sample data
    ranking_df = create_sample_ranking_data()
    
    # Split data
    train_size = int(0.8 * len(ranking_df))
    train_df = ranking_df[:train_size]
    
    # Initialize and train model
    model = RankingModel(iterations=10, learning_rate=0.1, depth=3, random_seed=42)
    metrics = model.train(train_df)
    
    # Check training worked
    assert model.is_fitted == True
    assert 'train_auc' in metrics
    assert 'feature_importance' in metrics
    assert len(metrics['feature_importance']) > 0  # Should have feature importance


def test_ranking_model_prediction():
    """Test that ranking model can make predictions"""
    # Create and train model
    ranking_df = create_sample_ranking_data()
    train_size = int(0.8 * len(ranking_df))
    train_df = ranking_df[:train_size]
    test_df = ranking_df[train_size:]
    
    model = RankingModel(iterations=10, learning_rate=0.1, depth=3, random_seed=42)
    model.train(train_df)
    
    # Make predictions
    probabilities = model.predict_proba(test_df)
    
    # Check predictions
    assert len(probabilities) == len(test_df)
    assert all(0 <= p <= 1 for p in probabilities)  # Probabilities should be between 0 and 1


def test_ranking_candidates():
    """Test ranking candidates for a customer"""
    # Create and train model
    ranking_df = create_sample_ranking_data()
    train_size = int(0.8 * len(ranking_df))
    train_df = ranking_df[:train_size]
    
    model = RankingModel(iterations=10, learning_rate=0.1, depth=3, random_seed=42)
    model.train(train_df)
    
    # Create sample customer and candidate articles
    customer_features = train_df.iloc[0].drop(['label', 'customer_id', 'article_id'])
    
    # Create some candidate articles
    candidates_df = ranking_df[['article_id', 'popularity_score', 'avg_selling_price', 
                               'sales_velocity', 'department_name']].head(10)
    
    # Rank candidates
    ranked_candidates = model.rank_candidates(candidates_df, customer_features, top_k=5)
    
    # Check results
    assert len(ranked_candidates) == 5
    assert 'ranking_score' in ranked_candidates.columns
    assert ranked_candidates['ranking_score'].is_monotonic_decreasing  # Should be sorted descending


def test_feature_importance():
    """Test feature importance extraction"""
    # Create and train model
    ranking_df = create_sample_ranking_data()
    train_size = int(0.8 * len(ranking_df))
    train_df = ranking_df[:train_size]
    
    model = RankingModel(iterations=10, learning_rate=0.1, depth=3, random_seed=42)
    model.train(train_df)
    
    # Get feature importance
    importance_df = model.get_feature_importance(top_n=5)
    
    # Check results
    assert len(importance_df) <= 5
    assert 'feature' in importance_df.columns
    assert 'importance' in importance_df.columns
    assert importance_df['importance'].is_monotonic_decreasing  # Should be sorted by importance


def test_model_save_load():
    """Test model saving and loading"""
    import tempfile
    
    # Create and train model
    ranking_df = create_sample_ranking_data()
    train_size = int(0.8 * len(ranking_df))
    train_df = ranking_df[:train_size]
    test_df = ranking_df[train_size:]
    
    model = RankingModel(iterations=10, learning_rate=0.1, depth=3, random_seed=42)
    model.train(train_df)
    
    # Get predictions before saving
    original_predictions = model.predict_proba(test_df)
    
    # Save model to temporary file
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
        model.save(tmp_file.name)
        
        # Load model
        loaded_model = RankingModel.load(tmp_file.name)
        
        # Test loaded model
        loaded_predictions = loaded_model.predict_proba(test_df)
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
        assert loaded_model.is_fitted == True


if __name__ == "__main__":
    print("🧪 Running ranking model tests...")
    
    test_ranking_model_initialization()
    print("✅ Initialization test passed")
    
    test_ranking_model_training()
    print("✅ Training test passed")
    
    test_ranking_model_prediction()
    print("✅ Prediction test passed")
    
    test_ranking_candidates()
    print("✅ Candidate ranking test passed")
    
    test_feature_importance()
    print("✅ Feature importance test passed")
    
    test_model_save_load()
    print("✅ Save/load test passed")
    
    print("\n🎉 All ranking model tests passed!") 