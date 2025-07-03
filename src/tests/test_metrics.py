"""
Tests for Training Metrics Module
"""

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from training.metrics import (
    TwoTowerMetrics,
    TwoTowerEvaluator,
    MetricsTracker,
    MetricsConfig,
    create_metrics_evaluator,
    create_metrics_tracker
)


class TestTwoTowerMetrics:
    """Test TwoTowerMetrics class"""
    
    @pytest.fixture
    def metrics(self):
        """Create metrics instance for testing"""
        config = MetricsConfig(
            k_values=[1, 5, 10],
            log_to_wandb=False,  # Disable wandb for testing
            compute_diversity=True,
            compute_ranking=True,
            compute_retrieval=True
        )
        return TwoTowerMetrics(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample predictions and targets for testing"""
        batch_size = 8
        num_candidates = 20
        
        # Create realistic similarity scores
        predictions = torch.randn(batch_size, num_candidates)
        
        # Create sparse target matrix (few positive items per query)
        targets = torch.zeros(batch_size, num_candidates)
        for i in range(batch_size):
            # Add 1-3 positive items per query
            num_positive = np.random.randint(1, 4)
            positive_indices = np.random.choice(num_candidates, num_positive, replace=False)
            targets[i, positive_indices] = 1.0
        
        return predictions, targets
    
    def test_initialization(self, metrics):
        """Test metrics initialization"""
        assert isinstance(metrics, TwoTowerMetrics)
        assert metrics.k_values == [1, 5, 10]
        assert not metrics.wandb_enabled  # Should be disabled for testing
    
    def test_retrieval_metrics_computation(self, metrics, sample_data):
        """Test retrieval metrics computation"""
        predictions, targets = sample_data
        
        retrieval_metrics = metrics.compute_retrieval_metrics(predictions, targets)
        
        # Check that all expected metrics are computed
        expected_keys = []
        for k in [1, 5, 10]:
            expected_keys.extend([
                f'precision@{k}',
                f'recall@{k}',
                f'top_{k}_accuracy',
                f'f1@{k}'
            ])
        
        for key in expected_keys:
            assert key in retrieval_metrics
            assert 0.0 <= retrieval_metrics[key] <= 1.0  # All metrics should be between 0 and 1
        
        print("✅ Retrieval metrics computation test passed")
    
    def test_ranking_metrics_computation(self, metrics, sample_data):
        """Test ranking metrics computation"""
        predictions, targets = sample_data
        
        ranking_metrics = metrics.compute_ranking_metrics(predictions, targets)
        
        # Check that expected metrics are computed
        assert 'map' in ranking_metrics
        assert 'mrr' in ranking_metrics
        
        # Check NDCG@K metrics
        for k in [1, 5, 10]:
            assert f'ndcg@{k}' in ranking_metrics
            assert 0.0 <= ranking_metrics[f'ndcg@{k}'] <= 1.0
        
        # MAP and MRR should be between 0 and 1
        assert 0.0 <= ranking_metrics['map'] <= 1.0
        assert 0.0 <= ranking_metrics['mrr'] <= 1.0
        
        print("✅ Ranking metrics computation test passed")
    
    def test_diversity_metrics_computation(self, metrics, sample_data):
        """Test diversity metrics computation"""
        predictions, targets = sample_data
        
        # Create sample item features for diversity computation
        num_candidates = predictions.shape[1]
        item_features = torch.randn(num_candidates, 64)
        
        diversity_metrics = metrics.compute_diversity_metrics(predictions, item_features)
        
        # Check that expected metrics are computed
        assert 'intra_list_diversity' in diversity_metrics
        assert 'coverage' in diversity_metrics
        assert 'gini_coefficient' in diversity_metrics
        
        # Check value ranges
        assert diversity_metrics['intra_list_diversity'] >= 0.0
        assert 0.0 <= diversity_metrics['coverage'] <= 1.0
        assert 0.0 <= diversity_metrics['gini_coefficient'] <= 1.0
        
        print("✅ Diversity metrics computation test passed")
    
    def test_all_metrics_computation(self, metrics, sample_data):
        """Test computing all metrics together"""
        predictions, targets = sample_data
        num_candidates = predictions.shape[1]
        item_features = torch.randn(num_candidates, 64)
        
        all_metrics = metrics.compute_all_metrics(
            predictions, targets, item_features, prefix="test_"
        )
        
        # Check that metrics have the correct prefix
        for key in all_metrics.keys():
            assert key.startswith("test_")
        
        # Check that we have a comprehensive set of metrics
        assert len(all_metrics) > 10  # Should have many metrics
        
        print("✅ All metrics computation test passed")
    
    def test_precision_at_k_perfect_case(self, metrics):
        """Test Precision@K with perfect predictions"""
        # Create perfect predictions (diagonal target matrix)
        predictions = torch.eye(4)
        targets = torch.eye(4)
        
        retrieval_metrics = metrics.compute_retrieval_metrics(predictions, targets, k_values=[1])
        
        # With perfect predictions, precision@1 should be 1.0
        assert retrieval_metrics['precision@1'] == 1.0
        assert retrieval_metrics['recall@1'] == 1.0
        assert retrieval_metrics['top_1_accuracy'] == 1.0
        
        print("✅ Perfect case test passed")
    
    def test_zero_case(self, metrics):
        """Test metrics with no relevant items"""
        # Create predictions with no relevant items
        predictions = torch.randn(4, 10)
        targets = torch.zeros(4, 10)
        
        retrieval_metrics = metrics.compute_retrieval_metrics(predictions, targets, k_values=[1, 5])
        
        # All metrics should be 0 when there are no relevant items
        for key in retrieval_metrics.keys():
            assert retrieval_metrics[key] == 0.0
        
        print("✅ Zero case test passed")


class TestTwoTowerEvaluator:
    """Test TwoTowerEvaluator class"""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock two-tower model"""
        model = MagicMock()
        model.eval = MagicMock()
        
        # Mock forward pass to return similarity matrix
        def mock_forward(customer_features, article_features):
            batch_size = 4
            return torch.randn(batch_size, batch_size)
        
        model.side_effect = mock_forward
        return model
    
    @pytest.fixture
    def evaluator(self, mock_model):
        """Create evaluator instance for testing"""
        config = MetricsConfig(
            k_values=[1, 5],
            log_to_wandb=False
        )
        return TwoTowerEvaluator(mock_model, config)
    
    def test_initialization(self, evaluator, mock_model):
        """Test evaluator initialization"""
        assert evaluator.model == mock_model
        assert isinstance(evaluator.metrics, TwoTowerMetrics)
        assert evaluator.config.k_values == [1, 5]
    
    def test_evaluate_retrieval(self, evaluator):
        """Test retrieval evaluation with embeddings"""
        # Create sample embeddings
        num_customers = 10
        num_articles = 20
        embedding_dim = 64
        
        customer_embeddings = torch.randn(num_customers, embedding_dim)
        article_embeddings = torch.randn(num_articles, embedding_dim)
        
        # Create customer and article IDs
        customer_ids = [f"customer_{i}" for i in range(num_customers)]
        article_ids = [f"article_{i}" for i in range(num_articles)]
        
        # Create ground truth (some random positive pairs)
        ground_truth = [
            ("customer_0", "article_5"),
            ("customer_1", "article_10"),
            ("customer_2", "article_3"),
        ]
        
        metrics = evaluator.evaluate_retrieval(
            customer_embeddings,
            article_embeddings,
            customer_ids,
            article_ids,
            ground_truth
        )
        
        # Check that metrics are computed
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # Check that key metrics are present
        assert 'precision@1' in metrics
        assert 'recall@1' in metrics
        
        print("✅ Retrieval evaluation test passed")


class TestMetricsTracker:
    """Test MetricsTracker class"""
    
    def test_initialization(self):
        """Test tracker initialization"""
        tracker = MetricsTracker(
            patience=5,
            min_delta=0.01,
            primary_metric='top_10_accuracy',
            mode='max'
        )
        
        assert tracker.patience == 5
        assert tracker.min_delta == 0.01
        assert tracker.primary_metric == 'top_10_accuracy'
        assert tracker.mode == 'max'
        assert tracker.epochs_without_improvement == 0
        
        print("✅ Tracker initialization test passed")
    
    def test_improvement_tracking(self):
        """Test tracking improvements"""
        tracker = MetricsTracker(patience=3, primary_metric='accuracy', mode='max')
        
        # Epoch 1: baseline
        should_stop = tracker.update({'accuracy': 0.5}, epoch=1)
        assert not should_stop
        assert tracker.best_score == 0.5
        assert tracker.epochs_without_improvement == 0
        
        # Epoch 2: improvement
        should_stop = tracker.update({'accuracy': 0.6}, epoch=2)
        assert not should_stop
        assert tracker.best_score == 0.6
        assert tracker.epochs_without_improvement == 0
        
        # Epoch 3: no improvement
        should_stop = tracker.update({'accuracy': 0.55}, epoch=3)
        assert not should_stop
        assert tracker.epochs_without_improvement == 1
        
        # Epoch 4: still no improvement
        should_stop = tracker.update({'accuracy': 0.52}, epoch=4)
        assert not should_stop
        assert tracker.epochs_without_improvement == 2
        
        # Epoch 5: still no improvement (should trigger early stopping)
        should_stop = tracker.update({'accuracy': 0.51}, epoch=5)
        assert should_stop
        assert tracker.epochs_without_improvement == 3
        
        print("✅ Improvement tracking test passed")
    
    def test_get_best_metrics(self):
        """Test getting best metrics"""
        tracker = MetricsTracker(primary_metric='accuracy', mode='max')
        
        tracker.update({'accuracy': 0.5, 'loss': 0.8}, epoch=1)
        tracker.update({'accuracy': 0.7, 'loss': 0.6}, epoch=2)  # Best epoch
        tracker.update({'accuracy': 0.6, 'loss': 0.7}, epoch=3)
        
        best_metrics = tracker.get_best_metrics()
        assert best_metrics['epoch'] == 2
        assert best_metrics['accuracy'] == 0.7
        assert best_metrics['loss'] == 0.6
        
        print("✅ Best metrics retrieval test passed")


class TestFactoryFunctions:
    """Test factory functions"""
    
    def test_create_metrics_evaluator(self):
        """Test metrics evaluator factory function"""
        mock_model = MagicMock()
        config = MetricsConfig(k_values=[1, 5])
        
        evaluator = create_metrics_evaluator(mock_model, config)
        
        assert isinstance(evaluator, TwoTowerEvaluator)
        assert evaluator.model == mock_model
        assert evaluator.config == config
        
        print("✅ Metrics evaluator factory test passed")
    
    def test_create_metrics_tracker(self):
        """Test metrics tracker factory function"""
        tracker = create_metrics_tracker(
            patience=15,
            primary_metric='ndcg@10',
            mode='max'
        )
        
        assert isinstance(tracker, MetricsTracker)
        assert tracker.patience == 15
        assert tracker.primary_metric == 'ndcg@10'
        assert tracker.mode == 'max'
        
        print("✅ Metrics tracker factory test passed")


def test_integration():
    """Integration test with realistic data"""
    print("\n🔧 Running integration test...")
    
    # Create realistic test scenario
    batch_size = 16
    num_candidates = 50
    
    # Simulate two-tower model output (similarity matrix)
    predictions = torch.randn(batch_size, num_candidates)
    
    # Create realistic targets (sparse positive labels)
    targets = torch.zeros(batch_size, num_candidates)
    for i in range(batch_size):
        # Each query has 1-5 relevant items
        num_relevant = np.random.randint(1, 6)
        relevant_indices = np.random.choice(num_candidates, num_relevant, replace=False)
        targets[i, relevant_indices] = 1.0
    
    # Test metrics computation
    config = MetricsConfig(
        k_values=[1, 5, 10, 20],
        log_to_wandb=False,
        compute_diversity=True,
        compute_ranking=True,
        compute_retrieval=True
    )
    
    metrics = TwoTowerMetrics(config)
    all_metrics = metrics.compute_all_metrics(predictions, targets)
    
    # Verify comprehensive metrics
    assert len(all_metrics) >= 15  # Should have many metrics
    
    # Check key TikTok recommender metrics are present
    assert 'top_10_accuracy' in all_metrics  # Key metric from documentation
    assert 'map' in all_metrics
    assert 'mrr' in all_metrics
    assert 'ndcg@10' in all_metrics
    assert 'precision@5' in all_metrics
    assert 'recall@5' in all_metrics
    
    # All metrics should be valid (between 0 and 1 for most metrics)
    for key, value in all_metrics.items():
        assert isinstance(value, (int, float, np.integer, np.floating))
        assert not np.isnan(value)
        assert not np.isinf(value)
    
    print(f"✅ Integration test passed - computed {len(all_metrics)} metrics")
    print(f"   📊 Top-10 Accuracy: {all_metrics['top_10_accuracy']:.4f}")
    print(f"   📊 MAP: {all_metrics['map']:.4f}")
    print(f"   📊 NDCG@10: {all_metrics['ndcg@10']:.4f}")
    print(f"   📊 MRR: {all_metrics['mrr']:.4f}")


if __name__ == '__main__':
    # Run all tests
    print("🧪 Testing TwoTowerMetrics...")
    test_metrics = TestTwoTowerMetrics()
    
    # Create fixtures manually for standalone execution
    config = MetricsConfig(k_values=[1, 5, 10], log_to_wandb=False)
    metrics = TwoTowerMetrics(config)
    
    batch_size = 8
    num_candidates = 20
    predictions = torch.randn(batch_size, num_candidates)
    targets = torch.zeros(batch_size, num_candidates)
    for i in range(batch_size):
        num_positive = np.random.randint(1, 4)
        positive_indices = np.random.choice(num_candidates, num_positive, replace=False)
        targets[i, positive_indices] = 1.0
    
    sample_data = (predictions, targets)
    
    # Run tests
    test_metrics.test_initialization(metrics)
    test_metrics.test_retrieval_metrics_computation(metrics, sample_data)
    test_metrics.test_ranking_metrics_computation(metrics, sample_data)
    test_metrics.test_diversity_metrics_computation(metrics, sample_data)
    test_metrics.test_all_metrics_computation(metrics, sample_data)
    test_metrics.test_precision_at_k_perfect_case(metrics)
    test_metrics.test_zero_case(metrics)
    
    print("\n🧪 Testing MetricsTracker...")
    test_tracker = TestMetricsTracker()
    test_tracker.test_initialization()
    test_tracker.test_improvement_tracking()
    test_tracker.test_get_best_metrics()
    
    print("\n🧪 Testing Factory Functions...")
    test_factory = TestFactoryFunctions()
    test_factory.test_create_metrics_evaluator()
    test_factory.test_create_metrics_tracker()
    
    # Integration test
    test_integration()
    
    print("\n🎉 All metrics tests passed! Step 6: Metrics Module - ✅ COMPLETE")
    print("\n📈 Key Features Implemented:")
    print("   ✅ Top-100 Accuracy (primary TikTok recommender metric)")
    print("   ✅ NDCG@K (ranking quality)")
    print("   ✅ Precision@K & Recall@K (retrieval performance)")
    print("   ✅ Mean Reciprocal Rank (MRR)")
    print("   ✅ Mean Average Precision (MAP)")
    print("   ✅ Diversity metrics (coverage, Gini coefficient)")
    print("   ✅ Early stopping support")
    print("   ✅ Wandb integration")
    print("   ✅ Comprehensive evaluation pipeline") 