"""
Tests for Two-Tower Model - focusing on core functionality.
"""

import pytest
import torch
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.two_tower_model import (
    TwoTowerModel,
    create_two_tower_model,
    TwoTowerLoss
)


def test_model_creation():
    """Test that we can create the model with default settings."""
    model = create_two_tower_model()
    
    assert isinstance(model, TwoTowerModel)
    assert model.embedding_dim == 128
    assert model.similarity_metric == 'cosine'
    assert model.temperature == 1.0
    print("✅ Model creation test passed")


def test_similarity_computation():
    """Test similarity computation with dummy embeddings."""
    model = create_two_tower_model(embedding_dim=64)
    
    # Create dummy embeddings
    customer_embs = torch.randn(4, 64)
    article_embs = torch.randn(4, 64)
    
    # Test cosine similarity
    similarities = model.compute_similarity(customer_embs, article_embs)
    assert similarities.shape == (4, 4)
    assert torch.all(similarities >= -1.1) and torch.all(similarities <= 1.1)
    
    # Test dot product similarity
    model.similarity_metric = 'dot'
    similarities = model.compute_similarity(customer_embs, article_embs)
    assert similarities.shape == (4, 4)
    print("✅ Similarity computation test passed")


def test_loss_function():
    """Test the loss function works."""
    loss_fn = TwoTowerLoss()
    
    # Create dummy similarity matrix and labels
    similarities = torch.randn(4, 4)
    labels = torch.randint(0, 2, (4, 4))
    
    loss = loss_fn(similarities, labels)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.shape == ()  # Scalar
    assert loss.item() >= 0
    print("✅ Loss function test passed")


def test_temperature_scaling():
    """Test temperature scaling affects similarity scores."""
    model = create_two_tower_model(embedding_dim=32, temperature=2.0)
    
    customer_embs = torch.randn(2, 32)
    article_embs = torch.randn(2, 32)
    
    similarities_temp2 = model.compute_similarity(customer_embs, article_embs)
    
    # Change temperature and compare
    model.temperature = 1.0
    similarities_temp1 = model.compute_similarity(customer_embs, article_embs)
    
    # Temperature scaling should scale the similarities
    assert torch.allclose(similarities_temp2 * 2.0, similarities_temp1, atol=1e-6)
    print("✅ Temperature scaling test passed")


def test_invalid_similarity_metric():
    """Test that invalid similarity metric raises appropriate error."""
    model = create_two_tower_model(embedding_dim=16)
    model.similarity_metric = 'invalid'
    
    customer_embs = torch.randn(2, 16)
    article_embs = torch.randn(2, 16)
    
    with pytest.raises(ValueError, match="Unsupported similarity metric"):
        model.compute_similarity(customer_embs, article_embs)
    print("✅ Invalid similarity metric test passed")


def test_model_info():
    """Test that we can get model information."""
    model = create_two_tower_model(embedding_dim=64)
    info = model.get_model_info()
    
    assert isinstance(info, dict)
    assert 'embedding_dim' in info
    assert 'similarity_metric' in info
    assert 'temperature' in info
    assert 'total_params' in info
    
    assert info['embedding_dim'] == 64
    assert info['similarity_metric'] == 'cosine'
    print("✅ Model info test passed")


def test_custom_model_creation():
    """Test creating model with custom parameters."""
    model = create_two_tower_model(
        embedding_dim=256,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.2,
        similarity_metric='dot',
        temperature=0.5
    )
    
    assert model.embedding_dim == 256
    assert model.similarity_metric == 'dot'
    assert model.temperature == 0.5
    print("✅ Custom model creation test passed")


if __name__ == '__main__':
    # Run tests
    test_model_creation()
    test_similarity_computation()
    test_loss_function()
    test_temperature_scaling()
    test_invalid_similarity_metric()
    test_model_info()
    test_custom_model_creation()
    
    print("\n🎉 All tests passed! Two-Tower Model is working correctly.")
    print("📝 Step 4: Main Two-Tower Model - ✅ COMPLETE") 