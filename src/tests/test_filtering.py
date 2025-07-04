"""
Tests for Stage 2: Filtering Module
"""

import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from inference.filtering import FilteringStage


def test_filtering_stage():
    """Test basic filtering functionality"""
    
    # Initialize filtering stage
    filter_stage = FilteringStage(bloom_capacity=1000, bloom_error_rate=0.01)
    
    # Create sample data
    candidates = pd.DataFrame({
        'article_id': ['A001', 'A002', 'A003', 'A004', 'A005'],
        'price': [10.0, 20.0, 30.0, 40.0, 50.0],
        'department_name': ['Men', 'Women', 'Kids', 'Men', 'Women'],
        'is_kidswear': [0, 0, 1, 0, 0]
    })
    
    # Test 1: No filtering (new customer)
    filtered = filter_stage.filter_candidates(candidates, 'customer_123')
    assert len(filtered) == 5
    print("✅ Test 1: No filtering for new customer passed")
    
    # Test 2: Filter seen items
    filter_stage.update_user_history('customer_123', ['A001', 'A003'])
    filtered = filter_stage.filter_candidates(candidates, 'customer_123')
    assert len(filtered) == 3
    assert 'A001' not in filtered['article_id'].values
    assert 'A003' not in filtered['article_id'].values
    print("✅ Test 2: Seen items filtered correctly")
    
    # Test 3: Business rules filtering
    business_rules = {
        'price_range': (15.0, 35.0),
        'allowed_departments': ['Men', 'Women']
    }
    filtered = filter_stage.filter_candidates(candidates, 'customer_123', business_rules=business_rules)
    assert len(filtered) == 1  # Only A002 matches all criteria
    assert filtered.iloc[0]['article_id'] == 'A002'
    print("✅ Test 3: Business rules applied correctly")
    
    # Test 4: Stock filtering
    stock_data = pd.DataFrame({
        'article_id': ['A002', 'A004', 'A005'],
        'in_stock': [True, True, False]
    })
    filtered = filter_stage.filter_candidates(
        candidates, 
        'customer_456',  # Different customer
        stock_data=stock_data
    )
    assert len(filtered) == 2  # Only A002 and A004 are in stock
    print("✅ Test 4: Stock filtering works correctly")
    
    # Test 5: Get filter stats
    stats = filter_stage.get_filter_stats('customer_123')
    assert stats['has_history'] == True
    print("✅ Test 5: Filter stats retrieved successfully")
    

def test_batch_update():
    """Test batch updating of user histories"""
    
    filter_stage = FilteringStage()
    
    # Sample transaction data
    transactions = pd.DataFrame({
        'customer_id': ['C001', 'C001', 'C002', 'C002', 'C003'],
        'article_id': ['A001', 'A002', 'A001', 'A003', 'A004']
    })
    
    # Batch update
    filter_stage.batch_update_history(transactions)
    
    # Verify histories
    assert 'C001' in filter_stage.user_bloom_filters
    assert 'C002' in filter_stage.user_bloom_filters
    assert 'C003' in filter_stage.user_bloom_filters
    
    print("✅ Batch update test passed")


if __name__ == '__main__':
    test_filtering_stage()
    test_batch_update()
    print("\n🎉 All filtering tests passed!") 