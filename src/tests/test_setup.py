"""
Test to verify the test setup and data availability
"""

import pytest
import pandas as pd
from test_config import create_sample_data, get_test_config, assert_dataframe_not_empty

class TestSetup:
    
    def test_data_availability(self):
        """Test that we can access the raw data files"""
        config = get_test_config()
        
        # Check if data files exist
        import os
        assert os.path.exists(config["data_paths"]["customers"]), "Customers data file should exist"
        assert os.path.exists(config["data_paths"]["articles"]), "Articles data file should exist" 
        assert os.path.exists(config["data_paths"]["transactions"]), "Transactions data file should exist"
    
    def test_sample_data_creation(self):
        """Test that we can create sample data for testing"""
        customers, articles, transactions = create_sample_data()
        
        # Verify sample data is loaded
        assert_dataframe_not_empty(customers, "Customers sample")
        assert_dataframe_not_empty(articles, "Articles sample")
        assert_dataframe_not_empty(transactions, "Transactions sample")
        
        # Check sample sizes
        config = get_test_config()
        assert len(customers) <= config["sample_sizes"]["customers"]
        assert len(articles) <= config["sample_sizes"]["articles"] 
        assert len(transactions) <= config["sample_sizes"]["transactions"]
    
    def test_config_loading(self):
        """Test that configuration can be loaded"""
        config = get_test_config()
        
        assert "data_paths" in config
        assert "sample_sizes" in config
        assert "config_path" in config
        
        # Check config file exists
        import os
        assert os.path.exists(config["config_path"]), "Configuration file should exist"

if __name__ == "__main__":
    # Run basic tests
    print("🧪 Running setup tests...")
    
    test_setup = TestSetup()
    
    try:
        test_setup.test_data_availability()
        print("✅ Data availability test passed")
        
        test_setup.test_sample_data_creation()  
        print("✅ Sample data creation test passed")
        
        test_setup.test_config_loading()
        print("✅ Configuration loading test passed")
        
        print("\n🎉 All setup tests passed! Ready to proceed with development.")
        
    except Exception as e:
        print(f"❌ Setup test failed: {e}")
        raise 