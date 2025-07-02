"""
Test configuration and utilities for H&M Recommender System
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

# Test data paths
TEST_DATA_DIR = PROJECT_ROOT / "data" / "raw"
CONFIG_PATH = PROJECT_ROOT / "config" / "data_config.yaml"

# Sample data sizes for testing (smaller subsets)
TEST_SAMPLE_SIZES = {
    "customers": 1000,
    "articles": 500, 
    "transactions": 5000
}

def create_sample_data():
    """Create small sample datasets for testing"""
    try:
        # Load small samples of actual data if available
        customers_sample = pd.read_csv(TEST_DATA_DIR / "customers.csv").head(TEST_SAMPLE_SIZES["customers"])
        articles_sample = pd.read_csv(TEST_DATA_DIR / "articles.csv").head(TEST_SAMPLE_SIZES["articles"])
        transactions_sample = pd.read_csv(TEST_DATA_DIR / "transactions_train.csv").head(TEST_SAMPLE_SIZES["transactions"])
        
        return customers_sample, articles_sample, transactions_sample
    except Exception as e:
        print(f"Warning: Could not load sample data: {e}")
        return None, None, None

def get_test_config():
    """Get test configuration parameters"""
    return {
        "data_paths": {
            "customers": str(TEST_DATA_DIR / "customers.csv"),
            "articles": str(TEST_DATA_DIR / "articles.csv"), 
            "transactions": str(TEST_DATA_DIR / "transactions_train.csv")
        },
        "sample_sizes": TEST_SAMPLE_SIZES,
        "config_path": str(CONFIG_PATH)
    }

# Common test assertions
def assert_dataframe_not_empty(df, name="DataFrame"):
    """Assert that dataframe is not empty"""
    assert df is not None, f"{name} should not be None"
    assert len(df) > 0, f"{name} should not be empty"
    assert df.shape[1] > 0, f"{name} should have columns"

def assert_no_null_in_required_columns(df, required_columns, name="DataFrame"):
    """Assert that required columns have no null values"""
    for col in required_columns:
        assert col in df.columns, f"{name} should have column '{col}'"
        null_count = df[col].isnull().sum()
        assert null_count == 0, f"{name} column '{col}' should not have null values, found {null_count}" 