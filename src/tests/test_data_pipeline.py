"""
Concise test suite for data loading and quality assessment modules
"""

import pandas as pd
import numpy as np
from test_config import create_sample_data, get_test_config, assert_dataframe_not_empty
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from data_pipelines.data_loader import DataLoader
from data_pipelines.data_quality import DataQualityAssessor


class TestDataPipeline:
    """Combined test cases for data pipeline components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.config = get_test_config()
        self.loader = DataLoader(self.config["config_path"])
        self.quality_assessor = DataQualityAssessor()
        self.customers, self.articles, self.transactions = create_sample_data()
    
    def test_data_loader_basic_functionality(self):
        """Test DataLoader initialization and basic loading"""
        # Test initialization
        assert self.loader.config is not None
        assert 'data_paths' in self.loader.config
        
        # Test file validation
        validation_results = self.loader.validate_data_files()
        assert all(validation_results.values()), f"Missing files: {validation_results}"
        
        # Test data loading with samples
        customers_df = self.loader.load_customers(500)
        articles_df = self.loader.load_articles(300)
        transactions_df = self.loader.load_transactions(1000)
        
        # Verify loaded data
        for df, name in [(customers_df, "Customers"), (articles_df, "Articles"), (transactions_df, "Transactions")]:
            assert_dataframe_not_empty(df, name)
        
        # Test data types optimization
        assert customers_df['customer_id'].dtype == 'string'
        assert articles_df['article_id'].dtype == 'int32'
        assert pd.api.types.is_datetime64_any_dtype(transactions_df['t_dat'])
    
    def test_data_quality_assessment(self):
        """Test DataQualityAssessor functionality"""
        test_datasets = [
            (self.customers, "customers"),
            (self.articles, "articles"), 
            (self.transactions, "transactions")
        ]
        
        for df, name in test_datasets:
            if df is not None and len(df) > 0:
                # Test missing values analysis
                missing_analysis = self.quality_assessor.assess_missing_values(df, name)
                assert 'overall' in missing_analysis
                assert 'by_column' in missing_analysis
                assert 0 <= missing_analysis['overall']['missing_rate'] <= 1
                
                # Test comprehensive quality report
                report = self.quality_assessor.generate_quality_report(df, name)
                assert 'dataset_info' in report
                assert 'overall_quality' in report
                assert 0 <= report['overall_quality']['score'] <= 1
                assert report['overall_quality']['grade'] in ['A', 'B', 'C', 'D']


if __name__ == "__main__":
    print("🧪 Running concise data pipeline tests...")
    
    test_pipeline = TestDataPipeline()
    test_pipeline.setup_method()
    
    try:
        test_pipeline.test_data_loader_basic_functionality()
        print("✅ DataLoader functionality test passed")
        
        test_pipeline.test_data_quality_assessment()
        print("✅ Data quality assessment test passed")
        
        print("\n🎉 All data pipeline tests passed! Step 1 complete.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise 