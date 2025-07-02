"""
Demo: Step 1 - Data Pipeline (Loading & Quality Assessment)
Demonstrates the data loading and quality assessment capabilities built in Phase 1
"""

import sys
from pathlib import Path
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from data_pipelines.data_loader import DataLoader
from data_pipelines.data_quality import DataQualityAssessor

def main():
    print("🚀 H&M Recommender System - Step 1 Demo")
    print("=" * 60)
    print("📊 Data Loading & Quality Assessment Pipeline")
    print()
    
    # 1. Initialize components
    print("🔧 Initializing data pipeline components...")
    loader = DataLoader()
    quality_assessor = DataQualityAssessor()
    print("✅ Components initialized")
    print()
    
    # 2. Load sample datasets
    print("📥 Loading sample datasets...")
    sample_sizes = {'customers': 2000, 'articles': 1000, 'transactions': 5000}
    customers, articles, transactions = loader.load_all(sample_sizes)
    print()
    
    # 3. Generate quality reports for each dataset
    datasets = [
        (customers, "Customers"),
        (articles, "Articles"), 
        (transactions, "Transactions")
    ]
    
    print("🔍 Generating Data Quality Reports...")
    print("=" * 40)
    
    overall_scores = {}
    for df, name in datasets:
        print(f"\n📋 {name} Dataset Analysis:")
        print("-" * 30)
        
        # Generate comprehensive quality report
        report = quality_assessor.generate_quality_report(df, name.lower())
        
        # Extract key metrics
        dataset_info = report['dataset_info']
        missing_info = report['missing_values']['overall']
        overall_quality = report['overall_quality']
        
        print(f"   📏 Shape: {dataset_info['shape']}")
        print(f"   💾 Memory: {dataset_info['memory_usage_mb']:.1f} MB")
        print(f"   ❓ Missing Rate: {missing_info['missing_rate']:.2%}")
        print(f"   🎯 Quality Score: {overall_quality['score']:.3f}")
        print(f"   📈 Quality Grade: {overall_quality['grade']}")
        
        # Store overall scores
        overall_scores[name] = overall_quality['score']
        
        # Show recommendations if any
        if overall_quality['recommendations']:
            print(f"   💡 Recommendations:")
            for rec in overall_quality['recommendations'][:2]:  # Show top 2
                print(f"      • {rec}")
    
    # 4. Summary
    print(f"\n📊 PIPELINE SUMMARY")
    print("=" * 40)
    avg_quality = sum(overall_scores.values()) / len(overall_scores)
    print(f"   🎯 Average Quality Score: {avg_quality:.3f}")
    print(f"   📈 Overall Grade: {'A' if avg_quality >= 0.9 else 'B' if avg_quality >= 0.7 else 'C'}")
    
    print(f"\n   📋 Individual Scores:")
    for dataset, score in overall_scores.items():
        print(f"      {dataset}: {score:.3f}")
    
    # 5. Data types optimization showcase
    print(f"\n🛠️  MEMORY OPTIMIZATION SHOWCASE")
    print("=" * 40)
    print("   ✅ Categorical columns optimized with 'category' dtype")
    print("   ✅ Numerical columns optimized (float32, int32, int16, int8)")
    print("   ✅ Date columns properly converted to datetime")
    print("   ✅ String columns optimized with 'string' dtype")
    
    total_memory = sum(df.memory_usage(deep=True).sum() for df, _ in datasets) / (1024**2)
    print(f"   💾 Total optimized memory usage: {total_memory:.1f} MB")
    
    print(f"\n🎉 Step 1 Complete: Data Pipeline Successfully Built!")
    print("   • Efficient data loading with memory optimization")
    print("   • Comprehensive data quality assessment")  
    print("   • Automated missing value and outlier analysis")
    print("   • Data consistency validation")
    print("   • Quality scoring and recommendation system")

if __name__ == "__main__":
    main() 