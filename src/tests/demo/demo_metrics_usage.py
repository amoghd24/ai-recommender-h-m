"""
Demo: Using the Metrics Module with Two-Tower Model

This demo shows how to use the metrics module to evaluate
a Two-Tower model with all the TikTok-like recommender metrics.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import torch
import numpy as np
from models.two_tower_model import create_two_tower_model
from training.metrics import (
    MetricsConfig,
    TwoTowerEvaluator,
    MetricsTracker,
    create_metrics_evaluator
)


def demo_metrics_evaluation():
    """Demonstrate comprehensive metrics evaluation"""
    print("🚀 H&M Two-Tower Model Metrics Evaluation Demo")
    print("=" * 55)
    
    # 1. Create a Two-Tower model
    print("\n📝 Step 1: Creating Two-Tower Model...")
    model = create_two_tower_model(
        embedding_dim=64,
        hidden_dims=[128, 64],
        similarity_metric='cosine',
        temperature=0.1
    )
    total_params = model.get_model_info()['total_params']
    print(f"   ✅ Model created with {total_params:,} parameters")
    
    # 2. Configure metrics
    print("\n📝 Step 2: Configuring Evaluation Metrics...")
    metrics_config = MetricsConfig(
        k_values=[1, 5, 10, 20, 50, 100],
        log_to_wandb=False,
        compute_diversity=True,
        compute_ranking=True,
        compute_retrieval=True
    )
    print(f"   ✅ Metrics configured for K values: {metrics_config.k_values}")
    
    # 3. Create evaluator
    print("\n📝 Step 3: Creating Model Evaluator...")
    evaluator = create_metrics_evaluator(model, metrics_config)
    print("   ✅ TwoTowerEvaluator created")
    
    # 4. Simulate evaluation data
    print("\n📝 Step 4: Simulating Evaluation Data...")
    
    num_customers = 100
    num_articles = 500
    embedding_dim = 64
    
    customer_embeddings = torch.randn(num_customers, embedding_dim)
    article_embeddings = torch.randn(num_articles, embedding_dim)
    
    # Normalize embeddings
    customer_embeddings = torch.nn.functional.normalize(customer_embeddings, p=2, dim=1)
    article_embeddings = torch.nn.functional.normalize(article_embeddings, p=2, dim=1)
    
    # Create IDs
    customer_ids = [f"customer_{i:03d}" for i in range(num_customers)]
    article_ids = [f"article_{i:04d}" for i in range(num_articles)]
    
    # Simulate ground truth interactions
    ground_truth = []
    for i in range(num_customers):
        num_interactions = np.random.randint(2, 9)
        interacted_articles = np.random.choice(num_articles, num_interactions, replace=False)
        for article_idx in interacted_articles:
            ground_truth.append((customer_ids[i], article_ids[article_idx]))
    
    print(f"   ✅ Created {num_customers} customers, {num_articles} articles")
    print(f"   ✅ Generated {len(ground_truth)} ground truth interactions")
    sparsity = len(ground_truth) / (num_customers * num_articles) * 100
    print(f"   ✅ Sparsity: {sparsity:.4f}%")
    
    # 5. Evaluate retrieval performance
    print("\n📝 Step 5: Computing Retrieval Metrics...")
    retrieval_metrics = evaluator.evaluate_retrieval(
        customer_embeddings=customer_embeddings,
        article_embeddings=article_embeddings,
        customer_ids=customer_ids,
        article_ids=article_ids,
        ground_truth=ground_truth,
        k_values=[1, 5, 10, 20, 50, 100]
    )
    
    # 6. Display key metrics
    print("\n📊 EVALUATION RESULTS:")
    print("-" * 25)
    
    print("🎯 Retrieval Performance:")
    for k in [1, 5, 10, 20, 50, 100]:
        precision_key = f'precision@{k}'
        recall_key = f'recall@{k}'
        accuracy_key = f'top_{k}_accuracy'
        
        if precision_key in retrieval_metrics:
            precision = retrieval_metrics[precision_key]
            recall = retrieval_metrics[recall_key]
            accuracy = retrieval_metrics[accuracy_key]
            print(f"   • Top-{k:2d}: Precision={precision:.4f}, Recall={recall:.4f}, Accuracy={accuracy:.4f}")
    
    print("\n🏆 Ranking Quality:")
    if 'map' in retrieval_metrics:
        map_score = retrieval_metrics['map']
        print(f"   • Mean Average Precision (MAP): {map_score:.4f}")
    if 'mrr' in retrieval_metrics:
        mrr_score = retrieval_metrics['mrr']
        print(f"   • Mean Reciprocal Rank (MRR): {mrr_score:.4f}")
    
    # NDCG metrics
    ndcg_metrics = [k for k in retrieval_metrics.keys() if k.startswith('ndcg@')]
    if ndcg_metrics:
        print("   • NDCG Scores:")
        for metric in sorted(ndcg_metrics):
            k_value = metric.split('@')[1]
            score = retrieval_metrics[metric]
            print(f"     - NDCG@{k_value}: {score:.4f}")
    
    # 7. Demonstrate metrics tracking
    print("\n📝 Step 6: Demonstrating Metrics Tracking...")
    tracker = MetricsTracker(
        patience=5,
        min_delta=0.001,
        primary_metric='top_10_accuracy',
        mode='max'
    )
    
    # Simulate training epochs
    print("   🔄 Simulating training epochs...")
    
    epoch_1_metrics = {'top_10_accuracy': 0.1250, 'map': 0.0800}
    should_stop = tracker.update(epoch_1_metrics, epoch=1)
    print(f"     Epoch 1: Top-10 Acc={epoch_1_metrics['top_10_accuracy']:.4f} | Continue")
    
    epoch_2_metrics = {'top_10_accuracy': 0.1800, 'map': 0.1200}
    should_stop = tracker.update(epoch_2_metrics, epoch=2)
    print(f"     Epoch 2: Top-10 Acc={epoch_2_metrics['top_10_accuracy']:.4f} | Continue")
    
    epoch_3_metrics = {'top_10_accuracy': 0.2200, 'map': 0.1500}
    should_stop = tracker.update(epoch_3_metrics, epoch=3)
    print(f"     Epoch 3: Top-10 Acc={epoch_3_metrics['top_10_accuracy']:.4f} | Continue")
    
    epoch_4_metrics = {'top_10_accuracy': 0.2400, 'map': 0.1600}
    should_stop = tracker.update(epoch_4_metrics, epoch=4)
    print(f"     Epoch 4: Top-10 Acc={epoch_4_metrics['top_10_accuracy']:.4f} | Continue (Best)")
    
    # Get best metrics
    best_metrics = tracker.get_best_metrics()
    best_epoch = best_metrics['epoch']
    best_acc = best_metrics['top_10_accuracy']
    best_map = best_metrics['map']
    print(f"\n🏆 Best Performance (Epoch {best_epoch}):")
    print(f"   • Top-10 Accuracy: {best_acc:.4f}")
    print(f"   • MAP: {best_map:.4f}")
    
    # 8. Summary
    print("\n📈 EVALUATION SUMMARY:")
    print("-" * 22)
    
    top_10_acc = retrieval_metrics.get('top_10_accuracy', 0.0)
    if top_10_acc >= 0.3:
        performance_level = "🟢 Excellent"
    elif top_10_acc >= 0.2:
        performance_level = "🟡 Good"
    elif top_10_acc >= 0.1:
        performance_level = "🟠 Moderate"
    else:
        performance_level = "🔴 Needs Improvement"
    
    print(f"🎯 Overall Performance: {performance_level}")
    print(f"📊 Key Metric (Top-10 Accuracy): {top_10_acc:.4f}")
    
    print("\n💡 RECOMMENDATIONS:")
    if top_10_acc < 0.2:
        print("   • Consider increasing model complexity")
        print("   • Add more diverse training data")
    
    print("\n✅ Metrics evaluation demo completed!")
    return retrieval_metrics


def demo_wandb_integration():
    """Demonstrate Wandb integration"""
    print("\n🔗 Wandb Integration Example:")
    print("-" * 30)
    print("# To enable Wandb logging in practice:")
    print("import wandb")
    print("wandb.init(project='h-and-m-recommender')")
    print("")
    print("config = MetricsConfig(log_to_wandb=True)")
    print("evaluator = TwoTowerEvaluator(model, config)")
    print("metrics = evaluator.evaluate_on_dataloader(val_loader)")
    print("# Metrics automatically logged to Wandb!")


if __name__ == "__main__":
    # Run the demo
    metrics = demo_metrics_evaluation()
    
    # Show Wandb integration
    demo_wandb_integration()
    
    print("\n🎉 Demo completed successfully!")
    print("\n📚 Next Steps:")
    print("   1. Integrate metrics into your training loop")
    print("   2. Set up Wandb for experiment tracking")
    print("   3. Use MetricsTracker for early stopping")
    print("   4. Monitor all key TikTok recommender metrics")
    print("   5. Move to Step 7: Trainer Module implementation") 