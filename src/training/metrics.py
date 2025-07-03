"""
Training Metrics Module for H&M Two-Tower Recommender System

This module provides comprehensive evaluation metrics for the two-tower model
training pipeline, with support for wandb logging and integration with
the existing training components.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
from sklearn.metrics import average_precision_score, ndcg_score
import logging
from datetime import datetime
import wandb
from dataclasses import dataclass

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricsConfig:
    """Configuration for metrics computation"""
    k_values: List[int] = None
    log_to_wandb: bool = True
    compute_diversity: bool = True
    compute_ranking: bool = True
    compute_retrieval: bool = True
    
    def __post_init__(self):
        if self.k_values is None:
            self.k_values = [1, 5, 10, 20, 50, 100]


class TwoTowerMetrics:
    """
    Comprehensive evaluation metrics for Two-Tower recommender systems.
    
    This class implements industry-standard metrics for evaluating retrieval
    and ranking performance of recommendation systems, specifically designed
    for the H&M TikTok-like recommender training pipeline.
    """
    
    def __init__(self, config: MetricsConfig = None):
        """
        Initialize metrics evaluator.
        
        Args:
            config: MetricsConfig instance with evaluation settings
        """
        self.config = config or MetricsConfig()
        self.k_values = self.config.k_values
        
        # Initialize wandb if requested
        if self.config.log_to_wandb:
            try:
                # Check if wandb is already initialized
                if wandb.run is None:
                    logger.info("Wandb not initialized. Metrics will be computed but not logged.")
                    self.wandb_enabled = False
                else:
                    self.wandb_enabled = True
                    logger.info("Wandb integration enabled for metrics logging")
            except Exception as e:
                logger.warning(f"Wandb integration failed: {e}. Continuing without wandb logging.")
                self.wandb_enabled = False
        else:
            self.wandb_enabled = False
        
        logger.info(f"TwoTowerMetrics initialized with K values: {self.k_values}")
    
    def compute_retrieval_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k_values: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Compute retrieval metrics including Precision@K, Recall@K, and Top-K Accuracy.
        
        This implements the core retrieval metrics mentioned in the TikTok-like
        recommender documentation, including the important Top-100 accuracy metric.
        
        Args:
            predictions: Similarity scores tensor [batch_size, num_candidates]
            targets: Binary relevance labels [batch_size, num_candidates]
            k_values: List of K values (uses self.k_values if None)
            
        Returns:
            Dictionary with retrieval metrics
        """
        if k_values is None:
            k_values = self.k_values
        
        # Convert to numpy for easier computation
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        metrics = {}
        
        for k in k_values:
            if k > pred_np.shape[1]:
                continue
                
            # Get top-K predictions for each query
            top_k_indices = np.argsort(pred_np, axis=1)[:, -k:][:, ::-1]
            
            # Compute metrics
            precision_k = self._compute_precision_at_k(target_np, top_k_indices, k)
            recall_k = self._compute_recall_at_k(target_np, top_k_indices, k)
            accuracy_k = self._compute_top_k_accuracy(target_np, top_k_indices, k)
            
            metrics[f'precision@{k}'] = precision_k
            metrics[f'recall@{k}'] = recall_k
            metrics[f'top_{k}_accuracy'] = accuracy_k
            
            # F1 score
            if precision_k + recall_k > 0:
                metrics[f'f1@{k}'] = 2 * (precision_k * recall_k) / (precision_k + recall_k)
            else:
                metrics[f'f1@{k}'] = 0.0
        
        return metrics
    
    def compute_ranking_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k_values: Optional[List[int]] = None
    ) -> Dict[str, float]:
        """
        Compute ranking metrics including NDCG and MAP.
        
        Implements NDCG (Normalized Discounted Cumulative Gain) and MAP
        as specified in the TikTok recommender documentation.
        
        Args:
            predictions: Similarity scores tensor [batch_size, num_candidates]
            targets: Binary relevance labels [batch_size, num_candidates]
            k_values: List of K values (uses self.k_values if None)
            
        Returns:
            Dictionary with ranking metrics
        """
        if k_values is None:
            k_values = self.k_values
        
        # Convert to numpy
        pred_np = predictions.detach().cpu().numpy()
        target_np = targets.detach().cpu().numpy()
        
        metrics = {}
        
        # Mean Average Precision (MAP)
        map_scores = []
        for i in range(pred_np.shape[0]):
            if target_np[i].sum() > 0:  # Only compute MAP if there are relevant items
                ap = average_precision_score(target_np[i], pred_np[i])
                map_scores.append(ap)
        
        metrics['map'] = np.mean(map_scores) if map_scores else 0.0
        
        # NDCG@K
        for k in k_values:
            if k > pred_np.shape[1]:
                continue
                
            ndcg_scores = []
            for i in range(pred_np.shape[0]):
                if target_np[i].sum() > 0:  # Only compute NDCG if there are relevant items
                    # Reshape for sklearn
                    true_relevance = target_np[i:i+1]
                    pred_scores = pred_np[i:i+1]
                    
                    try:
                        ndcg = ndcg_score(true_relevance, pred_scores, k=k)
                        ndcg_scores.append(ndcg)
                    except ValueError:
                        # Handle edge cases where NDCG computation fails
                        ndcg_scores.append(0.0)
            
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        # Mean Reciprocal Rank (MRR)
        mrr_scores = []
        for i in range(pred_np.shape[0]):
            # Get sorted indices by prediction score (descending)
            sorted_indices = np.argsort(pred_np[i])[::-1]
            
            # Find the rank of the first relevant item
            for rank, idx in enumerate(sorted_indices, 1):
                if target_np[i, idx] == 1:
                    mrr_scores.append(1.0 / rank)
                    break
            else:
                mrr_scores.append(0.0)  # No relevant items found
        
        metrics['mrr'] = np.mean(mrr_scores)
        
        return metrics
    
    def compute_diversity_metrics(
        self,
        predictions: torch.Tensor,
        item_features: Optional[torch.Tensor] = None,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Compute diversity metrics for recommendations.
        
        Args:
            predictions: Similarity scores tensor [batch_size, num_candidates]
            item_features: Optional item feature tensor for diversity computation
            k: Number of top items to consider
            
        Returns:
            Dictionary with diversity metrics
        """
        pred_np = predictions.detach().cpu().numpy()
        
        metrics = {}
        
        # Get top-K recommendations for each query
        top_k_indices = np.argsort(pred_np, axis=1)[:, -k:][:, ::-1]
        
        # Intra-list diversity (average pairwise distance)
        if item_features is not None:
            features_np = item_features.detach().cpu().numpy()
            
            diversity_scores = []
            for i in range(top_k_indices.shape[0]):
                indices = top_k_indices[i]
                selected_features = features_np[indices]
                
                # Compute pairwise distances
                distances = []
                for j in range(len(indices)):
                    for l in range(j + 1, len(indices)):
                        dist = np.linalg.norm(selected_features[j] - selected_features[l])
                        distances.append(dist)
                
                if distances:
                    diversity_scores.append(np.mean(distances))
                else:
                    diversity_scores.append(0.0)
            
            metrics['intra_list_diversity'] = np.mean(diversity_scores)
        
        # Coverage (percentage of unique items recommended)
        all_recommended = np.unique(top_k_indices.flatten())
        total_items = pred_np.shape[1]
        metrics['coverage'] = len(all_recommended) / total_items
        
        # Gini coefficient (measures concentration of recommendations)
        item_counts = np.bincount(top_k_indices.flatten(), minlength=total_items)
        metrics['gini_coefficient'] = self._compute_gini_coefficient(item_counts)
        
        return metrics
    
    def compute_all_metrics(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        item_features: Optional[torch.Tensor] = None,
        k_values: Optional[List[int]] = None,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            predictions: Similarity scores tensor [batch_size, num_candidates]
            targets: Binary relevance labels [batch_size, num_candidates]
            item_features: Optional item feature tensor for diversity computation
            k_values: List of K values (uses self.k_values if None)
            prefix: Prefix for metric names (e.g., 'train_', 'val_')
            
        Returns:
            Dictionary with all metrics
        """
        logger.info("Computing comprehensive evaluation metrics...")
        
        all_metrics = {}
        
        # Retrieval metrics
        if self.config.compute_retrieval:
            retrieval_metrics = self.compute_retrieval_metrics(predictions, targets, k_values)
            all_metrics.update(retrieval_metrics)
        
        # Ranking metrics
        if self.config.compute_ranking:
            ranking_metrics = self.compute_ranking_metrics(predictions, targets, k_values)
            all_metrics.update(ranking_metrics)
        
        # Diversity metrics
        if self.config.compute_diversity:
            diversity_metrics = self.compute_diversity_metrics(predictions, item_features)
            all_metrics.update(diversity_metrics)
        
        # Add prefix to metric names
        if prefix:
            all_metrics = {f"{prefix}{k}": v for k, v in all_metrics.items()}
        
        # Log to wandb if enabled
        if self.wandb_enabled:
            self.log_to_wandb(all_metrics)
        
        logger.info(f"Computed {len(all_metrics)} evaluation metrics")
        return all_metrics
    
    def log_to_wandb(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to wandb.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for logging
        """
        if self.wandb_enabled:
            try:
                if step is not None:
                    wandb.log(metrics, step=step)
                else:
                    wandb.log(metrics)
                logger.debug(f"Logged {len(metrics)} metrics to wandb")
            except Exception as e:
                logger.warning(f"Failed to log metrics to wandb: {e}")
    
    def _compute_precision_at_k(self, targets: np.ndarray, top_k_indices: np.ndarray, k: int) -> float:
        """Compute Precision@K"""
        precisions = []
        for i in range(targets.shape[0]):
            relevant_in_top_k = targets[i, top_k_indices[i]].sum()
            precisions.append(relevant_in_top_k / k)
        return np.mean(precisions)
    
    def _compute_recall_at_k(self, targets: np.ndarray, top_k_indices: np.ndarray, k: int) -> float:
        """Compute Recall@K"""
        recalls = []
        for i in range(targets.shape[0]):
            total_relevant = targets[i].sum()
            if total_relevant > 0:
                relevant_in_top_k = targets[i, top_k_indices[i]].sum()
                recalls.append(relevant_in_top_k / total_relevant)
            else:
                recalls.append(0.0)
        return np.mean(recalls)
    
    def _compute_top_k_accuracy(self, targets: np.ndarray, top_k_indices: np.ndarray, k: int) -> float:
        """Compute Top-K Accuracy (Hit Rate)"""
        hits = []
        for i in range(targets.shape[0]):
            has_relevant = targets[i, top_k_indices[i]].sum() > 0
            hits.append(float(has_relevant))
        return np.mean(hits)
    
    def _compute_gini_coefficient(self, values: np.ndarray) -> float:
        """Compute Gini coefficient for measuring inequality"""
        if len(values) == 0:
            return 0.0
        
        # Sort values
        sorted_values = np.sort(values)
        n = len(sorted_values)
        
        # Compute Gini coefficient
        cumsum = np.cumsum(sorted_values)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0.0
        
        return gini


class TwoTowerEvaluator:
    """
    High-level evaluator for Two-Tower models with convenient evaluation methods.
    Integrates with the training pipeline and provides wandb logging.
    """
    
    def __init__(self, model, config: MetricsConfig = None):
        """
        Initialize evaluator with a Two-Tower model.
        
        Args:
            model: TwoTowerModel instance
            config: MetricsConfig instance
        """
        self.model = model
        self.config = config or MetricsConfig()
        self.metrics = TwoTowerMetrics(self.config)
        logger.info("TwoTowerEvaluator initialized")
    
    def evaluate_on_dataloader(
        self,
        dataloader,
        device: torch.device = torch.device('cpu'),
        max_batches: Optional[int] = None,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Evaluate the model on a data loader.
        
        Args:
            dataloader: DataLoader with evaluation data
            device: Device to run evaluation on
            max_batches: Maximum number of batches to evaluate (None for all)
            prefix: Prefix for metric names (e.g., 'train_', 'val_')
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("Starting evaluation on dataloader...")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                
                # Extract features and labels
                customer_features = batch['customer_features']
                article_features = batch['article_features']
                labels = batch['label']
                
                # Move to device
                customer_features = {k: v.to(device) for k, v in customer_features.items()}
                article_features = {k: v.to(device) for k, v in article_features.items()}
                labels = labels.to(device)
                
                # Get predictions
                similarities = self.model(customer_features, article_features)
                
                # For evaluation, we need to create a matrix of similarities
                # where each row represents a query and columns represent candidates
                batch_size = similarities.shape[0]
                
                # Create target matrix (diagonal for positive pairs)
                target_matrix = torch.eye(batch_size, device=device)
                
                all_predictions.append(similarities)
                all_targets.append(target_matrix)
        
        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute metrics
        metrics = self.metrics.compute_all_metrics(predictions, targets, prefix=prefix)
        
        logger.info(f"Evaluation completed. Top-10 accuracy: {metrics.get(f'{prefix}top_10_accuracy', 0.0):.4f}")
        return metrics
    
    def evaluate_retrieval(
        self,
        customer_embeddings: torch.Tensor,
        article_embeddings: torch.Tensor,
        customer_ids: List[str],
        article_ids: List[str],
        ground_truth: List[Tuple[str, str]],
        k_values: Optional[List[int]] = None,
        prefix: str = ""
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance given embeddings and ground truth.
        
        Args:
            customer_embeddings: Customer embeddings tensor [num_customers, embedding_dim]
            article_embeddings: Article embeddings tensor [num_articles, embedding_dim]
            customer_ids: List of customer IDs
            article_ids: List of article IDs
            ground_truth: List of (customer_id, article_id) positive pairs
            k_values: List of K values for evaluation
            prefix: Prefix for metric names
            
        Returns:
            Dictionary with retrieval metrics
        """
        logger.info("Evaluating retrieval performance...")
        
        # Compute similarity matrix
        similarities = torch.matmul(customer_embeddings, article_embeddings.transpose(0, 1))
        
        # Create ground truth matrix
        ground_truth_set = set(ground_truth)
        targets = torch.zeros_like(similarities)
        
        for i, customer_id in enumerate(customer_ids):
            for j, article_id in enumerate(article_ids):
                if (customer_id, article_id) in ground_truth_set:
                    targets[i, j] = 1.0
        
        # Compute metrics
        metrics = self.metrics.compute_retrieval_metrics(similarities, targets, k_values)
        
        # Add prefix
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        # Log to wandb if enabled
        if self.metrics.wandb_enabled:
            self.metrics.log_to_wandb(metrics)
        
        logger.info(f"Retrieval evaluation completed. MAP: {metrics.get(f'{prefix}map', 0.0):.4f}")
        return metrics


class MetricsTracker:
    """
    Tracks metrics across training epochs with early stopping support.
    """
    
    def __init__(self, 
                 patience: int = 10,
                 min_delta: float = 0.001,
                 primary_metric: str = 'top_10_accuracy',
                 mode: str = 'max'):
        """
        Initialize metrics tracker.
        
        Args:
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum change in metric to qualify as improvement
            primary_metric: Primary metric to track for early stopping
            mode: 'max' or 'min' - whether higher or lower is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.primary_metric = primary_metric
        self.mode = mode
        
        self.best_score = float('-inf') if mode == 'max' else float('inf')
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.history = []
        
        logger.info(f"MetricsTracker initialized with primary metric: {primary_metric}")
    
    def update(self, metrics: Dict[str, float], epoch: int) -> bool:
        """
        Update tracker with new metrics.
        
        Args:
            metrics: Dictionary of metrics
            epoch: Current epoch number
            
        Returns:
            True if should stop early, False otherwise
        """
        # Store metrics
        metrics_with_epoch = {'epoch': epoch, **metrics}
        self.history.append(metrics_with_epoch)
        
        # Check if primary metric improved
        current_score = metrics.get(self.primary_metric, 0.0)
        
        if self.mode == 'max':
            improved = current_score > self.best_score + self.min_delta
        else:
            improved = current_score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
            logger.info(f"New best {self.primary_metric}: {current_score:.4f} at epoch {epoch}")
        else:
            self.epochs_without_improvement += 1
            logger.info(f"No improvement for {self.epochs_without_improvement} epochs")
        
        # Check for early stopping
        should_stop = self.epochs_without_improvement >= self.patience
        if should_stop:
            logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
        
        return should_stop
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get the best metrics recorded"""
        if not self.history:
            return {}
        
        # Find metrics from best epoch
        best_metrics = next(
            (m for m in self.history if m['epoch'] == self.best_epoch),
            self.history[-1]
        )
        
        return best_metrics
    
    def get_metrics_history(self) -> List[Dict[str, float]]:
        """Get full metrics history"""
        return self.history


def create_metrics_evaluator(model, config: MetricsConfig = None) -> TwoTowerEvaluator:
    """
    Factory function to create a TwoTowerEvaluator.
    
    Args:
        model: TwoTowerModel instance
        config: MetricsConfig instance
        
    Returns:
        TwoTowerEvaluator instance
    """
    return TwoTowerEvaluator(model, config)


def create_metrics_tracker(
    patience: int = 10,
    min_delta: float = 0.001,
    primary_metric: str = 'top_10_accuracy',
    mode: str = 'max'
) -> MetricsTracker:
    """
    Factory function to create a MetricsTracker.
    
    Args:
        patience: Number of epochs to wait before early stopping
        min_delta: Minimum change in metric to qualify as improvement
        primary_metric: Primary metric to track for early stopping
        mode: 'max' or 'min' - whether higher or lower is better
        
    Returns:
        MetricsTracker instance
    """
    return MetricsTracker(patience, min_delta, primary_metric, mode) 