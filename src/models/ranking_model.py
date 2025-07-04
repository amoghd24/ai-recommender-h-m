"""
Stage 3: Ranking Model Implementation using CatBoost
Binary classification to predict purchase probability as specified in Lesson 2
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from catboost import CatBoostClassifier
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)


class RankingModel:
    """
    CatBoost-based ranking model for predicting purchase probability
    """
    
    def __init__(
        self,
        iterations: int = 1000,
        learning_rate: float = 0.1,
        depth: int = 6,
        random_seed: int = 42
    ):
        """
        Initialize ranking model
        
        Args:
            iterations: Number of boosting iterations
            learning_rate: Learning rate
            depth: Tree depth
            random_seed: Random seed for reproducibility
        """
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            loss_function='Logloss',
            eval_metric='AUC',
            verbose=False,
            auto_class_weights='Balanced'
        )
        
        self.categorical_features = None
        self.is_fitted = False
        
        logger.info(f"RankingModel initialized with {iterations} iterations")
    
    def prepare_features(self, ranking_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Prepare features for training/prediction
        
        Args:
            ranking_df: Ranking dataset with features and labels
            
        Returns:
            Tuple of (features, labels, categorical_feature_names)
        """
        # Remove ID columns and target
        feature_cols = [col for col in ranking_df.columns 
                       if col not in ['customer_id', 'article_id', 'label']]
        
        X = ranking_df[feature_cols].copy()
        y = ranking_df['label'] if 'label' in ranking_df.columns else None
        
        # Identify categorical features
        categorical_features = []
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                categorical_features.append(col)
                # Fill missing values for categorical
                X[col] = X[col].fillna('unknown')
            else:
                # Fill missing values for numerical
                X[col] = X[col].fillna(0)
        
        return X, y, categorical_features
    
    def train(self, ranking_df: pd.DataFrame, eval_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Train the ranking model
        
        Args:
            ranking_df: Training ranking dataset
            eval_df: Optional evaluation dataset
            
        Returns:
            Training metrics dictionary
        """
        logger.info("Training ranking model...")
        
        # Prepare training data
        X_train, y_train, categorical_features = self.prepare_features(ranking_df)
        self.categorical_features = categorical_features
        
        # Prepare evaluation data if provided
        eval_set = None
        if eval_df is not None:
            X_eval, y_eval, _ = self.prepare_features(eval_df)
            eval_set = (X_eval, y_eval)
        
        # Train model
        self.model.fit(
            X_train, y_train,
            cat_features=categorical_features,
            eval_set=eval_set,
            early_stopping_rounds=50 if eval_set else None,
            plot=False
        )
        
        self.is_fitted = True
        
        # Get training metrics
        try:
            train_auc = self.model.get_best_score()['learn']['AUC']
        except (KeyError, TypeError):
            # Fallback if AUC is not available
            train_auc = 0.0
            
        metrics = {
            'train_auc': train_auc,
            'feature_importance': dict(zip(X_train.columns, self.model.feature_importances_))
        }
        
        if eval_set:
            try:
                metrics['eval_auc'] = self.model.get_best_score()['validation']['AUC']
            except (KeyError, TypeError):
                metrics['eval_auc'] = 0.0
        
        logger.info(f"Training complete. Train AUC: {metrics['train_auc']:.4f}")
        return metrics
    
    def train_with_validation(
        self, 
        train_df: pd.DataFrame, 
        val_df: pd.DataFrame,
        metrics_tracker=None
    ) -> Dict:
        """
        Train with validation and optional metrics tracking
        
        Args:
            train_df: Training dataset
            val_df: Validation dataset
            metrics_tracker: Optional MetricsTracker for early stopping
            
        Returns:
            Final metrics dictionary
        """
        logger.info("Training ranking model with validation...")
        
        # Prepare data
        X_train, y_train, categorical_features = self.prepare_features(train_df)
        X_val, y_val, _ = self.prepare_features(val_df)
        self.categorical_features = categorical_features
        
        # Train model with validation
        self.model.fit(
            X_train, y_train,
            cat_features=categorical_features,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            plot=False
        )
        
        self.is_fitted = True
        
        # Evaluate on validation set
        val_probs = self.model.predict_proba(X_val)[:, 1]
        val_preds = self.model.predict(X_val)
        
        try:
            train_auc = self.model.get_best_score()['learn']['AUC']
            val_auc = self.model.get_best_score()['validation']['AUC']
        except (KeyError, TypeError):
            train_auc = 0.0
            val_auc = 0.0
            
        final_metrics = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_accuracy': accuracy_score(y_val, val_preds),
            'feature_importance': dict(zip(X_train.columns, self.model.feature_importances_))
        }
        
        # Update metrics tracker if provided
        if metrics_tracker:
            should_stop = metrics_tracker.update(final_metrics, epoch=0)
        
        logger.info(f"Training complete. Val AUC: {final_metrics['val_auc']:.4f}")
        return final_metrics
    
    def predict_proba(self, ranking_df: pd.DataFrame) -> np.ndarray:
        """
        Predict purchase probabilities
        
        Args:
            ranking_df: Dataset with features
            
        Returns:
            Array of purchase probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X, _, _ = self.prepare_features(ranking_df)
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (purchase)
        
        return probabilities
    
    def rank_candidates(
        self, 
        candidates_df: pd.DataFrame, 
        customer_features: pd.Series,
        top_k: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Rank candidate articles for a customer
        
        Args:
            candidates_df: Candidate articles with features
            customer_features: Customer features
            top_k: Number of top items to return
            
        Returns:
            Ranked candidates with scores
        """
        # Create customer-article pairs
        ranking_data = []
        for _, article in candidates_df.iterrows():
            # Combine customer and article features
            pair = {**customer_features.to_dict(), **article.to_dict()}
            ranking_data.append(pair)
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Predict scores
        scores = self.predict_proba(ranking_df)
        
        # Add scores and rank
        result = candidates_df.copy()
        result['ranking_score'] = scores
        result = result.sort_values('ranking_score', ascending=False)
        
        if top_k:
            result = result.head(top_k)
        
        return result
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get top feature importances
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.model.feature_names_,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def save(self, filepath: str) -> None:
        """Save model to file"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'categorical_features': self.categorical_features,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load(cls, filepath: str) -> 'RankingModel':
        """Load model from file"""
        model_data = joblib.load(filepath)
        
        # Create instance and restore state
        instance = cls()
        instance.model = model_data['model']
        instance.categorical_features = model_data['categorical_features']
        instance.is_fitted = model_data['is_fitted']
        
        logger.info(f"Model loaded from {filepath}")
        return instance 