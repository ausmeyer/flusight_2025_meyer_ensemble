#!/usr/bin/env python
"""
Two-Stage Probabilistic Forecasting Pipeline with Optuna Hyperparameter Optimization

This script implements a complete two-stage approach for probabilistic time series forecasting:

Stage 1: Train ordinary LightGBM for μ (location/median) parameter using Optuna
Stage 2: Train LightGBMLSS for σ (scale) parameter with frozen μ, using Optuna  

Key Features:
- Comprehensive CLI controls for all hyperparameters
- Dual hyperparameter optimization (separate for each stage)
- Support for multiple target locations
- Automated feature selection
- Proper data leakage prevention with rolling window validation
- Compatible with retrospective forecasting script

Usage:
    python src/train_two_stage.py --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \\
                                  --cut-off 2023-07-01 \\
                                  --locations California Texas \\
                                  --trials-stage1 100 \\
                                  --trials-stage2 100 \\
                                  --horizon 1
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

# Import utilities
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils.tabularizer import create_features, create_features_for_prediction
from utils.enhanced_features import create_enhanced_features, create_enhanced_features_for_prediction
from utils.scoring import mae

# Import from utils directory
from utils.lgbm_timeseries import TimeSeriesDataProcessor

# Import LightGBMLSS
try:
    import lightgbmlss
    from lightgbmlss.model import LightGBMLSS
    from lightgbmlss.distributions.Gaussian import Gaussian
    from utils.distributions import GaussianFrozenLoc
except ImportError:
    raise ImportError("lightgbmlss is required. Install with: pip install lightgbmlss")

warnings.filterwarnings("ignore")


class TwoStageTrainer:
    """
    Complete two-stage training pipeline with Optuna optimization for both stages.
    """
    
    def __init__(self, cut_off_date: str, horizon: int = 1, max_lags: int = 12, 
                 trials_stage1: int = 100, trials_stage2: int = 100, random_seed: int = 42,
                 use_enhanced_features: bool = False, n_features: int = 5,
                 use_log_transform: bool = False,
                 num_threads: int = None,
                 optuna_jobs: int = 1,
                 disable_stage2_pruning: bool = False):
        self.cut_off_date = pd.to_datetime(cut_off_date)
        self.horizon = horizon
        self.max_lags = max_lags
        self.trials_stage1 = trials_stage1
        self.trials_stage2 = trials_stage2
        self.random_seed = random_seed
        self.use_enhanced_features = use_enhanced_features
        self.n_features = n_features
        self.use_log_transform = use_log_transform
        self.processor = TimeSeriesDataProcessor()
        # Threading / parallelism controls
        self.num_threads = num_threads if num_threads and num_threads > 0 else (os.cpu_count() or 1)
        self.optuna_jobs = max(1, int(optuna_jobs))
        
        # Storage for results
        self.stage1_results = {}
        self.stage2_results = {}
        self.disable_stage2_pruning = disable_stage2_pruning
        
    def apply_log_transform(self, X: pd.DataFrame, y: np.ndarray = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """Apply log(1+x) transformation to features and optionally targets."""
        if not self.use_log_transform:
            return X, y
            
        # Transform features (add small constant to avoid log(0))
        X_transformed = np.log1p(np.maximum(X, 0))
        
        # Transform targets if provided
        y_transformed = None
        if y is not None:
            y_transformed = np.log1p(np.maximum(y, 0))
            
        return X_transformed, y_transformed
    
    def inverse_log_transform(self, values: np.ndarray) -> np.ndarray:
        """Apply inverse log transformation: exp(x) - 1."""
        if not self.use_log_transform:
            return values
        return np.expm1(values)
        
    def load_and_prepare_data(self, data_file: str, locations: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
        """Load data and prepare location-specific feature sets."""
        
        print(f"Loading data from {data_file}")
        
        # Load all data
        full_df = self.processor.load_and_pivot_data(data_file, exclude_locations=None)
        
        # Filter to cut-off date (training data only)
        # Last training date is the last date where data is available before the cutoff date
        train_df = full_df[full_df["date"] < self.cut_off_date].copy()
        
        print(f"Data loaded: {len(full_df)} total samples")
        print(f"Training data: {len(train_df)} samples up to {(train_df['date'].max() if len(train_df) > 0 else 'N/A')}")
        print(f"Cut-off date: {self.cut_off_date.strftime('%Y-%m-%d')}")
        print(f"Available locations: {[col for col in full_df.columns if col != 'date']}")
        
        # Feature selection for each location
        location_features = {}
        all_states = [col for col in full_df.columns if col != 'date']
        
        for location in locations:
            if location not in all_states:
                raise ValueError(f"Location {location} not found in data")
                
            covariate_states = [col for col in all_states if col != location]
            selected_states = self.feature_selection(train_df, location, covariate_states)
            location_features[location] = selected_states
            
        return train_df, full_df, location_features
        
    def feature_selection(self, df: pd.DataFrame, location: str, 
                         covariate_states: List[str]) -> List[str]:
        """Select top features using simple LightGBM on raw states."""
        
        print(f"Performing feature selection for {location}")
        
        # Use first half of training data for feature selection
        split_idx = len(df) // 2
        df_fs = df.iloc[:split_idx]
        
        # Prepare simple features: raw state values + week number
        df_features = df_fs[['date'] + covariate_states].copy()
        df_features['week_num'] = pd.to_datetime(df_features['date']).dt.isocalendar().week
        df_features['week_num'] = df_features['week_num'].clip(upper=52)
        df_features = df_features.drop('date', axis=1)
        
        # Target variable
        y_fs = df_fs[location]
        
        # Clean data
        combined_df = pd.concat([df_features, y_fs], axis=1).dropna()
        feature_cols = [col for col in combined_df.columns if col != location]
        X_fs = combined_df[feature_cols]
        y_fs_clean = combined_df[location]
        
        # Train feature selection model
        train_data = lgb.Dataset(X_fs, label=y_fs_clean, params={'verbose': -1})
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'random_state': self.random_seed,
            'num_threads': self.num_threads
        }
        
        model = lgb.train(params, train_data, num_boost_round=100, callbacks=[])
        
        # Get feature importance and select top N states
        importance_df = pd.DataFrame({
            'feature': X_fs.columns,
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        # Select top features (exclude week_num from state selection)
        selected_features = importance_df['feature'].head(self.n_features).tolist()
        selected_states = [f for f in selected_features if f != 'week_num']
        
        print(f"  Selected {len(selected_states)} states: {selected_states}")
        
        return selected_states
        
    def stage1_objective(self, trial: optuna.Trial, train_df: pd.DataFrame, 
                        location: str, selected_states: List[str],
                        precomputed: Tuple[np.ndarray, np.ndarray, pd.Index] = None) -> float:
        """Stage 1 Optuna objective function for point predictions (μ parameter).

        precomputed: optional tuple (X_full, y_full, time_idx) for enhanced features
                     to avoid recomputation across trials.
        """
        
        # Hyperparameter search space for LightGBM point model (match example.py exactly)
        num_boost_round = trial.suggest_int("n_estimators", 100, 1000, step=100)
        
        params = {
            "objective": "regression",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": trial.suggest_int("num_leaves", 5, 30),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "verbosity": -1,
            "random_state": self.random_seed,
            "num_threads": self.num_threads
        }
        
        # Suggest lags dynamically
        if self.use_enhanced_features:
            # Enhanced features handle lags internally
            # We don't need to suggest lags, but need a placeholder for the code
            lags = None  # Will be handled by enhanced_features
        else:
            # Allow lags starting from 1 (no lag_0 to avoid leakage)
            # For weekly hosp, use dense ladder of recent lags + seasonal
            n_recent = trial.suggest_int("n_recent_lags", 4, min(12, self.max_lags))
            use_seasonal = trial.suggest_categorical("use_seasonal_lags", [True, False])
            lags = list(range(1, n_recent + 1))  # Start from lag_1
            if use_seasonal:
                lags.extend([52])  # Add seasonal lags
        
        # Precompute enhanced features once (fast path)
        X_full = y_full = time_idx = None
        if self.use_enhanced_features:
            try:
                X_full, y_full, time_idx = create_enhanced_features(
                    train_df, location, selected_states,
                    end_date=None, horizon=self.horizon
                )
            except Exception:
                return float('inf')

        # Rolling window validation
        train_dates = sorted(train_df['date'].unique())
        original_length = len(train_dates)
        start_test = original_length // 2  # Use last half
        
        # Ensure sufficient data
        if self.use_enhanced_features:
            # Enhanced features need at least 52 weeks of history
            min_required = start_test + 52 + self.horizon
        else:
            max_lag = max(lags) if lags else 1
            min_required = start_test + max_lag + self.horizon
        if len(train_dates) < min_required:
            return float('inf')
            
        cv_scores = []
        folds_total = 0
        folds_kept = 0
        
        for i in range(start_test, len(train_dates) - self.horizon):
            try:
                folds_total += 1
                current_date = train_dates[i]
                train_end_date = current_date - pd.Timedelta(days=1)
                
                # Build training matrix up to (current_date - horizon week)
                if self.use_enhanced_features:
                    if X_full is None or len(X_full) == 0:
                        continue
                    mask = time_idx <= train_end_date
                    if mask.sum() < 1:
                        continue
                    X_cv_train = X_full[mask]
                    y_cv_train = y_full[mask]
                else:
                    X_cv_train, y_cv_train, _ = create_features(
                        train_df, location, selected_states, lags,
                        end_date=train_end_date, horizon=self.horizon
                    )
                
                if len(X_cv_train) < 25:
                    if self.stage2_debug and folds_kept < 3:
                        print(f"[ST2-DBG] {location} @ {current_date.date()} | small fold skipped: X={len(X_cv_train)}")
                    continue
                
                # Apply log transformation if enabled
                X_cv_train, y_cv_train = self.apply_log_transform(X_cv_train, y_cv_train)
                    
                # Build single validation point
                if self.use_enhanced_features:
                    # Fast path: take the precomputed row at the anchor date
                    idx = np.where(time_idx == current_date)[0]
                    if len(idx) == 0:
                        # Fallback: last row <= current_date
                        idx = np.where(time_idx <= current_date)[0]
                        if len(idx) == 0:
                            continue
                        row_idx = idx[-1]
                    else:
                        row_idx = idx[0]
                    X_pred = X_full[row_idx:row_idx+1]
                    # Get target value directly from data
                    target_date = current_date + pd.Timedelta(weeks=self.horizon)
                    target_rows = train_df.loc[train_df['date'] == target_date, location]
                    if len(target_rows) == 0:
                        continue
                    target_value = target_rows.iloc[0]
                else:
                    X_pred, _ = create_features_for_prediction(
                        train_df, location, selected_states, lags,
                        anchor_date=current_date, horizon=self.horizon
                    )
                    # Get target value directly from data
                    target_date = current_date + pd.Timedelta(weeks=self.horizon)
                    target_rows = train_df.loc[train_df['date'] == target_date, location]
                    if len(target_rows) == 0:
                        continue
                    target_value = target_rows.iloc[0]
                
                if len(X_pred) == 0:
                    continue
                
                # Apply log transformation to prediction features
                X_pred, _ = self.apply_log_transform(X_pred, None)
                
                # Train model
                dtrain = lgb.Dataset(X_cv_train, label=y_cv_train, params={'verbose': -1})
                model = lgb.train(params, dtrain, num_boost_round=num_boost_round, callbacks=[])
                
                # Predict and score
                prediction = model.predict(X_pred[-1:])
                # Inverse transform prediction if using log
                prediction = self.inverse_log_transform(prediction)
                point_mae = abs(prediction[0] - target_value)
                cv_scores.append(point_mae)

                # Optuna pruning: report intermediate CV mean and allow prune
                if not self.disable_stage2_pruning:
                    trial.report(float(np.mean(cv_scores)), step=len(cv_scores))
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    
            except Exception:
                continue
                
        if len(cv_scores) == 0:
            # If pruning is disabled, return a large loss instead of pruning
            if getattr(self, 'disable_stage2_pruning', False):
                return float('inf')
            else:
                raise optuna.exceptions.TrialPruned()
            
        return np.mean(cv_scores)
    
    def stage2_objective(self, trial: optuna.Trial, train_df: pd.DataFrame, location: str, 
                        selected_states: List[str], stage1_model: Dict, frozen_mu: np.ndarray,
                        precomputed: Tuple[np.ndarray, np.ndarray, pd.Index] = None) -> float:
        """Stage 2 Optuna objective function for scale predictions (σ parameter) with frozen μ."""
        
        # Hyperparameter search space for LightGBMLSS scale model (tighter bounds for stability)
        num_boost_round = trial.suggest_int("n_estimators", 200, 800, step=100)
        
        params = {
            "learning_rate": 0.05,
            "num_leaves": trial.suggest_int("num_leaves", 5, 30),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 2),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 3.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 3.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.05),
            "feature_pre_filter": False,
            "force_col_wise": True,
            "verbosity": -1,
            "verbose": -1,
            "random_state": self.random_seed,
            "num_threads": self.num_threads
        }
        
        # Use same lags as Stage 1
        lags = stage1_model['lags']

        # Precompute enhanced features once (fast path)
        X_full = y_full = time_idx = None
        if self.use_enhanced_features:
            try:
                X_full, y_full, time_idx = create_enhanced_features(
                    train_df, location, selected_states,
                    end_date=None, horizon=self.horizon
                )
            except Exception:
                return float('inf')

        # Rolling window validation
        train_dates = sorted(train_df['date'].unique())
        original_length = len(train_dates)
        start_test = original_length // 2  # Use last half
        
        if not self.use_enhanced_features and lags:
            max_lag = max(lags)
            min_required = start_test + max_lag + self.horizon
            if len(train_dates) < min_required:
                return float('inf')
        else:
            # Enhanced features handle requirements internally
            min_required = start_test + 52 + self.horizon  # Conservative estimate
            if len(train_dates) < min_required:
                return float('inf')
            
        cv_scores = []
        folds_total = 0
        folds_kept = 0
        
        for i in range(start_test, len(train_dates) - self.horizon):
            try:
                folds_total += 1
                current_date = train_dates[i]
                train_end_date = current_date - pd.Timedelta(days=1)
                
                # Build training matrix up to (current_date - horizon week)
                if self.use_enhanced_features:
                    if X_full is None or len(X_full) == 0:
                        continue
                    mask = time_idx <= train_end_date
                    if mask.sum() < 1:
                        continue
                    X_cv_train = X_full[mask]
                    y_cv_train = y_full[mask]
                else:
                    X_cv_train, y_cv_train, _ = create_features(
                        train_df, location, selected_states, lags,
                        end_date=train_end_date, horizon=self.horizon
                    )
                
                if len(X_cv_train) < 25:
                    continue
                
                # Apply log transformation if enabled
                X_cv_train, y_cv_train = self.apply_log_transform(X_cv_train, y_cv_train)
                
                # Train Stage 1 model on CV training data only (no data leakage)
                cv_stage1_params = stage1_model['best_params'].copy()
                cv_stage1_params.update({
                    'verbosity': -1,
                    'verbose': -1,
                    'random_state': self.random_seed,
                    'objective': 'regression',
                    'metric': 'mae',
                    'num_threads': self.num_threads
                })
                
                dtrain_cv = lgb.Dataset(X_cv_train, label=y_cv_train, params={'verbose': -1})
                cv_stage1_booster = lgb.train(cv_stage1_params, dtrain_cv,
                                              num_boost_round=stage1_model['num_boost_round'], callbacks=[])
                
                # Get μ predictions from CV-trained Stage 1 model (no leakage)
                mu_predictions = cv_stage1_booster.predict(X_cv_train)
                if self.stage2_debug and folds_kept < 3:
                    try:
                        print(f"[ST2-DBG] {location} @ {current_date.date()} | mu[min,max,mean]=({np.min(mu_predictions):.3f},{np.max(mu_predictions):.3f},{np.mean(mu_predictions):.3f}) X={X_cv_train.shape}")
                    except Exception:
                        pass
                
                # Create init_score with frozen μ and zeros for σ (1-D format for LightGBM)
                init_score = np.column_stack([
                    mu_predictions,
                    np.zeros_like(mu_predictions)
                ]).ravel(order='F')
                
                # Build validation point using prediction-mode creators
                if self.use_enhanced_features:
                    # Select the row with anchor_date == current_date
                    idx = np.where(time_idx == current_date)[0]
                    if len(idx) == 0:
                        continue
                    X_pred = X_full[idx[0]:idx[0]+1]
                    # Get target value directly from data
                    target_date = current_date + pd.Timedelta(weeks=self.horizon)
                    target_rows = train_df.loc[train_df['date'] == target_date, location]
                    if len(target_rows) == 0:
                        continue
                    target_value = target_rows.iloc[0]
                else:
                    X_pred, _ = create_features_for_prediction(
                        train_df, location, selected_states, lags,
                        anchor_date=current_date, horizon=self.horizon
                    )
                    # Get target value directly from data
                    target_date = current_date + pd.Timedelta(weeks=self.horizon)
                    target_rows = train_df.loc[train_df['date'] == target_date, location]
                    if len(target_rows) == 0:
                        continue
                    target_value = target_rows.iloc[0]
                
                if len(X_pred) == 0:
                    continue
                
                # Apply log transformation to prediction features
                X_pred, _ = self.apply_log_transform(X_pred, None)
                
                # Train LightGBMLSS model with frozen μ
                dtrain = lgb.Dataset(X_cv_train, label=y_cv_train, init_score=init_score, params={'verbose': -1})
                lgbmlss_model = LightGBMLSS(GaussianFrozenLoc())
                # Train directly (no wrapper)
                lgbmlss_model.train(params, dtrain, num_boost_round=num_boost_round)
                
                # Predict validation point using CV-trained Stage 1 model (no leakage)
                mu_pred_val = cv_stage1_booster.predict(X_pred[-1:])
                # Inverse transform mu if using log
                mu_pred_val = self.inverse_log_transform(mu_pred_val)
                
                dist_params = lgbmlss_model.predict(X_pred[-1:], pred_type="parameters")
                # Extract σ robustly
                if hasattr(dist_params, 'values'):
                    dist_params = dist_params.values
                if dist_params.ndim == 2:
                    if dist_params.shape[1] == 1:
                        sigma_pred = dist_params[0, 0]
                    else:
                        sigma_pred = dist_params[0, -1]
                else:
                    sigma_pred = dist_params[-1]
                if self.stage2_debug and folds_kept < 3:
                    try:
                        print(f"[ST2-DBG] {location} @ {current_date.date()} | sigma_pred={sigma_pred:.6f}")
                    except Exception:
                        pass
                
                # If using log transform, sigma is in log space - need to scale it
                if self.use_log_transform:
                    # Convert sigma from log space to original space
                    # This is approximate - assumes log-normal distribution
                    sigma_pred = sigma_pred * mu_pred_val[0]
                
                # Calculate WIS using quantile predictions (proper Stage 2 metric)
                from scipy.stats import norm
                # Define CDC FluSight quantiles
                CDC_QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
                                         0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
                # Import WIS function from utils
                from utils.wis_function_python import wis as cdc_wis
                
                sigma_pred = max(sigma_pred, 1e-6)  # Ensure positive
                
                # Generate quantile predictions from Gaussian(μ, σ)
                quantile_preds = norm.ppf(CDC_QUANTILES, loc=mu_pred_val[0], scale=sigma_pred)
                # Ensure non-negative (for hospitalization data)
                quantile_preds = np.maximum(quantile_preds, 0.0)
                
                # Calculate WIS for this single prediction
                wis_score = cdc_wis(np.array([target_value]), 
                                   quantile_preds.reshape(1, -1), 
                                   CDC_QUANTILES)
                cv_scores.append(np.mean(wis_score))
                folds_kept += 1

                # Optuna pruning: report intermediate CV mean and allow prune
                if not self.disable_stage2_pruning:
                    trial.report(float(np.mean(cv_scores)), step=len(cv_scores))
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
                    
            except Exception as e:
                if self.stage2_debug and folds_kept < 3:
                    print(f"[ST2-DBG] {location} @ {current_date.date()} | Exception: {e}")
                continue
                
        if len(cv_scores) == 0:
            if self.disable_stage2_pruning:
                return float('inf')
            else:
                raise optuna.exceptions.TrialPruned()
        if self.stage2_debug:
            try:
                print(f"[ST2-DBG] {location} | folds kept/total={folds_kept}/{folds_total} | mean WIS={np.mean(cv_scores):.6f}")
            except Exception:
                pass
        
        return np.mean(cv_scores)

    def prepare_stage2_folds(self, train_df: pd.DataFrame, location: str,
                             selected_states: List[str], stage1_model: Dict) -> List[Dict]:
        """Precompute per-fold data for Stage 2 to avoid retraining Stage 1 for every trial.

        Returns a list of dicts with keys:
          - dtrain: lgb.Dataset with label and init_score (frozen μ)
          - X_pred: 2D array with a single prediction row
          - target_value: float actual value at target date
        """
        folds = []

        # Lags and feature precompute
        lags = stage1_model['lags']

        X_full = y_full = time_idx = None
        if self.use_enhanced_features:
            try:
                X_full, y_full, time_idx = create_enhanced_features(
                    train_df, location, selected_states,
                    end_date=None, horizon=self.horizon
                )
            except Exception:
                return folds

        # Rolling dates
        train_dates = sorted(train_df['date'].unique())
        start_test = len(train_dates) // 2

        # Iterate folds once to build cache
        for i in range(start_test, len(train_dates) - self.horizon):
            try:
                current_date = train_dates[i]
                train_end_date = current_date - pd.Timedelta(days=1)

                # Build training matrix for this fold (enhanced: slice; else: compute)
                if self.use_enhanced_features:
                    if X_full is None or len(X_full) == 0:
                        continue
                    mask = time_idx <= train_end_date
                    if mask.sum() < 25:
                        continue
                    X_cv_train = X_full[mask]
                    y_cv_train = y_full[mask]
                else:
                    X_cv_train, y_cv_train, _ = create_features(
                        train_df, location, selected_states, lags,
                        end_date=train_end_date, horizon=self.horizon
                    )
                    if len(X_cv_train) < 25:
                        continue

                # Log transform if needed
                X_cv_train, y_cv_train = self.apply_log_transform(X_cv_train, y_cv_train)

                # Train Stage-1 model once for this fold to get μ
                cv_stage1_params = stage1_model['best_params'].copy()
                cv_stage1_params.update({
                    'verbosity': -1,
                    'verbose': -1,
                    'random_state': self.random_seed,
                    'objective': 'regression',
                    'metric': 'mae',
                    'num_threads': self.num_threads
                })
                dtrain_cv = lgb.Dataset(X_cv_train, label=y_cv_train, params={'verbose': -1})
                cv_stage1_booster = lgb.train(cv_stage1_params, dtrain_cv,
                                              num_boost_round=stage1_model['num_boost_round'], callbacks=[])
                mu_predictions = cv_stage1_booster.predict(X_cv_train)

                # Build init_score (μ frozen, σ zeros) in Fortran order
                init_score = np.column_stack([
                    mu_predictions, np.zeros_like(mu_predictions)
                ]).ravel(order='F')

                # Prediction row and target
                if self.use_enhanced_features:
                    idx = np.where(time_idx == current_date)[0]
                    if len(idx) == 0:
                        continue
                    X_pred = X_full[idx[0]:idx[0]+1]
                else:
                    X_pred, _ = create_features_for_prediction(
                        train_df, location, selected_states, lags,
                        anchor_date=current_date, horizon=self.horizon
                    )
                    if len(X_pred) == 0:
                        continue
                # Apply log transform to prediction features if enabled
                X_pred, _ = self.apply_log_transform(X_pred, None)

                # Target value at horizon
                target_date = current_date + pd.Timedelta(weeks=self.horizon)
                target_rows = train_df.loc[train_df['date'] == target_date, location]
                if len(target_rows) == 0:
                    continue
                target_value = target_rows.iloc[0]

                # Also cache μ prediction for the validation point (inverse-transformed if needed)
                mu_pred_val = cv_stage1_booster.predict(X_pred[-1:])
                mu_pred_val = self.inverse_log_transform(mu_pred_val)
                # Store raw arrays to safely rebuild Dataset per trial (avoids races / pre-filter issues)
                folds.append({
                    'X': X_cv_train,
                    'y': y_cv_train,
                    'init_score': init_score,
                    'X_pred': X_pred[-1:],
                    'target_value': target_value,
                    'mu_pred_val': float(mu_pred_val[0])
                })
            except Exception:
                continue

        return folds

    def stage2_objective_from_folds(self, trial: optuna.Trial, location: str,
                                     folds: List[Dict]) -> float:
        """Stage 2 objective using precomputed per-fold datasets and prediction rows."""
        # Hyperparameter search space for LightGBMLSS scale model
        num_boost_round = trial.suggest_int("n_estimators", 200, 800, step=100)
        params = {
            "learning_rate": 0.05,
            "num_leaves": trial.suggest_int("num_leaves", 5, 30),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 2),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 3.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 3.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.05),
            "feature_pre_filter": False,
            "force_col_wise": True,
            "verbosity": -1,
            "verbose": -1,
            "random_state": self.random_seed,
            "num_threads": self.num_threads
        }

        cv_scores = []
        folds_total = len(folds)
        folds_kept = 0

        for fold in folds:
            try:
                # Build Dataset per trial to avoid reusing the same Dataset across parallel trials
                X = fold['X']
                y = fold['y']
                init_score = fold['init_score']
                dtrain = lgb.Dataset(X, label=y, init_score=init_score, params={'verbose': -1}, free_raw_data=False)
                X_pred = fold['X_pred']
                target_value = fold['target_value']
                mu_pred_val = fold['mu_pred_val']

                # Train LightGBMLSS and predict parameters (no wrapper)
                lgbmlss_model = LightGBMLSS(GaussianFrozenLoc())
                lgbmlss_model.train(params, dtrain, num_boost_round=num_boost_round)

                dist_params = lgbmlss_model.predict(X_pred, pred_type="parameters")
                if hasattr(dist_params, 'values'):
                    dist_params = dist_params.values
                if dist_params.ndim == 2:
                    if dist_params.shape[1] == 1:
                        sigma_pred = dist_params[0, 0]
                    else:
                        sigma_pred = dist_params[0, -1]
                else:
                    sigma_pred = dist_params[-1]
                sigma_pred = max(sigma_pred, 1e-6)
                if self.use_log_transform:
                    sigma_pred = sigma_pred * mu_pred_val

                # WIS score for this fold
                from scipy.stats import norm
                CDC_QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                                          0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])
                from utils.wis_function_python import wis as cdc_wis
                quantile_preds = norm.ppf(CDC_QUANTILES, loc=mu_pred_val, scale=sigma_pred)
                quantile_preds = np.maximum(quantile_preds, 0.0)
                wis_score = cdc_wis(np.array([target_value]), quantile_preds.reshape(1, -1), CDC_QUANTILES)
                cv_scores.append(np.mean(wis_score))
                folds_kept += 1

            except Exception:
                continue

        if len(cv_scores) == 0:
            return float('inf')
        return np.mean(cv_scores)
        
    def train_stage1(self, train_df: pd.DataFrame, full_df: pd.DataFrame, location: str, selected_states: List[str]) -> Dict:
        """Train Stage 1: LightGBM for μ parameter."""
        
        print(f"\\n{'='*60}")
        print(f"STAGE 1: Training LightGBM point model for {location}")
        print(f"Cut-off date: {self.cut_off_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        # Hyperparameter optimization with Optuna
        print(f"Starting Stage 1 hyperparameter optimization with {self.trials_stage1} trials...")
        
        # Optuna pruner to stop bad trials early
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
        study = optuna.create_study(
            direction="minimize", 
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=pruner,
            study_name=f"stage1_{location}"
        )
        
        # Precompute enhanced features once for all trials (massive speedup)
        precomputed_enh = None
        if self.use_enhanced_features:
            precomputed_enh = create_enhanced_features(
                train_df, location, selected_states, end_date=None, horizon=self.horizon
            )

        study.optimize(
            lambda trial: self.stage1_objective(trial, train_df, location, selected_states, precomputed=precomputed_enh),
            n_trials=self.trials_stage1,
            n_jobs=self.optuna_jobs,
            show_progress_bar=True
        )
        
        # Get best parameters and train final model
        best_params = study.best_params.copy()
        num_boost_round = best_params.pop("n_estimators")
        
        if self.use_enhanced_features:
            best_lags = None  # Enhanced features handle lags internally
        else:
            # Check if parameters exist (for backward compatibility)
            if "n_recent_lags" in best_params:
                n_recent = best_params.pop("n_recent_lags")
                use_seasonal = best_params.pop("use_seasonal_lags", False)
                # Compute lags starting from 1 (not horizon)
                best_lags = list(range(1, n_recent + 1))
                if use_seasonal:
                    best_lags.extend([52])
            elif "n_lags" in best_params:
                # Old parameter name for backward compatibility
                n_lags = best_params.pop("n_lags")
                best_lags = list(range(1, n_lags + 1))
            else:
                # Default fallback
                best_lags = list(range(1, 9))
        
        print(f"Best Stage 1 parameters: {best_params}")
        print(f"Best lags: {best_lags}")
        print(f"Best CV MAE: {study.best_value:.6f}")
        
        # Create final training data
        if self.use_enhanced_features:
            X_train, y_train, _ = create_enhanced_features(
                train_df, location, selected_states,
                end_date=self.cut_off_date, horizon=self.horizon
            )
        else:
            X_train, y_train, _ = create_features(
                train_df, location, selected_states, best_lags, 
                end_date=self.cut_off_date, horizon=self.horizon
            )
        
        if len(X_train) == 0:
            raise ValueError("No training samples created with optimized parameters.")
            
        # Apply log transformation if enabled
        X_train, y_train = self.apply_log_transform(X_train, y_train)
            
        print(f"Final training data shape: {X_train.shape}")
        
        # Train final model with same parameters as example.py
        best_params.update({
            'objective': 'regression',
            'metric': 'mae',
            'verbosity': -1,
            'verbose': -1,
            'random_state': self.random_seed,
            'num_threads': self.num_threads
        })
        
        dtrain = lgb.Dataset(X_train, label=y_train, params={'verbose': -1})
        final_booster = lgb.train(best_params, dtrain, num_boost_round=num_boost_round, callbacks=[])
        
        # Use CV MAE as the performance metric
        print(f"Stage 1 CV MAE: {study.best_value:.6f}")
        
        return {
            'booster': final_booster,
            'best_params': best_params,
            'num_boost_round': num_boost_round,
            'selected_states': selected_states,
            'lags': best_lags,
            'cv_mae': study.best_value,
            'location': location,
            'stage': 1,
            'use_log_transform': self.use_log_transform
        }
        
    def train_stage2(self, train_df: pd.DataFrame, full_df: pd.DataFrame, location: str, stage1_model: Dict) -> Dict:
        """Train Stage 2: LightGBMLSS for σ parameter with frozen μ."""
        
        print(f"\\n{'='*60}")
        print(f"STAGE 2: Training LightGBMLSS scale model for {location}")
        print(f"Using frozen μ from Stage 1 model")
        print(f"{'='*60}")
        
        # Use same features and lags as Stage 1
        selected_states = stage1_model['selected_states']
        lags = stage1_model['lags']
        
        # Create training data for Stage 2 (same as Stage 1)
        if self.use_enhanced_features:
            X_train, y_train, _ = create_enhanced_features(
                train_df, location, selected_states,
                end_date=self.cut_off_date, horizon=self.horizon
            )
        else:
            X_train, y_train, _ = create_features(
                train_df, location, selected_states, lags, 
                end_date=self.cut_off_date, horizon=self.horizon
            )
        
        # Apply log transformation if enabled
        X_train, y_train = self.apply_log_transform(X_train, y_train)
        
        # Get μ predictions from Stage 1 model (in log space if transformed)
        mu_predictions = stage1_model['booster'].predict(X_train)
        
        # Hyperparameter optimization for Stage 2
        print(f"Starting Stage 2 hyperparameter optimization with {self.trials_stage2} trials...")
        
        pruner2 = None if self.disable_stage2_pruning else optuna.pruners.MedianPruner(n_warmup_steps=10)
        study = optuna.create_study(
            direction="minimize", 
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            pruner=pruner2,
            study_name=f"stage2_{location}"
        )
        
        # Precompute enhanced features once for all trials
        precomputed_enh2 = None
        if self.use_enhanced_features:
            precomputed_enh2 = create_enhanced_features(
                train_df, location, selected_states, end_date=None, horizon=self.horizon
            )

        # Precompute folds (cache μ and init_score per fold once)
        folds_cache = self.prepare_stage2_folds(train_df, location, selected_states, stage1_model)
        if len(folds_cache) == 0:
            raise RuntimeError("Stage 2 fold cache is empty; not enough data for validation folds.")

        study.optimize(
            lambda trial: self.stage2_objective_from_folds(trial, location, folds_cache),
            n_trials=self.trials_stage2,
            n_jobs=self.optuna_jobs,
            show_progress_bar=True
        )
        
        # Get best parameters and train final model
        best_params = study.best_params.copy()
        num_boost_round = best_params.pop("n_estimators")
        
        print(f"Best Stage 2 parameters: {best_params}")
        print(f"Best CV WIS: {study.best_value:.6f}")
        
        # Train final Stage 2 model with frozen μ
        init_score = np.column_stack([
            mu_predictions,
            np.zeros_like(mu_predictions)
        ]).ravel(order='F')
        
        best_params.update({
            'verbosity': -1,
            'verbose': -1,
            'feature_pre_filter': False,
            'force_col_wise': True,
            'random_state': self.random_seed,
            'num_threads': self.num_threads
        })
        
        dtrain = lgb.Dataset(X_train, label=y_train, init_score=init_score, params={'verbose': -1})
        lgbmlss_model = LightGBMLSS(GaussianFrozenLoc())
        lgbmlss_model.train(best_params, dtrain, num_boost_round=num_boost_round)
        
        # Use CV WIS as the performance metric
        print(f"Stage 2 CV WIS: {study.best_value:.6f}")
        print(f"Stage 2 training completed")
        
        return {
            'lgbmlss_model': lgbmlss_model,
            'best_params': best_params,
            'num_boost_round': num_boost_round,
            'selected_states': selected_states,
            'lags': lags,
            'cv_wis': study.best_value,
            'location': location,
            'stage': 2,
            'use_log_transform': self.use_log_transform
        }
        
    def train_all_locations(self, data_file: str, locations: List[str]) -> None:
        """Train two-stage models for all specified locations."""
        
        print(f"\\n{'='*80}")
        print(f"TWO-STAGE PROBABILISTIC FORECASTING PIPELINE")
        print(f"{'='*80}")
        print(f"Locations: {locations}")
        print(f"Cut-off date: {self.cut_off_date.strftime('%Y-%m-%d')}")
        print(f"Horizon: {self.horizon}")
        print(f"Stage 1 trials: {self.trials_stage1}")
        print(f"Stage 2 trials: {self.trials_stage2}")
        
        # Load and prepare data
        train_df, full_df, location_features = self.load_and_prepare_data(data_file, locations)
        
        # Train models for each location
        for location in locations:
            selected_states = location_features[location]
            
            # Stage 1: Train point model
            stage1_result = self.train_stage1(train_df, full_df, location, selected_states)
            self.stage1_results[location] = stage1_result
            
            # Stage 2: Train scale model with frozen μ
            stage2_result = self.train_stage2(train_df, full_df, location, stage1_result)
            self.stage2_results[location] = stage2_result
            
        print(f"\\n{'='*80}")
        print(f"TRAINING COMPLETED FOR ALL LOCATIONS")
        print(f"{'='*80}")
        
    def save_models(self) -> None:
        """Save all trained models and parameters."""
        
        print("\\nSaving trained models...")
        
        # Create output directories
        os.makedirs("models/point_mu", exist_ok=True)
        os.makedirs("models/scale_sigma", exist_ok=True)
        
        for location in self.stage1_results.keys():
            # Save Stage 1 model
            stage1_data = self.stage1_results[location]
            
            # Save booster
            booster_file = f"models/point_mu/{location}_h{self.horizon}_booster.txt"
            stage1_data['booster'].save_model(booster_file)
            
            # Save parameters
            params_file = f"models/point_mu/{location}_h{self.horizon}_best_params.pkl"
            save_params = {k: v for k, v in stage1_data.items() if k != 'booster'}
            with open(params_file, 'wb') as f:
                pickle.dump(save_params, f)
            
            # Save Stage 2 model
            stage2_data = self.stage2_results[location]
            
            # Save LightGBMLSS model (as pickle)
            model_file = f"models/scale_sigma/{location}_h{self.horizon}_lgbmlss_model.pkl"
            with open(model_file, 'wb') as f:
                pickle.dump(stage2_data['lgbmlss_model'], f)
            
            # Save parameters
            params_file = f"models/scale_sigma/{location}_h{self.horizon}_best_params.pkl"
            save_params = {k: v for k, v in stage2_data.items() if k != 'lgbmlss_model'}
            with open(params_file, 'wb') as f:
                pickle.dump(save_params, f)
            
            print(f"  Saved models for {location}")
            
        # Save combined hyperparameters for retrospective script
        combined_params = {}
        for location in self.stage1_results.keys():
            combined_params[location] = {
                'stage1': self.stage1_results[location],
                'stage2': self.stage2_results[location],
                'horizon': self.horizon,
                'cut_off_date': self.cut_off_date.strftime('%Y-%m-%d'),
                'use_log_transform': self.use_log_transform
            }
            # Remove unpicklable objects
            combined_params[location]['stage1'].pop('booster', None)
            combined_params[location]['stage2'].pop('lgbmlss_model', None)
            
        params_file = f"models/two_stage_hyperparameters_h{self.horizon}.pkl"
        with open(params_file, 'wb') as f:
            pickle.dump(combined_params, f)
            
        print(f"  Combined hyperparameters saved to: {params_file}")
        
    def print_summary(self) -> None:
        """Print training summary."""
        
        print(f"\\n{'='*80}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*80}")
        
        for location in self.stage1_results.keys():
            stage1 = self.stage1_results[location]
            stage2 = self.stage2_results[location]
            
            print(f"\\n{location}:")
            print(f"  Stage 1 (μ) - CV MAE: {stage1['cv_mae']:.6f}")
            print(f"  Stage 1 (μ) - Lags: {stage1['lags']}")
            print(f"  Stage 1 (μ) - Features: {stage1['selected_states']}")
            print(f"  Stage 2 (σ) - CV WIS: {stage2['cv_wis']:.6f}")
            print(f"  Ready for retrospective forecasting")


def main():
    """Main function for two-stage training pipeline."""
    
    parser = argparse.ArgumentParser(description='Two-Stage Probabilistic Forecasting Training Pipeline')
    
    # Data arguments
    parser.add_argument('--data-file', type=str, required=True,
                       help='Path to input data file')
    parser.add_argument('--cut-off', type=str, required=True,
                       help='Cut-off date for training (YYYY-MM-DD)')
    parser.add_argument('--locations', type=str, nargs='+', required=True,
                       help='Target locations to train models for')
    
    # Model arguments
    parser.add_argument('--horizon', type=int, default=1,
                       help='Forecast horizon (default: 1)')
    parser.add_argument('--max-lags', type=int, default=12,
                       help='Maximum number of lags to consider (default: 12)')
    parser.add_argument('--n-features', type=int, default=5,
                       help='Number of covariate features to select (default: 5)')
    parser.add_argument('--use-enhanced-features', action='store_true',
                       help='Use enhanced features including rolling stats and seasonality')
    parser.add_argument('--use-log-transform', action='store_true',
                       help='Apply log transformation to features and targets')
    
    # Optimization arguments
    parser.add_argument('--trials-stage1', type=int, default=100,
                       help='Number of Optuna trials for Stage 1 (default: 100)')
    parser.add_argument('--trials-stage2', type=int, default=100,
                       help='Number of Optuna trials for Stage 2 (default: 100)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no-stage2-pruning', action='store_true',
                       help='Disable Optuna pruning for Stage 2 trials')
    parser.add_argument('--num-threads', type=int, default=None,
                       help='Number of threads for LightGBM (default: os.cpu_count())')
    parser.add_argument('--optuna-jobs', type=int, default=1,
                       help='Number of parallel Optuna trials (default: 1)')
    parser.add_argument('--stage2-debug', action='store_true',
                       help='Print Stage 2 debugging info for early folds')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")
    
    # Initialize trainer
    trainer = TwoStageTrainer(
        cut_off_date=args.cut_off,
        horizon=args.horizon,
        max_lags=args.max_lags,
        trials_stage1=args.trials_stage1,
        trials_stage2=args.trials_stage2,
        random_seed=args.random_seed,
        use_enhanced_features=args.use_enhanced_features,
        n_features=args.n_features,
        use_log_transform=args.use_log_transform,
        num_threads=args.num_threads,
        optuna_jobs=args.optuna_jobs
    )

    # Apply runtime flags (backward-compatible with older __init__ signatures)
    trainer.disable_stage2_pruning = args.no_stage2_pruning
    trainer.stage2_debug = args.stage2_debug
    
    # Train models
    trainer.train_all_locations(args.data_file, args.locations)
    
    # Save results
    trainer.save_models()
    
    # Print summary
    trainer.print_summary()
    
    print(f"\\n{'='*80}")
    print(f"READY FOR RETROSPECTIVE FORECASTING")
    print(f"{'='*80}")
    print(f"Use: python src/generate_retrospective_forecasts.py \\")
    print(f"       --hyperparams models/two_stage_hyperparameters_h{args.horizon}.pkl \\")
    print(f"       --data-file {args.data_file} \\")
    print(f"       --cut-off {args.cut_off}")


if __name__ == "__main__":
    main()
