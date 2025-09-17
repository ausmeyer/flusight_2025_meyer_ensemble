#!/usr/bin/env python
"""
SVM Regression Probabilistic Forecasting Pipeline with Optuna Hyperparameter Optimization

This script implements a Support Vector Machine regression approach for probabilistic time series forecasting:

Key Features:
- Two-way data split (first half for feature selection; second half for hyperparameter tuning and residual-based interval calibration)
- Probabilistic forecasts using empirical residuals (log1p/pct/original scale) collected with dynamic, expanding-window validation
- MAE-only Optuna objective (point skill), with separate residual-based calibration for WIS
- Optional bias correction and persistence blending for improved median MAE
- Compatible with retrospective forecasting script

Usage:
    python src/train_svm.py --data-file data/imputed_and_stitched_hosp_2025-05-24.csv \
                            --cut-off 2023-07-01 \
                            --locations California Texas \
                            --trials 100 \
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
import optuna
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from scipy import stats
from contextlib import redirect_stdout
from io import StringIO

# Import utilities
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from utils.tabularizer import create_features, create_features_for_prediction
from utils.enhanced_features import create_enhanced_features, create_enhanced_features_for_prediction
from utils.scoring import mae
from utils.wis_function_python import wis as cdc_wis
from utils.lgbm_timeseries import TimeSeriesDataProcessor

warnings.filterwarnings("ignore")

# CDC FluSight quantiles
CDC_QUANTILES = np.array([
    0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99
])


class SVMTrainer:
    """
    SVM regression training pipeline with Optuna optimization and probabilistic forecasting.
    """

    def __init__(self, cut_off_date: str, horizon: int = 1, max_lags: int = 12,
                 trials: int = 100, random_seed: int = 42,
                 use_enhanced_features: bool = False, n_features: int = 5,
                 use_log_transform: bool = False,
                 residual_scale: str = 'log1p',
                 kernels: List[str] = None,
                 cv_max_anchors: int = 30,
                 seasonal_in_search: bool = True):
        self.cut_off_date = pd.to_datetime(cut_off_date)
        self.horizon = horizon
        self.max_lags = max_lags
        self.trials = trials
        self.random_seed = random_seed
        self.use_enhanced_features = use_enhanced_features
        self.n_features = n_features
        self.use_log_transform = use_log_transform
        self.processor = TimeSeriesDataProcessor()
        # Residual modeling for probabilistic intervals
        # Options: 'log1p' (default), 'pct' (relative), or 'none' (original scale)
        self.residual_scale = residual_scale
        # Hyperparameter search constraints for speed
        self.kernels = kernels if kernels is not None else ['rbf']
        self.cv_max_anchors = cv_max_anchors
        self.seasonal_in_search = seasonal_in_search

        # Storage for results
        self.model_results = {}
        self.error_distributions = {}

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
        """Select top features using simple SVM on raw states."""

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

        # Standardize features for SVM
        scaler = StandardScaler()
        X_fs_scaled = scaler.fit_transform(X_fs)

        # Train simple SVM for feature selection
        svm = SVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale')
        svm.fit(X_fs_scaled, y_fs_clean)

        # Calculate feature importance using permutation importance
        from sklearn.inspection import permutation_importance
        result = permutation_importance(svm, X_fs_scaled, y_fs_clean,
                                      n_repeats=5, random_state=self.random_seed)

        # Get feature importance
        importance_df = pd.DataFrame({
            'feature': X_fs.columns,
            'importance': result.importances_mean
        }).sort_values('importance', ascending=False)

        # Select top features (exclude week_num from state selection)
        selected_features = importance_df['feature'].head(self.n_features).tolist()
        selected_states = [f for f in selected_features if f != 'week_num']

        print(f"  Selected {len(selected_states)} states: {selected_states}")

        return selected_states

    def generate_error_distribution(self, train_data: pd.DataFrame, location: str,
                                   selected_states: List[str], best_params: Dict,
                                   lags: List[int]) -> Dict[int, List[float]]:
        """(Deprecated) Generate error distribution using a mid validation window.
        Kept for backward compatibility but superseded by collect_dynamic_residuals.
        """

        # Split training data into halves (legacy code previously used thirds)
        train_dates = sorted(train_data['date'].unique())
        n_dates = len(train_dates)
        half_end = n_dates // 2

        # First half for initial training endpoint
        train_end_date = train_dates[half_end - 1]
        # Early part of second half for error distribution
        error_start_date = train_dates[half_end]
        error_end_date = train_dates[-self.horizon-1] if n_dates > self.horizon else train_dates[-1]

        # Build training features from first half
        if self.use_enhanced_features:
            X_train, y_train, _ = create_enhanced_features(
                train_data, location, selected_states,
                end_date=train_end_date, horizon=self.horizon
            )
        else:
            X_train, y_train, _ = create_features(
                train_data, location, selected_states, lags,
                end_date=train_end_date, horizon=self.horizon
            )

        if len(X_train) == 0:
            return {h: [] for h in range(1, 5)}

        # Apply log transformation if enabled
        X_train, y_train = self.apply_log_transform(X_train, y_train)

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train SVM model
        svm = SVR(**best_params)
        svm.fit(X_train_scaled, y_train)

        # Generate errors on the validation half (expanding window)
        errors_by_horizon = {h: [] for h in range(1, min(5, self.horizon + 1))}

        error_dates = [d for d in train_dates if error_start_date <= d <= error_end_date]

        for i, current_date in enumerate(error_dates[:-self.horizon]):
            # Expanding window: include all data up to current_date
            if self.use_enhanced_features:
                X_pred, _ = create_enhanced_features_for_prediction(
                    train_data, location, selected_states,
                    anchor_date=current_date, horizon=self.horizon
                )
            else:
                X_pred, _ = create_features_for_prediction(
                    train_data, location, selected_states, lags,
                    anchor_date=current_date, horizon=self.horizon
                )

            if len(X_pred) == 0:
                continue

            # Apply log transformation to prediction features
            X_pred, _ = self.apply_log_transform(X_pred, None)
            X_pred_scaled = scaler.transform(X_pred[-1:])

            # Make prediction
            prediction = svm.predict(X_pred_scaled)
            prediction = self.inverse_log_transform(prediction)

            # Get actual value
            target_date = current_date + pd.Timedelta(weeks=self.horizon)
            if target_date in train_data['date'].values:
                actual = train_data.loc[train_data['date'] == target_date, location].iloc[0]
                error = actual - prediction[0]
                errors_by_horizon[self.horizon].append(error)

        return errors_by_horizon

    def collect_dynamic_residuals(self, train_df: pd.DataFrame, location: str,
                                  selected_states: List[str], best_params: Dict,
                                  lags: List[int]) -> Dict[str, List[float]]:
        """
        Collect residuals on a validation window using the dynamic training
        procedure that matches generation. Returns a dict with keys:
          - 'scale': residual scale ('log1p'|'pct'|'none')
          - 'residuals': list of residuals at the chosen scale
        """
        train_dates = sorted(train_df['date'].unique())
        n_dates = len(train_dates)
        half_end = n_dates // 2

        # Validation anchors: use the second half (like generation pre-cutoff)
        val_full = train_dates[half_end: n_dates - self.horizon]
        if len(val_full) > self.cv_max_anchors:
            idxs = np.linspace(0, len(val_full) - 1, num=self.cv_max_anchors, dtype=int)
            val_dates = [val_full[i] for i in idxs]
        else:
            val_dates = val_full
        residuals = []  # residuals on configured scale for intervals
        residuals_raw = []  # raw residuals (actual - point_pred) for bias/MAE calibration
        blend_triplets = []  # (svm_pred, baseline_pred, actual)

        for current_date in val_dates:
            try:
                # Train on data up to the day before current_date (expanding window)
                train_end_date = current_date - pd.Timedelta(days=1)

                if self.use_enhanced_features:
                    X_train, y_train, _ = create_enhanced_features(
                        train_df, location, selected_states,
                        end_date=train_end_date, horizon=self.horizon
                    )
                else:
                    X_train, y_train, _ = create_features(
                        train_df, location, selected_states, lags,
                        end_date=train_end_date, horizon=self.horizon
                    )

                if len(X_train) < 25:
                    continue

                # Optional log transform on features/targets for SVM fitting
                X_train_t, y_train_t = self.apply_log_transform(X_train, y_train)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_t)
                svm = SVR(**best_params)
                svm.fit(X_train_scaled, y_train_t)

                # Build prediction features at current_date
                if self.use_enhanced_features:
                    X_pred, _ = create_enhanced_features_for_prediction(
                        train_df, location, selected_states,
                        anchor_date=current_date, horizon=self.horizon
                    )
                else:
                    X_pred, _ = create_features_for_prediction(
                        train_df, location, selected_states, lags,
                        anchor_date=current_date, horizon=self.horizon
                    )

                if len(X_pred) == 0:
                    continue

                X_pred_t, _ = self.apply_log_transform(X_pred, None)
                X_pred_scaled = scaler.transform(X_pred_t[-1:])

                # Point prediction (undo feature/target log transform if used)
                point_pred = svm.predict(X_pred_scaled)[0]
                point_pred = self.inverse_log_transform(np.array([point_pred]))[0]

                # Actual
                target_date = current_date + pd.Timedelta(weeks=self.horizon)
                if target_date not in train_df['date'].values:
                    continue
                actual = train_df.loc[train_df['date'] == target_date, location].iloc[0]

                # Residual on chosen scale
                if self.residual_scale == 'log1p':
                    r = np.log1p(max(actual, 0)) - np.log1p(max(point_pred, 0))
                elif self.residual_scale == 'pct':
                    denom = max(point_pred, 1e-8)
                    r = (actual - point_pred) / denom
                else:
                    r = actual - point_pred

                # Filter NaNs / infs
                if np.isfinite(r):
                    residuals.append(float(r))

                # Store raw residual for bias
                r_raw = actual - point_pred
                if np.isfinite(r_raw):
                    residuals_raw.append(float(r_raw))

                # Baseline (persistence) prediction for blending: last observed at anchor
                baseline_pred = train_df.loc[train_df['date'] == current_date, location]
                if len(baseline_pred) > 0 and np.isfinite(baseline_pred.iloc[0]):
                    blend_triplets.append((float(point_pred), float(baseline_pred.iloc[0]), float(actual)))
            except Exception:
                continue

        # Bias calibration (median bias on raw scale)
        point_bias = float(np.median(residuals_raw)) if len(residuals_raw) >= 5 else 0.0

        # Blending with persistence to improve MAE: search alpha in [0,1]
        blend_alpha = 1.0
        if len(blend_triplets) >= 20:
            alphas = np.linspace(0.0, 1.0, 21)
            best_mae = float('inf')
            best_alpha = 1.0
            for a in alphas:
                errs = []
                for sv, bl, ac in blend_triplets:
                    pred = a * (sv + point_bias) + (1.0 - a) * bl
                    errs.append(abs(pred - ac))
                mae_a = float(np.mean(errs)) if len(errs) > 0 else float('inf')
                if mae_a < best_mae:
                    best_mae = mae_a
                    best_alpha = float(a)
            blend_alpha = best_alpha

        return {
            'scale': self.residual_scale,
            'residuals': residuals,
            'point_bias': point_bias,
            'blend_alpha': blend_alpha
        }

    def create_quantile_forecasts(self, point_forecast: float, errors: List[float]) -> np.ndarray:
        """Convert point forecast to quantile forecasts using empirical errors."""

        if len(errors) < 10:
            # Fallback: use normal distribution with conservative std
            std = max(abs(point_forecast) * 0.25, 5.0)
            quantile_forecasts = [
                max(0, stats.norm.ppf(q, loc=point_forecast, scale=std))
                for q in CDC_QUANTILES
            ]
        else:
            # Use empirical error distribution
            error_quantiles = np.percentile(errors, CDC_QUANTILES * 100)
            quantile_forecasts = [
                max(0, point_forecast + eq)  # Ensure non-negative
                for eq in error_quantiles
            ]

        return np.array(quantile_forecasts)

    def objective(self, trial: optuna.Trial, train_df: pd.DataFrame,
                 location: str, selected_states: List[str]) -> float:
        """Optuna objective function optimizing MAE of point forecasts only."""

        # SVM hyperparameter search space (narrowed for speed)
        # Default: focus on 'rbf' which typically performs best
        kernel = trial.suggest_categorical('kernel', self.kernels)

        # Common parameters for all kernels (narrowed ranges for speed/stability)
        params = {
            'kernel': kernel,
            'C': trial.suggest_float('C', 0.5, 50.0, log=True),
            'epsilon': trial.suggest_float('epsilon', 0.003, 0.2, log=True)
        }

        # Kernel-specific parameters (reduced search)
        if kernel == 'rbf':
            # Use 'scale' to reduce branch count
            params['gamma'] = 'scale'
        elif kernel == 'sigmoid':
            params['gamma'] = 'scale'
            params['coef0'] = trial.suggest_float('coef0', 0.0, 5.0)

        # Suggest lags dynamically
        if self.use_enhanced_features:
            lags = None  # Enhanced features handle lags internally
        else:
            n_recent = trial.suggest_int("n_recent_lags", 3, self.max_lags)
            use_seasonal = trial.suggest_categorical("use_seasonal_lags", [True, False]) if self.seasonal_in_search else True
            lags = list(range(1, n_recent + 1))  # Start from lag_1
            if use_seasonal:
                lags.extend([52])  # Add seasonal lags

        # Split training data into halves
        train_dates = sorted(train_df['date'].unique())
        n_dates = len(train_dates)
        half_end = n_dates // 2

        # Ensure sufficient data before validation
        min_required = 52 if self.use_enhanced_features else (max(lags) if lags else 1)
        if half_end < min_required + 10:
            return float('inf')

        # Rolling window validation on second half (expanding window)
        mae_scores = []

        # Sample up to cv_max_anchors evenly across the second half
        third_dates_full = train_dates[half_end: (len(train_dates) - self.horizon)]
        if len(third_dates_full) == 0:
            raise optuna.exceptions.TrialPruned()
        if len(third_dates_full) > self.cv_max_anchors:
            idxs = np.linspace(0, len(third_dates_full) - 1, num=self.cv_max_anchors, dtype=int)
            third_dates = [third_dates_full[i] for i in idxs]
        else:
            third_dates = third_dates_full

        for current_date in third_dates:
            try:
                # EXPANDING WINDOW: Include all data from start through current position in validation half
                train_end_date = current_date - pd.Timedelta(days=1)

                # Build training features
                if self.use_enhanced_features:
                    X_train, y_train, _ = create_enhanced_features(
                        train_df, location, selected_states,
                        end_date=train_end_date, horizon=self.horizon
                    )
                else:
                    X_train, y_train, _ = create_features(
                        train_df, location, selected_states, lags,
                        end_date=train_end_date, horizon=self.horizon
                    )

                if len(X_train) < 25:
                    continue

                # Apply log transformation if enabled
                X_train, y_train = self.apply_log_transform(X_train, y_train)

                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)

                # Train model
                svm = SVR(**params)
                svm.fit(X_train_scaled, y_train)

                # Build prediction features
                if self.use_enhanced_features:
                    X_pred, _ = create_enhanced_features_for_prediction(
                        train_df, location, selected_states,
                        anchor_date=current_date, horizon=self.horizon
                    )
                else:
                    X_pred, _ = create_features_for_prediction(
                        train_df, location, selected_states, lags,
                        anchor_date=current_date, horizon=self.horizon
                    )

                if len(X_pred) == 0:
                    continue

                # Apply log transformation to prediction features
                X_pred, _ = self.apply_log_transform(X_pred, None)
                X_pred_scaled = scaler.transform(X_pred[-1:])

                # Make point prediction
                point_pred = svm.predict(X_pred_scaled)[0]
                point_pred = self.inverse_log_transform(np.array([point_pred]))[0]

                # Get actual value
                target_date = current_date + pd.Timedelta(weeks=self.horizon)
                actual = train_df.loc[train_df['date'] == target_date, location].iloc[0]

                # Calculate MAE
                mae_score = abs(point_pred - actual)
                mae_scores.append(mae_score)

            except Exception:
                continue

        if len(mae_scores) == 0:
            raise optuna.exceptions.TrialPruned()

        # Optimize MAE only
        avg_mae = np.mean(mae_scores)
        return avg_mae

    def train_location(self, train_df: pd.DataFrame, full_df: pd.DataFrame,
                      location: str, selected_states: List[str]) -> Dict:
        """Train SVM model for a specific location."""

        print(f"\n{'='*60}")
        print(f"Training SVM model for {location}")
        print(f"Cut-off date: {self.cut_off_date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")

        # Hyperparameter optimization with Optuna
        print(f"Starting hyperparameter optimization with {self.trials} trials...")

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.random_seed),
            study_name=f"svm_{location}"
        )

        study.optimize(
            lambda trial: self.objective(trial, train_df, location, selected_states),
            n_trials=self.trials,
            show_progress_bar=True
        )

        # Get best parameters - extract all SVM-relevant parameters
        kernel = study.best_params.get('kernel', 'rbf')
        best_params = {
            'kernel': kernel,
            'C': study.best_params['C'],
            'epsilon': study.best_params['epsilon']
        }

        # Add kernel-specific parameters
        if kernel == 'rbf':
            best_params['gamma'] = study.best_params.get('gamma', 'scale')
        elif kernel == 'sigmoid':
            best_params['gamma'] = study.best_params.get('gamma', 'scale')
            best_params['coef0'] = study.best_params.get('coef0', 0.0)

        if self.use_enhanced_features:
            best_lags = None
        else:
            n_recent = study.best_params.get("n_recent_lags", self.max_lags)
            use_seasonal = study.best_params.get("use_seasonal_lags", True)
            best_lags = list(range(1, n_recent + 1))
            if use_seasonal:
                best_lags.extend([52])

        print(f"Best parameters: {best_params}")
        print(f"Best lags: {best_lags}")
        print(f"Best CV score: {study.best_value:.6f}")

        # Collect dynamic residuals for probabilistic intervals (validation on second half)
        residual_info = self.collect_dynamic_residuals(
            train_df, location, selected_states, best_params, best_lags
        )

        # Train final model on all training data
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

        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train final model
        final_svm = SVR(**best_params)
        final_svm.fit(X_train_scaled, y_train)

        print(f"Training completed for {location}")

        return {
            'model': final_svm,
            'scaler': scaler,
            'best_params': best_params,
            'selected_states': selected_states,
            'lags': best_lags,
            'residuals': residual_info.get('residuals', []),
            'residual_scale': residual_info.get('scale', self.residual_scale),
            'point_bias': residual_info.get('point_bias', 0.0),
            'blend_alpha': residual_info.get('blend_alpha', 1.0),
            'cv_score': study.best_value,
            'location': location,
            'horizon': self.horizon,
            'use_log_transform': self.use_log_transform
        }

    def train_all_locations(self, data_file: str, locations: List[str]) -> None:
        """Train SVM models for all specified locations."""

        print(f"\n{'='*80}")
        print(f"SVM PROBABILISTIC FORECASTING PIPELINE")
        print(f"{'='*80}")
        print(f"Locations: {locations}")
        print(f"Cut-off date: {self.cut_off_date.strftime('%Y-%m-%d')}")
        print(f"Horizon: {self.horizon}")
        print(f"Trials: {self.trials}")

        # Load and prepare data
        train_df, full_df, location_features = self.load_and_prepare_data(data_file, locations)

        # Train models for each location
        for location in locations:
            selected_states = location_features[location]

            # Train model
            result = self.train_location(train_df, full_df, location, selected_states)
            self.model_results[location] = result
            # Backward compatibility: store uncertainty info if present
            if 'error_distribution' in result:
                self.error_distributions[location] = result['error_distribution']
            elif 'residuals' in result:
                # Store residuals under the same attribute for compatibility
                self.error_distributions[location] = {self.horizon: result['residuals']}

        print(f"\n{'='*80}")
        print(f"TRAINING COMPLETED FOR ALL LOCATIONS")
        print(f"{'='*80}")

    def save_models(self) -> None:
        """Save all trained models and parameters."""

        print("\nSaving trained models...")

        # Create output directory
        os.makedirs("models", exist_ok=True)

        for location in self.model_results.keys():
            # Save model data
            model_data = self.model_results[location]

            # Create filename with unique identifier
            identifier = f"svm_{location}_h{self.horizon}_t{self.trials}"
            model_file = f"models/{identifier}_model.pkl"

            # Save everything as pickle
            with open(model_file, 'wb') as f:
                pickle.dump(model_data, f)

            print(f"  Saved model for {location}: {model_file}")

        # Save combined hyperparameters for retrospective script
        combined_params = {}
        for location in self.model_results.keys():
            combined_params[location] = {
                'best_params': self.model_results[location]['best_params'],
                'selected_states': self.model_results[location]['selected_states'],
                'lags': self.model_results[location]['lags'],
                # New residual-based uncertainty
                'residuals': self.model_results[location].get('residuals', []),
                'residual_scale': self.model_results[location].get('residual_scale', self.residual_scale),
                'point_bias': self.model_results[location].get('point_bias', 0.0),
                'blend_alpha': self.model_results[location].get('blend_alpha', 1.0),
                'horizon': self.horizon,
                'use_log_transform': self.use_log_transform
            }

        # Include enhanced features flag in filename to avoid overwriting
        features_suffix = "_enhanced" if self.use_enhanced_features else ""
        params_file = f"models/svm_hyperparameters_h{self.horizon}_t{self.trials}{features_suffix}.pkl"
        with open(params_file, 'wb') as f:
            pickle.dump(combined_params, f)

        print(f"  Combined hyperparameters saved to: {params_file}")

    def print_summary(self) -> None:
        """Print training summary."""

        print(f"\n{'='*80}")
        print(f"TRAINING SUMMARY")
        print(f"{'='*80}")

        for location in self.model_results.keys():
            result = self.model_results[location]

            print(f"\n{location}:")
            print(f"  CV MAE: {result['cv_score']:.6f}")
            print(f"  Best kernel: {result['best_params']['kernel']}")
            print(f"  Best C: {result['best_params']['C']:.4f}")
            print(f"  Best epsilon: {result['best_params']['epsilon']:.4f}")

            # Print kernel-specific parameters
            kernel = result['best_params']['kernel']
            if kernel in ['rbf', 'sigmoid']:
                print(f"  Best gamma: {result['best_params']['gamma']}")
            if kernel == 'sigmoid':
                print(f"  Best coef0: {result['best_params'].get('coef0', 0.0):.4f}")

            print(f"  Lags: {result['lags']}")
            print(f"  Features: {result['selected_states']}")
            res = result.get('residuals', [])
            print(f"  Residual scale: {result.get('residual_scale', self.residual_scale)}")
            print(f"  Residual samples: {len(res)}")
            print(f"  Point bias (raw): {result.get('point_bias', 0.0):.3f}")
            print(f"  Blend alpha (svm vs persistence): {result.get('blend_alpha', 1.0):.2f}")
            print(f"  Ready for retrospective forecasting")


def main():
    """Main function for SVM training pipeline."""

    parser = argparse.ArgumentParser(description='SVM Probabilistic Forecasting Training Pipeline')

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
    parser.add_argument('--residual-scale', type=str, default='log1p', choices=['log1p', 'pct', 'none'],
                       help="Scale for residuals used to build prediction intervals: 'log1p' (default), 'pct', or 'none'")
    parser.add_argument('--kernels', type=str, default='rbf',
                       help="Comma-separated list of kernels to consider (default: 'rbf'). Options: rbf,linear,sigmoid")
    parser.add_argument('--cv-max-anchors', type=int, default=30,
                       help='Max validation anchors in objective/residuals (default: 30)')
    parser.add_argument('--seasonal-in-search', action='store_true',
                       help='Include seasonal lag 52 as a boolean hyperparameter (default off if not set)')

    # Optimization arguments
    parser.add_argument('--trials', type=int, default=100,
                       help='Number of Optuna trials (default: 100)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.data_file):
        raise FileNotFoundError(f"Data file not found: {args.data_file}")

    # Initialize trainer
    # Parse kernels list
    kernels = [k.strip() for k in args.kernels.split(',') if k.strip()]

    trainer = SVMTrainer(
        cut_off_date=args.cut_off,
        horizon=args.horizon,
        max_lags=args.max_lags,
        trials=args.trials,
        random_seed=args.random_seed,
        use_enhanced_features=args.use_enhanced_features,
        n_features=args.n_features,
        use_log_transform=args.use_log_transform,
        residual_scale=args.residual_scale,
        kernels=kernels,
        cv_max_anchors=args.cv_max_anchors,
        seasonal_in_search=args.seasonal_in_search
    )

    # Train models
    trainer.train_all_locations(args.data_file, args.locations)

    # Save results
    trainer.save_models()

    # Print summary
    trainer.print_summary()

    print(f"\n{'='*80}")
    print(f"READY FOR RETROSPECTIVE FORECASTING")
    print(f"{'='*80}")
    features_suffix = "_enhanced" if args.use_enhanced_features else ""
    print(f"Use: python src/generate_retrospective_forecasts.py \\")
    print(f"       --hyperparams models/svm_hyperparameters_h{args.horizon}_t{args.trials}{features_suffix}.pkl \\")
    print(f"       --data-file {args.data_file} \\")
    print(f"       --cut-off {args.cut_off}")


if __name__ == "__main__":
    main()
