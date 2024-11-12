import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from datetime import datetime, timedelta
import warnings
import joblib
import os
from pathlib import Path
warnings.filterwarnings('ignore')

class CompleteWalmartForecaster:
    def __init__(self, n_trials=100, model_dir='models'):
        """
        Initialize the forecaster with specified parameters
        
        Args:
            n_trials (int): Number of optimization trials
            model_dir (str): Directory to save/load models
        """
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = RobustScaler()
        self.le_store = LabelEncoder()
        self.le_dept = LabelEncoder()
        self.le_type = LabelEncoder()
        self.feature_importance = None
        self.n_trials = n_trials
        self.best_params = {'xgb': None, 'lgb': None}
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        
    def save_models(self, prefix='walmart'):
        """Save all models and encoders"""
        joblib.dump(self.xgb_model, self.model_dir / f'{prefix}_xgb.pkl')
        joblib.dump(self.lgb_model, self.model_dir / f'{prefix}_lgb.pkl')
        joblib.dump(self.scaler, self.model_dir / f'{prefix}_scaler.pkl')
        joblib.dump(self.le_store, self.model_dir / f'{prefix}_le_store.pkl')
        joblib.dump(self.le_dept, self.model_dir / f'{prefix}_le_dept.pkl')
        joblib.dump(self.le_type, self.model_dir / f'{prefix}_le_type.pkl')
        
    def load_models(self, prefix='walmart'):
        """Load all models and encoders"""
        self.xgb_model = joblib.load(self.model_dir / f'{prefix}_xgb.pkl')
        self.lgb_model = joblib.load(self.model_dir / f'{prefix}_lgb.pkl')
        self.scaler = joblib.load(self.model_dir / f'{prefix}_scaler.pkl')
        self.le_store = joblib.load(self.model_dir / f'{prefix}_le_store.pkl')
        self.le_dept = joblib.load(self.model_dir / f'{prefix}_le_dept.pkl')
        self.le_type = joblib.load(self.model_dir / f'{prefix}_le_type.pkl')

    def create_store_features(self, df, stores_df):
        """Create store-specific features"""
        # Store performance metrics
        if 'Weekly_Sales' in df.columns:
            store_metrics = df.groupby('Store').agg({
                'Weekly_Sales': ['mean', 'std', 'median']
            }).reset_index()
            store_metrics.columns = ['Store', 'Store_Avg_Sales', 'Store_Std_Sales', 'Store_Median_Sales']
            df = df.merge(store_metrics, on='Store', how='left')
            
            dept_metrics = df.groupby(['Store', 'Dept']).agg({
                'Weekly_Sales': ['mean', 'std', 'median']
            }).reset_index()
            dept_metrics.columns = ['Store', 'Dept', 'Dept_Avg_Sales', 'Dept_Std_Sales', 'Dept_Median_Sales']
            df = df.merge(dept_metrics, on=['Store', 'Dept'], how='left')
        
        stores_df['Size_Category'] = pd.qcut(stores_df['Size'], q=5, labels=['XS', 'S', 'M', 'L', 'XL'])
        df = df.merge(stores_df[['Store', 'Size_Category']], on='Store', how='left')
        df = pd.get_dummies(df, columns=['Size_Category'], prefix='Size_Cat')
        
        return df

    def create_lag_features(self, df):
        """Create lag features for time series"""
        df = df.sort_values(['Store', 'Dept', 'Date'])
        
        lag_features = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        for feature in lag_features:
            if feature in df.columns:
                for lag in [1, 2, 4, 8]:
                    df[f'{feature}_lag_{lag}'] = df.groupby(['Store', 'Dept'])[feature].shift(lag)
                
                # Create rolling statistics
                for window in [2, 4, 8]:
                    rolling_mean = df.groupby(['Store', 'Dept'])[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).mean()
                    )
                    df[f'{feature}_rolling_mean_{window}'] = rolling_mean
                    
                    rolling_std = df.groupby(['Store', 'Dept'])[feature].transform(
                        lambda x: x.rolling(window=window, min_periods=1).std()
                    )
                    df[f'{feature}_rolling_std_{window}'] = rolling_std
                
                expanding_mean = df.groupby(['Store', 'Dept'])[feature].transform(
                    lambda x: x.expanding(min_periods=1).mean()
                )
                df[f'{feature}_expanding_mean'] = expanding_mean
                
                expanding_std = df.groupby(['Store', 'Dept'])[feature].transform(
                    lambda x: x.expanding(min_periods=1).std()
                )
                df[f'{feature}_expanding_std'] = expanding_std
        
        return df

    def create_seasonal_features(self, df):
        """Create seasonal features"""
        # Cyclical encoding
        df['Month_Sin'] = np.sin(2 * np.pi * df['Month']/12)
        df['Month_Cos'] = np.cos(2 * np.pi * df['Month']/12)
        df['Week_Sin'] = np.sin(2 * np.pi * df['Week']/52)
        df['Week_Cos'] = np.cos(2 * np.pi * df['Week']/52)
        
        # Seasons
        df['Season'] = pd.cut(df['Month'], 
                            bins=[0, 3, 6, 9, 12], 
                            labels=['Winter', 'Spring', 'Summer', 'Fall'])
        df = pd.get_dummies(df, columns=['Season'], prefix='Season')
        
        # Holiday seasons
        holiday_months = [11, 12, 1]  # November, December, January
        df['Is_Holiday_Season'] = df['Month'].isin(holiday_months).astype(int)
        
        # Back to school season (August-September)
        df['Is_BackToSchool'] = df['Month'].isin([8, 9]).astype(int)
        
        return df

    def create_interaction_features(self, df):
        """Create interaction features"""
        df['Size_per_Type'] = df.groupby('Type_Encoded')['Size_Normalized'].transform('mean')
        if 'Weekly_Sales' in df.columns:
            df['Sales_per_Size'] = df['Weekly_Sales'] / df['Size_Normalized']
        df['Holiday_Size_Interaction'] = df['IsHoliday'] * df['Size_Normalized']
        df['Temperature_Holiday_Interaction'] = df['Temperature'] * df['IsHoliday']
        df['CPI_Unemployment_Interaction'] = df['CPI'] * df['Unemployment']
        
        # Create day of week features
        df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.dayofweek
        df['Is_Weekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        return df

    def merge_all_data(self, sales_df, features_df, stores_df, test_df=None):
        """Merge all available data sources with enhanced feature creation"""
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        if sales_df is not None:
            sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        if test_df is not None:
            test_df['Date'] = pd.to_datetime(test_df['Date'])

        base_df = sales_df if sales_df is not None else test_df
        
        # Merge with features and stores data
        df = base_df.merge(features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
        df = df.merge(stores_df, on='Store', how='left')

        df = self.create_store_features(df, stores_df)
        
        df['Type_Encoded'] = self.le_type.fit_transform(df['Type'])
        df['Size_Normalized'] = df['Size'] / df['Size'].max()
        
        if 'Dept' in df.columns:
            if sales_df is not None:
                df['Dept_Encoded'] = self.le_dept.fit_transform(df['Dept'])
            else:
                df['Dept_Encoded'] = self.le_dept.transform(df['Dept'])
        
        return df

    def optimize_xgb(self, trial, X, y):
        """Optimize XGBoost hyperparameters with enhanced parameter space"""
        param = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'max_depth': trial.suggest_int('max_depth', 4, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'max_bin': trial.suggest_int('max_bin', 200, 500)
        }

        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(**param)
            model.fit(
                X_train, 
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(rmse)
            
        return np.mean(scores)
    
    def optimize_lgb(self, trial, X, y):
        """Optimize LightGBM hyperparameters with enhanced parameter space"""
        param = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'max_depth': trial.suggest_int('max_depth', 4, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
            'num_leaves': trial.suggest_int('num_leaves', 10, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 1e-8, 1.0, log=True),
            'max_bin': trial.suggest_int('max_bin', 200, 500)
        }
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMRegressor(**param)
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=50,
                     verbose=False)
            
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(rmse)
            
        return np.mean(scores)

    def prepare_features(self, df, is_training=True):
        """Prepare features with enhanced feature engineering"""
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['DayOfYear'] = df['Date'].dt.dayofyear
        
        holiday_dates = df[df['IsHoliday']]['Date'].unique()
        df['Days_To_Holiday'] = df['Date'].apply(
            lambda x: min((hd - x).days for hd in holiday_dates if hd >= x) 
            if any(hd >= x for hd in holiday_dates) else 365
        )
        df['Days_From_Holiday'] = df['Date'].apply(
            lambda x: min((x - hd).days for hd in holiday_dates if hd <= x) 
            if any(hd <= x for hd in holiday_dates) else 365
        )

        if is_training:
            df['Store_Encoded'] = self.le_store.fit_transform(df['Store'])
            if 'Dept' in df.columns:
                df['Dept_Encoded'] = self.le_dept.fit_transform(df['Dept'])
        else:
            df['Store_Encoded'] = self.le_store.transform(df['Store'])
            if 'Dept' in df.columns:
                df['Dept_Encoded'] = self.le_dept.transform(df['Dept'])
        
        df['IsHoliday'] = df['IsHoliday'].astype(int)
        
        df = self.create_seasonal_features(df)
        df = self.create_interaction_features(df)
        if is_training:
            df = self.create_lag_features(df)

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Weekly_Sales', 'Store', 'Dept']:
                df[col] = df[col].fillna(df[col].mean())
        
        return df

    def evaluate_predictions(self, y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'RelativeError': np.mean(np.abs(y_true - y_pred) / y_true) * 100
        }
        return metrics

    def train_and_predict(self, train_df, features_df, stores_df, test_df):
        """Train models with enhanced optimization and generate predictions"""
        print("Preparing data...")
        train_full = self.merge_all_data(train_df, features_df, stores_df)
        train_full = self.prepare_features(train_full)
        
        train_full = train_full.dropna()

        feature_cols = ['Store_Encoded', 'Dept_Encoded', 'Type_Encoded', 'Size_Normalized',
                    'IsHoliday', 'Year', 'Month', 'Week', 'DayOfYear', 
                    'Days_To_Holiday', 'Days_From_Holiday',
                    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
                    'Month_Sin', 'Month_Cos', 'Week_Sin', 'Week_Cos',
                    'Size_per_Type', 'Holiday_Size_Interaction', 
                    'Temperature_Holiday_Interaction', 'CPI_Unemployment_Interaction',
                    'Is_Holiday_Season', 'Is_BackToSchool', 'Is_Weekend'] + \
                    [col for col in train_full.columns if 'Season_' in col] + \
                    [col for col in train_full.columns if 'Size_Cat_' in col] + \
                    [col for col in train_full.columns if '_lag_' in col or '_rolling_' in col or '_expanding_' in col]
        
        feature_cols = [col for col in feature_cols if col in train_full.columns]
        
        X = train_full[feature_cols].values
        y = train_full['Weekly_Sales'].values
        
        X = self.scaler.fit_transform(X)
        
        print("Optimizing XGBoost...")
        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(lambda trial: self.optimize_xgb(trial, X, y), n_trials=self.n_trials)
        
        print("Optimizing LightGBM...")
        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(lambda trial: self.optimize_lgb(trial, X, y), n_trials=self.n_trials)
        
        print("Training final models...")
        self.xgb_model = xgb.XGBRegressor(**study_xgb.best_params)
        self.lgb_model = lgb.LGBMRegressor(**study_lgb.best_params)

        X_train, X_val = X[:-4000], X[-4000:]
        y_train, y_val = y[:-4000], y[-4000:]
        
        self.xgb_model.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        self.lgb_model.fit(
            X_train, 
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Save feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'xgb_importance': self.xgb_model.feature_importances_,
            'lgb_importance': self.lgb_model.feature_importances_
        })
        
        print("Generating predictions...")
        test_full = self.merge_all_data(None, features_df, stores_df, test_df)
        test_full = self.prepare_features(test_full, is_training=False)
        
        X_test = test_full[feature_cols].values
        X_test = self.scaler.transform(X_test)
        
        xgb_pred = self.xgb_model.predict(X_test)
        lgb_pred = self.lgb_model.predict(X_test)
        
        # Weighted ensemble based on optimization scores
        xgb_weight = 1 / study_xgb.best_value
        lgb_weight = 1 / study_lgb.best_value
        total_weight = xgb_weight + lgb_weight
        
        ensemble_pred = (xgb_pred * (xgb_weight / total_weight) + 
                        lgb_pred * (lgb_weight / total_weight))
        
        predictions_df = pd.DataFrame({
            'Store': test_df['Store'],
            'Dept': test_df['Dept'],
            'Date': test_df['Date'],
            'IsHoliday': test_df['IsHoliday'],
            'Prediction': ensemble_pred
        })
        
        self.save_models()
        
        return {
            'predictions': predictions_df,
            'xgb_best_params': study_xgb.best_params,
            'lgb_best_params': study_lgb.best_params,
            'xgb_best_score': study_xgb.best_value,
            'lgb_best_score': study_lgb.best_value,
            'feature_importance': self.feature_importance
        }

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    features_df = pd.read_csv('data/features.csv')
    stores_df = pd.read_csv('data/stores.csv')
    
    # Initialize and train forecaster
    forecaster = CompleteWalmartForecaster(n_trials=100)
    results = forecaster.train_and_predict(
        train_df, features_df, stores_df, test_df
    )
    
    # Print results
    print("\nBest Parameters:")
    print("XGBoost:", results['xgb_best_params'])
    print("LightGBM:", results['lgb_best_params'])
    
    print("\nBest Scores:")
    print("XGBoost RMSE:", results['xgb_best_score'])
    print("LightGBM RMSE:", results['lgb_best_score'])
    
    results['predictions'].to_csv('predictions.csv', index=False)
    
    results['feature_importance'].sort_values(
        by=['xgb_importance', 'lgb_importance'],
        ascending=False
    ).to_csv('feature_importance.csv', index=False)