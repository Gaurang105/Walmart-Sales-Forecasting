import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CompleteWalmartForecaster:
    def __init__(self, n_trials=50):
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = StandardScaler()
        self.le_store = LabelEncoder()
        self.le_dept = LabelEncoder()
        self.le_type = LabelEncoder()
        self.feature_importance = None
        self.n_trials = n_trials
        self.best_params = {'xgb': None, 'lgb': None}
        
    def optimize_xgb(self, trial, X, y):
        """Optimize XGBoost hyperparameters"""
        param = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }

        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = xgb.XGBRegressor(**param)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(rmse)
            
        return np.mean(scores)

    def optimize_lgb(self, trial, X, y):
        """Optimize LightGBM hyperparameters"""
        param = {
            'objective': 'regression',
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        }
        
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model = lgb.LGBMRegressor(**param)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            scores.append(rmse)
            
        return np.mean(scores)

    def evaluate_predictions(self, y_true, y_pred):
        """Calculate multiple evaluation metrics"""
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        return metrics

    def merge_all_data(self, sales_df, features_df, stores_df, test_df=None):
        """Merge all available data sources"""
        features_df['Date'] = pd.to_datetime(features_df['Date'])
        if sales_df is not None:
            sales_df['Date'] = pd.to_datetime(sales_df['Date'])
        if test_df is not None:
            test_df['Date'] = pd.to_datetime(test_df['Date'])

        base_df = sales_df if sales_df is not None else test_df
        
        # Merge with features and stores data
        df = base_df.merge(features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
        df = df.merge(stores_df, on='Store', how='left')
        
        # Encode store type and normalize size
        df['Type_Encoded'] = self.le_type.fit_transform(df['Type'])
        df['Size_Normalized'] = df['Size'] / df['Size'].max()
        
        if 'Dept' in df.columns:
            if sales_df is not None:
                df['Dept_Encoded'] = self.le_dept.fit_transform(df['Dept'])
            else:  # Testing phase
                df['Dept_Encoded'] = self.le_dept.transform(df['Dept'])
        
        return df

    def prepare_features(self, df, is_training=True):
        """Prepare features with time-based calculations"""
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.isocalendar().week
        df['DayOfYear'] = df['Date'].dt.dayofyear

        holiday_dates = df[df['IsHoliday']]['Date'].unique()
        df['Days_To_Holiday'] = df['Date'].apply(
            lambda x: min((hd - x).days for hd in holiday_dates if hd >= x) 
            if any(hd >= x for hd in holiday_dates) else 365
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
        
        return df

    def train_and_predict(self, train_df, features_df, stores_df, test_df):
        """Train models with optimization and generate predictions"""
        print("Preparing data...")
        train_full = self.merge_all_data(train_df, features_df, stores_df)
        train_full = self.prepare_features(train_full)
        
        feature_cols = ['Store_Encoded', 'Dept_Encoded', 'Type_Encoded', 'Size_Normalized',
                    'IsHoliday', 'Year', 'Month', 'Week', 'DayOfYear', 'Days_To_Holiday',
                    'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        
        X = train_full[feature_cols].values
        y = train_full['Weekly_Sales'].values
        
        print("Optimizing XGBoost...")
        study_xgb = optuna.create_study(direction='minimize')
        study_xgb.optimize(lambda trial: self.optimize_xgb(trial, X, y), n_trials=self.n_trials)
        
        print("Optimizing LightGBM...")
        study_lgb = optuna.create_study(direction='minimize')
        study_lgb.optimize(lambda trial: self.optimize_lgb(trial, X, y), n_trials=self.n_trials)
        
        print("Training final models...")
        self.xgb_model = xgb.XGBRegressor(**study_xgb.best_params)
        self.lgb_model = lgb.LGBMRegressor(**study_lgb.best_params)
        
        self.xgb_model.fit(X, y)
        self.lgb_model.fit(X, y)
        
        print("Generating predictions...")
        test_full = self.merge_all_data(None, features_df, stores_df, test_df)
        test_full = self.prepare_features(test_full, is_training=False)
        
        X_test = test_full[feature_cols].values
        
        xgb_pred = self.xgb_model.predict(X_test)
        lgb_pred = self.lgb_model.predict(X_test)
        ensemble_pred = (xgb_pred + lgb_pred) / 2
        
        predictions_df = pd.DataFrame({
            'Store': test_df['Store'],
            'Dept': test_df['Dept'],
            'Date': test_df['Date'],
            'IsHoliday': test_df['IsHoliday'],
            'Prediction': ensemble_pred
        })
        
        return {
            'predictions': predictions_df,
            'xgb_best_params': study_xgb.best_params,
            'lgb_best_params': study_lgb.best_params,
            'xgb_best_score': study_xgb.best_value,
            'lgb_best_score': study_lgb.best_value
        }



if __name__ == "__main__":
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    features_df = pd.read_csv('data/features.csv')
    stores_df = pd.read_csv('data/stores.csv')
    
    forecaster = CompleteWalmartForecaster(n_trials=50)
    results = forecaster.train_and_predict(
        train_df, features_df, stores_df, test_df
    )
    
    print("\nBest Parameters:")
    print("XGBoost:", results['xgb_best_params'])
    print("LightGBM:", results['lgb_best_params'])
    
    print("\nBest Scores:")
    print("XGBoost RMSE:", results['xgb_best_score'])
    print("LightGBM RMSE:", results['lgb_best_score'])
    

    results['predictions'].to_csv('walmart_predictions.csv', index=False)