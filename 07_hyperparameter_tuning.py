"""
Stage 3.1: Hyperparameter Tuning with Optuna
Smart Product Pricing Challenge

What this does:
1. Uses Optuna for Bayesian optimization
2. Tunes LightGBM and CatBoost hyperparameters
3. Uses 5-fold cross-validation
4. Saves best models and parameters

Expected improvement: 14.57% → 13.5-14.0% SMAPE
Runtime: 1-2 hours (depends on n_trials)
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" STAGE 3.1: HYPERPARAMETER TUNING")
print("="*80)

# ============================================================================
# 0. CHECK DEPENDENCIES
# ============================================================================

print("\n[*] Checking dependencies...")
try:
    import optuna
    print("   [+] Optuna available")
    HAS_OPTUNA = True
except ImportError:
    print("   [!] Optuna not found. Install: pip install optuna")
    print("   [!] Falling back to manual tuning...")
    HAS_OPTUNA = False

try:
    import lightgbm as lgb
    print("   [+] LightGBM available")
    HAS_LGB = True
except ImportError:
    print("   [!] LightGBM not found")
    HAS_LGB = False

try:
    import catboost as cb
    print("   [+] CatBoost available")
    HAS_CAT = True
except ImportError:
    print("   [!] CatBoost not found")
    HAS_CAT = False

if not (HAS_LGB or HAS_CAT):
    print("\n[!] ERROR: Need at least one model library installed!")
    exit(1)

# ============================================================================
# 1. LOAD DATA & CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(SCRIPT_DIR, '..', 'dataset')

print("\n[*] Loading model-ready datasets...")
train = pd.read_csv(os.path.join(DATASET_FOLDER, 'train_model_ready.csv'))
test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test_model_ready.csv'))

print(f"   [+] Training: {len(train):,} samples")
print(f"   [+] Test: {len(test):,} samples")

# Load feature configuration
config_path = os.path.join(DATASET_FOLDER, 'feature_config.json')
with open(config_path, 'r') as f:
    feature_config = json.load(f)

numerical_features = feature_config['numerical_features']
categorical_features = feature_config['categorical_features']
target = feature_config['target']
target_log = feature_config['target_log']

# ============================================================================
# 2. PREPARE FEATURES
# ============================================================================

print("\n[*] Encoding categorical features...")
label_encoders = {}

for col in categorical_features:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]]).astype(str)
    le.fit(combined)
    
    train[col + '_encoded'] = le.transform(train[col].astype(str))
    test[col + '_encoded'] = le.transform(test[col].astype(str))
    
    label_encoders[col] = le

categorical_encoded = [col + '_encoded' for col in categorical_features]
model_features = numerical_features + categorical_encoded

X = train[model_features]
y = train[target]
y_log = train[target_log]
X_test = test[model_features]
test_ids = test['sample_id']

# ============================================================================
# 3. SMAPE METRIC
# ============================================================================

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape

def smape_cv(y_log_true, y_log_pred):
    """SMAPE for cross-validation (log scale)"""
    y_true = np.expm1(y_log_true)
    y_pred = np.expm1(y_log_pred)
    return calculate_smape(y_true, y_pred)

# ============================================================================
# 4. OPTUNA OPTIMIZATION (LIGHTGBM)
# ============================================================================

if HAS_LGB and HAS_OPTUNA:
    print("\n" + "="*80)
    print(" TUNING LIGHTGBM (5-FOLD CV)")
    print("="*80)
    
    def objective_lgb(trial):
        """Optuna objective for LightGBM"""
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1,
            
            # Hyperparameters to tune
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 15, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'n_estimators': 2000
        }
        
        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y_log.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y_log.iloc[val_idx]
            
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            
            # Predict and calculate SMAPE
            y_pred_log = model.predict(X_val_fold)
            fold_smape = smape_cv(y_val_fold, y_pred_log)
            cv_scores.append(fold_smape)
        
        return np.mean(cv_scores)
    
    # Run optimization
    print("\n[*] Starting Optuna optimization (50 trials)...")
    print("[*] This will take 30-60 minutes depending on your hardware")
    print("[*] You can stop early with Ctrl+C if needed\n")
    
    study_lgb = optuna.create_study(direction='minimize', study_name='lgb_tuning')
    study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True)
    
    print(f"\n[+] Best LightGBM SMAPE: {study_lgb.best_value:.4f}%")
    print(f"[+] Best parameters:")
    for key, value in study_lgb.best_params.items():
        print(f"   {key}: {value}")
    
    # Save best parameters
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'random_state': 42,
        'n_jobs': -1,
        'n_estimators': 2000
    })

# ============================================================================
# 5. OPTUNA OPTIMIZATION (CATBOOST)
# ============================================================================

if HAS_CAT and HAS_OPTUNA:
    print("\n" + "="*80)
    print(" TUNING CATBOOST (5-FOLD CV)")
    print("="*80)
    
    cat_features_idx = [model_features.index(col + '_encoded') for col in categorical_features]
    
    def objective_cat(trial):
        """Optuna objective for CatBoost"""
        # Try to use GPU by default
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except:
            use_gpu = False
        
        params = {
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': 0,
            'task_type': 'GPU' if use_gpu else 'CPU',
            
            # Hyperparameters to tune
            'iterations': 2000,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'depth': trial.suggest_int('depth', 3, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
            'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'early_stopping_rounds': 100
        }
        
        # 5-fold cross-validation
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y_log.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y_log.iloc[val_idx]
            
            model = cb.CatBoostRegressor(**params)
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=(X_val_fold, y_val_fold),
                cat_features=cat_features_idx,
                verbose=False
            )
            
            # Predict and calculate SMAPE
            y_pred_log = model.predict(X_val_fold)
            fold_smape = smape_cv(y_val_fold, y_pred_log)
            cv_scores.append(fold_smape)
        
        return np.mean(cv_scores)
    
    # Run optimization
    print("\n[*] Starting Optuna optimization (50 trials)...")
    print("[*] This will take 30-60 minutes depending on your hardware")
    print("[*] GPU acceleration will be tested automatically\n")
    
    study_cat = optuna.create_study(direction='minimize', study_name='cat_tuning')
    study_cat.optimize(objective_cat, n_trials=50, show_progress_bar=True)
    
    print(f"\n[+] Best CatBoost SMAPE: {study_cat.best_value:.4f}%")
    print(f"[+] Best parameters:")
    for key, value in study_cat.best_params.items():
        print(f"   {key}: {value}")
    
    # Save best parameters
    best_cat_params = study_cat.best_params
    # Check GPU availability
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("\n[+] GPU detected! Using GPU for CatBoost training")
    except:
        use_gpu = False
        print("\n[!] GPU not detected, using CPU")
    
    best_cat_params.update({
        'loss_function': 'RMSE',
        'random_seed': 42,
        'verbose': 200,
        'task_type': 'GPU' if use_gpu else 'CPU',
        'iterations': 2000,
        'early_stopping_rounds': 100
    })

# ============================================================================
# 6. TRAIN FINAL MODELS WITH BEST PARAMS
# ============================================================================

print("\n" + "="*80)
print(" TRAINING FINAL MODELS")
print("="*80)

final_predictions = {}

# LightGBM
if HAS_LGB and HAS_OPTUNA:
    print("\n[*] Training final LightGBM with best parameters...")
    lgb_model = lgb.LGBMRegressor(**best_lgb_params)
    
    # Train on full training data
    lgb_model.fit(X, y_log)
    
    # Predictions
    lgb_pred_test = np.expm1(lgb_model.predict(X_test))
    final_predictions['lightgbm_tuned'] = lgb_pred_test
    
    print(f"[+] LightGBM trained successfully")

# CatBoost
if HAS_CAT and HAS_OPTUNA:
    print("\n[*] Training final CatBoost with best parameters...")
    cat_model = cb.CatBoostRegressor(**best_cat_params)
    
    # Train on full training data
    cat_model.fit(X, y_log, cat_features=cat_features_idx)
    
    # Predictions
    cat_pred_test = np.expm1(cat_model.predict(X_test))
    final_predictions['catboost_tuned'] = cat_pred_test
    
    print(f"[+] CatBoost trained successfully")

# ============================================================================
# 7. ENSEMBLE & CREATE SUBMISSION
# ============================================================================

print("\n" + "="*80)
print(" CREATING SUBMISSION")
print("="*80)

if len(final_predictions) > 1:
    # Weighted average (can be tuned further)
    weights = [0.5, 0.5]  # Equal weights
    final_pred = np.average(list(final_predictions.values()), axis=0, weights=weights)
    method = "Tuned Ensemble"
elif len(final_predictions) == 1:
    final_pred = list(final_predictions.values())[0]
    method = list(final_predictions.keys())[0]
else:
    print("[!] ERROR: No predictions generated!")
    exit(1)

submission = pd.DataFrame({
    'sample_id': test_ids,
    'price': final_pred
})

submission['price'] = submission['price'].clip(lower=0.1)

submission_path = os.path.join(DATASET_FOLDER, 'test_out_tuned.csv')
submission.to_csv(submission_path, index=False)

print(f"\n[+] Submission created: {submission_path}")
print(f"   Method: {method}")
print(f"\n[*] Submission statistics:")
print(f"   Min: ${submission['price'].min():.2f}")
print(f"   Max: ${submission['price'].max():.2f}")
print(f"   Mean: ${submission['price'].mean():.2f}")
print(f"   Median: ${submission['price'].median():.2f}")

# ============================================================================
# 8. SAVE BEST PARAMETERS
# ============================================================================

if HAS_OPTUNA:
    params_output = {}
    
    if HAS_LGB:
        params_output['lightgbm'] = {
            'best_score': study_lgb.best_value,
            'best_params': study_lgb.best_params
        }
    
    if HAS_CAT:
        params_output['catboost'] = {
            'best_score': study_cat.best_value,
            'best_params': study_cat.best_params
        }
    
    params_path = os.path.join(DATASET_FOLDER, 'tuned_hyperparameters.json')
    with open(params_path, 'w') as f:
        json.dump(params_output, f, indent=2)
    
    print(f"\n[+] Best parameters saved: {params_path}")

print("\n" + "="*80)
print(" TUNING COMPLETE!")
print("="*80)

if HAS_OPTUNA:
    print("\n[*] Results:")
    if HAS_LGB:
        print(f"   LightGBM CV SMAPE: {study_lgb.best_value:.4f}%")
    if HAS_CAT:
        print(f"   CatBoost CV SMAPE: {study_cat.best_value:.4f}%")
    print(f"\n[*] Expected improvement: 14.57% → 13.5-14.0%")

print(f"\n[*] Files created:")
print(f"   {submission_path}")
if HAS_OPTUNA:
    print(f"   {params_path}")

print("\n[!] NEXT STEP: Share the tuned_hyperparameters.json back for analysis")
print("\n")

