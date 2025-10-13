"""
Stage 3.4: Advanced Ensemble (Out-of-Fold Stacking)
Smart Product Pricing Challenge

What this does:
1. Creates out-of-fold predictions from multiple models
2. Trains meta-learner on OOF predictions
3. Intelligent weighted blending
4. Generates final submission

Expected improvement: 12.0% → 11.5-12.0% SMAPE
Runtime: 30-60 minutes
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" STAGE 3.4: ADVANCED ENSEMBLE")
print("="*80)

# ============================================================================
# 0. CHECK DEPENDENCIES
# ============================================================================

print("\n[*] Checking dependencies...")
try:
    import lightgbm as lgb
    print("   [+] LightGBM available")
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    print("   [+] CatBoost available")
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    import xgboost as xgb
    print("   [+] XGBoost available")
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

if not (HAS_LGB or HAS_CAT or HAS_XGB):
    print("\n[!] ERROR: Need at least one model library!")
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

# ============================================================================
# 4. OUT-OF-FOLD PREDICTIONS
# ============================================================================

print("\n" + "="*80)
print(" GENERATING OUT-OF-FOLD PREDICTIONS")
print("="*80)

N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Load tuned hyperparameters if available
tuned_params_path = os.path.join(DATASET_FOLDER, 'tuned_hyperparameters.json')
if os.path.exists(tuned_params_path):
    print("\n[*] Loading tuned hyperparameters...")
    with open(tuned_params_path, 'r') as f:
        tuned_params = json.load(f)
else:
    tuned_params = {}

# Storage for OOF predictions
oof_predictions = {}
test_predictions = {}

# -------------------------
# 4.1 LIGHTGBM OOF
# -------------------------
if HAS_LGB:
    print(f"\n[*] LightGBM {N_FOLDS}-Fold OOF...")
    
    if 'lightgbm' in tuned_params:
        lgb_params = tuned_params['lightgbm']['best_params'].copy()
        lgb_params.update({
            'objective': 'regression',
            'metric': 'rmse',
            'verbosity': -1,
            'random_state': 42,
            'n_jobs': -1,
            'n_estimators': 2000
        })
    else:
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': 7,
            'learning_rate': 0.03,
            'n_estimators': 2000,
            'num_leaves': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
    
    oof_lgb = np.zeros(len(X))
    test_lgb = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"   Fold {fold+1}/{N_FOLDS}...", end=" ")
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y_log.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y_log.iloc[val_idx]
        
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            callbacks=[lgb.early_stopping(100, verbose=False)]
        )
        
        # OOF predictions
        oof_lgb[val_idx] = model.predict(X_val_fold)
        
        # Test predictions
        test_lgb += model.predict(X_test) / N_FOLDS
        
        fold_smape = calculate_smape(np.expm1(y_val_fold), np.expm1(oof_lgb[val_idx]))
        print(f"SMAPE: {fold_smape:.4f}%")
    
    # Convert to actual scale
    oof_lgb_actual = np.expm1(oof_lgb)
    test_lgb_actual = np.expm1(test_lgb)
    
    lgb_oof_smape = calculate_smape(y, oof_lgb_actual)
    print(f"\n[+] LightGBM OOF SMAPE: {lgb_oof_smape:.4f}%")
    
    oof_predictions['lightgbm'] = oof_lgb_actual
    test_predictions['lightgbm'] = test_lgb_actual

# -------------------------
# 4.2 CATBOOST OOF
# -------------------------
if HAS_CAT:
    print(f"\n[*] CatBoost {N_FOLDS}-Fold OOF...")
    
    # Check GPU availability
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("[+] GPU detected! Using GPU for CatBoost 🔥")
    except:
        use_gpu = False
    
    cat_features_idx = [model_features.index(col + '_encoded') for col in categorical_features]
    
    if 'catboost' in tuned_params:
        cat_params = tuned_params['catboost']['best_params'].copy()
        cat_params.update({
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': 0,
            'iterations': 2000,
            'early_stopping_rounds': 100,
            'task_type': 'GPU' if use_gpu else 'CPU'
        })
        if use_gpu:
            cat_params['devices'] = '0'
    else:
        cat_params = {
            'iterations': 2000,
            'learning_rate': 0.03,
            'depth': 7,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': 0,
            'early_stopping_rounds': 100,
            'task_type': 'GPU' if use_gpu else 'CPU',
            'loss_function': 'RMSE'
        }
        if use_gpu:
            cat_params['devices'] = '0'
    
    oof_cat = np.zeros(len(X))
    test_cat = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"   Fold {fold+1}/{N_FOLDS}...", end=" ")
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y_log.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y_log.iloc[val_idx]
        
        model = cb.CatBoostRegressor(**cat_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=(X_val_fold, y_val_fold),
            cat_features=cat_features_idx,
            verbose=False
        )
        
        # OOF predictions
        oof_cat[val_idx] = model.predict(X_val_fold)
        
        # Test predictions
        test_cat += model.predict(X_test) / N_FOLDS
        
        fold_smape = calculate_smape(np.expm1(y_val_fold), np.expm1(oof_cat[val_idx]))
        print(f"SMAPE: {fold_smape:.4f}%")
    
    # Convert to actual scale
    oof_cat_actual = np.expm1(oof_cat)
    test_cat_actual = np.expm1(test_cat)
    
    cat_oof_smape = calculate_smape(y, oof_cat_actual)
    print(f"\n[+] CatBoost OOF SMAPE: {cat_oof_smape:.4f}%")
    
    oof_predictions['catboost'] = oof_cat_actual
    test_predictions['catboost'] = test_cat_actual

# -------------------------
# 4.3 XGBOOST OOF
# -------------------------
if HAS_XGB:
    print(f"\n[*] XGBoost {N_FOLDS}-Fold OOF...")
    
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 7,
        'learning_rate': 0.03,
        'n_estimators': 2000,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    
    oof_xgb = np.zeros(len(X))
    test_xgb = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"   Fold {fold+1}/{N_FOLDS}...", end=" ")
        
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y_log.iloc[train_idx]
        X_val_fold = X.iloc[val_idx]
        y_val_fold = y_log.iloc[val_idx]
        
        model = xgb.XGBRegressor(**xgb_params)
        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            verbose=False
        )
        
        # OOF predictions
        oof_xgb[val_idx] = model.predict(X_val_fold)
        
        # Test predictions
        test_xgb += model.predict(X_test) / N_FOLDS
        
        fold_smape = calculate_smape(np.expm1(y_val_fold), np.expm1(oof_xgb[val_idx]))
        print(f"SMAPE: {fold_smape:.4f}%")
    
    # Convert to actual scale
    oof_xgb_actual = np.expm1(oof_xgb)
    test_xgb_actual = np.expm1(test_xgb)
    
    xgb_oof_smape = calculate_smape(y, oof_xgb_actual)
    print(f"\n[+] XGBoost OOF SMAPE: {xgb_oof_smape:.4f}%")
    
    oof_predictions['xgboost'] = oof_xgb_actual
    test_predictions['xgboost'] = test_xgb_actual

# ============================================================================
# 5. META-LEARNER (STACKING)
# ============================================================================

print("\n" + "="*80)
print(" TRAINING META-LEARNER")
print("="*80)

# Prepare stacking features
oof_stack = np.column_stack([oof_predictions[m] for m in oof_predictions.keys()])
test_stack = np.column_stack([test_predictions[m] for m in test_predictions.keys()])

print(f"\n[*] Stacking features shape: {oof_stack.shape}")
print(f"[*] Base models: {list(oof_predictions.keys())}")

# Train meta-learner
print("\n[*] Training Ridge meta-learner...")
meta_model = Ridge(alpha=1.0)
meta_model.fit(oof_stack, y)

# Predictions
stack_pred_oof = meta_model.predict(oof_stack)
stack_pred_test = meta_model.predict(test_stack)

stack_smape = calculate_smape(y, stack_pred_oof)

print(f"\n[+] Stacking SMAPE: {stack_smape:.4f}%")
print(f"\n[*] Meta-model weights:")
for i, model_name in enumerate(oof_predictions.keys()):
    print(f"   {model_name}: {meta_model.coef_[i]:.4f}")

# ============================================================================
# 6. WEIGHTED ENSEMBLE
# ============================================================================

print("\n" + "="*80)
print(" WEIGHTED ENSEMBLE")
print("="*80)

# Calculate weights based on inverse SMAPE
smape_scores = {}
for model_name in oof_predictions.keys():
    smape = calculate_smape(y, oof_predictions[model_name])
    smape_scores[model_name] = smape

print("\n[*] Individual model SMAPE:")
for model_name, smape in smape_scores.items():
    print(f"   {model_name}: {smape:.4f}%")

# Calculate inverse SMAPE weights
total_inverse = sum([1/s for s in smape_scores.values()])
weights = {m: (1/smape_scores[m])/total_inverse for m in smape_scores.keys()}

print(f"\n[*] Calculated weights (inverse SMAPE):")
for model_name, weight in weights.items():
    print(f"   {model_name}: {weight:.4f}")

# Weighted predictions
weighted_pred_oof = np.sum([oof_predictions[m] * weights[m] for m in oof_predictions.keys()], axis=0)
weighted_pred_test = np.sum([test_predictions[m] * weights[m] for m in test_predictions.keys()], axis=0)

weighted_smape = calculate_smape(y, weighted_pred_oof)
print(f"\n[+] Weighted Ensemble SMAPE: {weighted_smape:.4f}%")

# ============================================================================
# 7. SIMPLE AVERAGE (BASELINE)
# ============================================================================

avg_pred_oof = np.mean([oof_predictions[m] for m in oof_predictions.keys()], axis=0)
avg_pred_test = np.mean([test_predictions[m] for m in test_predictions.keys()], axis=0)

avg_smape = calculate_smape(y, avg_pred_oof)
print(f"[+] Simple Average SMAPE: {avg_smape:.4f}%")

# ============================================================================
# 8. SELECT BEST ENSEMBLE METHOD
# ============================================================================

print("\n" + "="*80)
print(" FINAL ENSEMBLE SELECTION")
print("="*80)

ensemble_methods = {
    'stacking': (stack_smape, stack_pred_test),
    'weighted': (weighted_smape, weighted_pred_test),
    'average': (avg_smape, avg_pred_test)
}

# Find best method
best_method = min(ensemble_methods.keys(), key=lambda m: ensemble_methods[m][0])
best_smape = ensemble_methods[best_method][0]
best_predictions = ensemble_methods[best_method][1]

print(f"\n[*] Ensemble comparison:")
for method, (smape, _) in ensemble_methods.items():
    marker = " [BEST]" if method == best_method else ""
    print(f"   {method}: {smape:.4f}%{marker}")

# ============================================================================
# 9. CREATE SUBMISSION
# ============================================================================

print("\n" + "="*80)
print(" CREATING FINAL SUBMISSION")
print("="*80)

submission = pd.DataFrame({
    'sample_id': test_ids,
    'price': best_predictions
})

submission['price'] = submission['price'].clip(lower=0.1)

submission_path = os.path.join(DATASET_FOLDER, 'test_out_final.csv')
submission.to_csv(submission_path, index=False)

print(f"\n[+] Final submission created: {submission_path}")
print(f"   Method: {best_method.capitalize()} Ensemble")
print(f"   Validation SMAPE: {best_smape:.4f}%")
print(f"\n[*] Submission statistics:")
print(f"   Min: ${submission['price'].min():.2f}")
print(f"   Max: ${submission['price'].max():.2f}")
print(f"   Mean: ${submission['price'].mean():.2f}")
print(f"   Median: ${submission['price'].median():.2f}")

print("\n" + "="*80)
print(" 🎉 ADVANCED ENSEMBLE COMPLETE! 🎉")
print("="*80)

print(f"\n[*] Final Results:")
print(f"   Best Method: {best_method.capitalize()}")
print(f"   Validation SMAPE: {best_smape:.4f}%")
print(f"   Target: < 12% (Top 10%)")
print(f"\n[*] Files created: {submission_path}")
print("\n[!] READY FOR SUBMISSION! 🚀")
print("\n")

