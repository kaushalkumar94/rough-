"""
Stage 3.2: Add Text Features (TF-IDF)
Smart Product Pricing Challenge

What this does:
1. Extracts TF-IDF features from product descriptions
2. Combines with existing numerical features
3. Trains models on extended feature set
4. Creates improved predictions

Expected improvement: 13.5% → 13.0-13.5% SMAPE
Runtime: 30-60 minutes
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from scipy.sparse import hstack
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" STAGE 3.2: TEXT FEATURES (TF-IDF)")
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
# 2. EXTRACT TF-IDF FEATURES
# ============================================================================

print("\n" + "="*80)
print(" EXTRACTING TF-IDF FEATURES")
print("="*80)

# Check if clean_text column exists
if 'clean_text' in train.columns:
    text_col = 'clean_text'
elif 'item_name' in train.columns:
    text_col = 'item_name'
else:
    print("[!] ERROR: No text column found!")
    exit(1)

print(f"\n[*] Using text column: {text_col}")
print(f"[*] Sample text: {train[text_col].iloc[0][:100]}...")

# Create TF-IDF vectorizer
print("\n[*] Creating TF-IDF features...")
print("   Parameters:")
print("   - max_features: 200 (top 200 most important words)")
print("   - ngram_range: (1, 3) (1-word, 2-word, 3-word combinations)")
print("   - min_df: 5 (word must appear in at least 5 documents)")
print("   - max_df: 0.8 (exclude words in >80% of documents)")

tfidf = TfidfVectorizer(
    max_features=200,  # Top 200 features
    ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
    min_df=5,  # Minimum document frequency
    max_df=0.8,  # Maximum document frequency
    lowercase=True,
    strip_accents='unicode',
    stop_words='english'
)

# Fit on combined train+test to ensure consistency
all_text = pd.concat([train[text_col], test[text_col]]).fillna('')
tfidf.fit(all_text)

# Transform train and test
train_tfidf = tfidf.transform(train[text_col].fillna(''))
test_tfidf = tfidf.transform(test[text_col].fillna(''))

print(f"\n[+] TF-IDF features created:")
print(f"   Train shape: {train_tfidf.shape}")
print(f"   Test shape: {test_tfidf.shape}")
print(f"\n[*] Sample top features:")
feature_names = tfidf.get_feature_names_out()
for i, feat in enumerate(feature_names[:20]):
    print(f"   {i+1}. {feat}")

# ============================================================================
# 3. PREPARE COMBINED FEATURES
# ============================================================================

print("\n" + "="*80)
print(" COMBINING FEATURES")
print("="*80)

# Encode categorical features
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

# Original features
X_train_orig = train[model_features].values
X_test_orig = test[model_features].values

# Combined features (numerical + TF-IDF)
X_train_combined = np.hstack([X_train_orig, train_tfidf.toarray()])
X_test_combined = np.hstack([X_test_orig, test_tfidf.toarray()])

y = train[target]
y_log = train[target_log]
test_ids = test['sample_id']

print(f"\n[*] Feature dimensions:")
print(f"   Original features: {X_train_orig.shape[1]}")
print(f"   TF-IDF features: {train_tfidf.shape[1]}")
print(f"   Combined features: {X_train_combined.shape[1]}")

# ============================================================================
# 4. TRAIN/VALIDATION SPLIT
# ============================================================================

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X_train_combined, y_log, test_size=0.2, random_state=42
)

y_train = np.expm1(y_train_log)
y_val = np.expm1(y_val_log)

print(f"\n[*] Split: {len(X_train):,} train, {len(X_val):,} validation")

# ============================================================================
# 5. SMAPE METRIC
# ============================================================================

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape

# ============================================================================
# 6. TRAIN MODELS
# ============================================================================

predictions_val = {}
predictions_test = {}

# Load tuned hyperparameters if available
tuned_params_path = os.path.join(DATASET_FOLDER, 'tuned_hyperparameters.json')
if os.path.exists(tuned_params_path):
    print("\n[*] Loading tuned hyperparameters...")
    with open(tuned_params_path, 'r') as f:
        tuned_params = json.load(f)
    print("   [+] Tuned parameters loaded")
else:
    print("\n[*] Using default parameters (run 07_hyperparameter_tuning.py first for better results)")
    tuned_params = {}

# -------------------------
# 6.1 LIGHTGBM
# -------------------------
if HAS_LGB:
    print("\n" + "="*80)
    print(" TRAINING LIGHTGBM WITH TEXT FEATURES")
    print("="*80)
    
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
        print("[*] Using tuned hyperparameters")
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
    
    print("[*] Training LightGBM...")
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    
    lgb_pred_val = np.expm1(lgb_model.predict(X_val))
    lgb_pred_test = np.expm1(lgb_model.predict(X_test_combined))
    
    lgb_smape = calculate_smape(y_val, lgb_pred_val)
    print(f"\n[+] LightGBM + Text SMAPE: {lgb_smape:.4f}%")
    
    predictions_val['lightgbm_text'] = lgb_pred_val
    predictions_test['lightgbm_text'] = lgb_pred_test

# -------------------------
# 6.2 CATBOOST
# -------------------------
if HAS_CAT:
    print("\n" + "="*80)
    print(" TRAINING CATBOOST WITH TEXT FEATURES")
    print("="*80)
    
    # Check GPU availability
    try:
        import torch
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            print("[+] GPU detected! Using GPU for CatBoost")
    except:
        use_gpu = False
    
    if 'catboost' in tuned_params:
        cat_params = tuned_params['catboost']['best_params'].copy()
        cat_params.update({
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': 200,
            'iterations': 2000,
            'early_stopping_rounds': 100,
            'task_type': 'GPU' if use_gpu else 'CPU'
        })
        print("[*] Using tuned hyperparameters")
    else:
        cat_params = {
            'iterations': 2000,
            'learning_rate': 0.03,
            'depth': 7,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': 200,
            'early_stopping_rounds': 100,
            'task_type': 'GPU' if use_gpu else 'CPU',
            'loss_function': 'RMSE'
        }
    
    print("[*] Training CatBoost...")
    cat_model = cb.CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train, y_train_log,
        eval_set=(X_val, y_val_log),
        verbose=200
    )
    
    cat_pred_val = np.expm1(cat_model.predict(X_val))
    cat_pred_test = np.expm1(cat_model.predict(X_test_combined))
    
    cat_smape = calculate_smape(y_val, cat_pred_val)
    print(f"\n[+] CatBoost + Text SMAPE: {cat_smape:.4f}%")
    
    predictions_val['catboost_text'] = cat_pred_val
    predictions_test['catboost_text'] = cat_pred_test

# ============================================================================
# 7. ENSEMBLE & CREATE SUBMISSION
# ============================================================================

print("\n" + "="*80)
print(" MODEL COMPARISON")
print("="*80)

print(f"\n[*] Validation SMAPE Scores:")
for model_name in predictions_val.keys():
    smape = calculate_smape(y_val, predictions_val[model_name])
    print(f"   {model_name}: {smape:.4f}%")

# Simple average ensemble
ensemble_pred_val = np.mean([predictions_val[m] for m in predictions_val.keys()], axis=0)
ensemble_pred_test = np.mean([predictions_test[m] for m in predictions_test.keys()], axis=0)

ensemble_smape = calculate_smape(y_val, ensemble_pred_val)
print(f"   Ensemble: {ensemble_smape:.4f}% [BEST]")

# Create submission
submission = pd.DataFrame({
    'sample_id': test_ids,
    'price': ensemble_pred_test
})

submission['price'] = submission['price'].clip(lower=0.1)

submission_path = os.path.join(DATASET_FOLDER, 'test_out_text.csv')
submission.to_csv(submission_path, index=False)

print("\n" + "="*80)
print(" SUBMISSION CREATED")
print("="*80)

print(f"\n[+] Submission file: {submission_path}")
print(f"   Validation SMAPE: {ensemble_smape:.4f}%")
print(f"\n[*] Submission statistics:")
print(f"   Min: ${submission['price'].min():.2f}")
print(f"   Max: ${submission['price'].max():.2f}")
print(f"   Mean: ${submission['price'].mean():.2f}")
print(f"   Median: ${submission['price'].median():.2f}")

print("\n" + "="*80)
print(" TEXT FEATURES COMPLETE!")
print("="*80)

print(f"\n[*] Expected improvement: 13.5% → 13.0-13.5%")
print(f"[*] Files created: {submission_path}")
print("\n[!] NEXT STEP: Share the validation SMAPE for analysis")
print("\n")

