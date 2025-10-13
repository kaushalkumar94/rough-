"""
Stage 3.3: Image Features Extraction (GPU ACCELERATED)
Smart Product Pricing Challenge

What this does:
1. Downloads product images from URLs
2. Extracts deep features using pre-trained CNN (ResNet50)
3. Combines with numerical + text features
4. Trains multi-modal models

Expected improvement: 13.0% → 12.0-12.5% SMAPE
Runtime: 2-4 hours (with GPU: ~30-45 min, without GPU: 3-4 hours)
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print(" STAGE 3.3: IMAGE FEATURES (GPU ACCELERATED)")
print("="*80)

# ============================================================================
# 0. CHECK DEPENDENCIES
# ============================================================================

print("\n[*] Checking dependencies...")

# Check PyTorch and torchvision
try:
    import torch
    import torchvision
    from torchvision import models, transforms
    from PIL import Image
    print(f"   [+] PyTorch {torch.__version__} available")
    print(f"   [+] Torchvision available")
    HAS_TORCH = True
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"   [+] GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   [+] CUDA version: {torch.version.cuda}")
        device = torch.device('cuda')
        print(f"   [+] Using GPU for image processing! 🔥")
    else:
        print("   [!] GPU not available, using CPU (will be slower)")
        device = torch.device('cpu')
        print("   [!] WARNING: Image processing will take 3-4 hours without GPU!")
        
except ImportError:
    print("   [!] PyTorch not found. Install: pip install torch torchvision")
    print("   [!] Skipping image features...")
    HAS_TORCH = False
    exit(1)

# Check requests for downloading images
try:
    import requests
    from io import BytesIO
    print("   [+] Requests library available")
    HAS_REQUESTS = True
except ImportError:
    print("   [!] Requests not found. Install: pip install requests")
    HAS_REQUESTS = False
    exit(1)

# Check model libraries
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

# ============================================================================
# 1. LOAD DATA & CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(SCRIPT_DIR, '..', 'dataset')
IMAGE_CACHE_FOLDER = os.path.join(DATASET_FOLDER, 'image_features_cache')

# Create cache folder
os.makedirs(IMAGE_CACHE_FOLDER, exist_ok=True)

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

# Check if image_link column exists
if 'image_link' not in train.columns:
    print("[!] ERROR: image_link column not found!")
    print("[!] Make sure original train.csv has image_link column")
    exit(1)

# ============================================================================
# 2. SETUP IMAGE FEATURE EXTRACTOR
# ============================================================================

print("\n" + "="*80)
print(" SETTING UP CNN FEATURE EXTRACTOR")
print("="*80)

print("\n[*] Loading pre-trained ResNet50 model...")
# Load pre-trained ResNet50
resnet_model = models.resnet50(pretrained=True)

# Remove final classification layer (we want features, not classes)
resnet_model = torch.nn.Sequential(*list(resnet_model.children())[:-1])
resnet_model = resnet_model.to(device)
resnet_model.eval()

print(f"[+] ResNet50 loaded on {device}")
print(f"[+] Output feature dimension: 2048")

# Image preprocessing
image_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ============================================================================
# 3. IMAGE DOWNLOAD & FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def download_image(url, timeout=10):
    """Download image from URL"""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content)).convert('RGB')
            return img
    except Exception as e:
        pass
    return None

def extract_image_features(img):
    """Extract features from image using ResNet50"""
    try:
        # Preprocess
        img_tensor = image_transform(img).unsqueeze(0).to(device)
        
        # Extract features
        with torch.no_grad():
            features = resnet_model(img_tensor)
        
        # Convert to numpy
        features = features.cpu().numpy().flatten()
        return features
    except Exception as e:
        return None

def create_default_features():
    """Create zero features for failed downloads"""
    return np.zeros(2048)

# ============================================================================
# 4. EXTRACT IMAGE FEATURES
# ============================================================================

print("\n" + "="*80)
print(" EXTRACTING IMAGE FEATURES")
print("="*80)

# Check if cached features exist
train_cache_path = os.path.join(IMAGE_CACHE_FOLDER, 'train_image_features.npy')
test_cache_path = os.path.join(IMAGE_CACHE_FOLDER, 'test_image_features.npy')

if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
    print("\n[*] Loading cached image features...")
    train_img_features = np.load(train_cache_path)
    test_img_features = np.load(test_cache_path)
    print(f"   [+] Train features: {train_img_features.shape}")
    print(f"   [+] Test features: {test_img_features.shape}")
    
else:
    print("\n[*] Extracting image features (this will take 30-60 min)...")
    print("[*] Progress will be shown every 1000 images")
    
    # Train images
    print("\n[*] Processing training images...")
    train_img_features = []
    failed_train = 0
    
    for idx, url in enumerate(train['image_link']):
        if idx % 1000 == 0:
            print(f"   [{idx}/{len(train)}] Downloaded, {failed_train} failed")
        
        img = download_image(url)
        if img is not None:
            features = extract_image_features(img)
            if features is not None:
                train_img_features.append(features)
            else:
                train_img_features.append(create_default_features())
                failed_train += 1
        else:
            train_img_features.append(create_default_features())
            failed_train += 1
    
    train_img_features = np.array(train_img_features)
    print(f"\n[+] Training images complete: {train_img_features.shape}")
    print(f"   Failed/skipped: {failed_train} ({failed_train/len(train)*100:.2f}%)")
    
    # Test images
    print("\n[*] Processing test images...")
    test_img_features = []
    failed_test = 0
    
    for idx, url in enumerate(test['image_link']):
        if idx % 1000 == 0:
            print(f"   [{idx}/{len(test)}] Downloaded, {failed_test} failed")
        
        img = download_image(url)
        if img is not None:
            features = extract_image_features(img)
            if features is not None:
                test_img_features.append(features)
            else:
                test_img_features.append(create_default_features())
                failed_test += 1
        else:
            test_img_features.append(create_default_features())
            failed_test += 1
    
    test_img_features = np.array(test_img_features)
    print(f"\n[+] Test images complete: {test_img_features.shape}")
    print(f"   Failed/skipped: {failed_test} ({failed_test/len(test)*100:.2f}%)")
    
    # Save cache
    print("\n[*] Saving features to cache...")
    np.save(train_cache_path, train_img_features)
    np.save(test_cache_path, test_img_features)
    print(f"   [+] Cache saved")

# ============================================================================
# 5. COMBINE ALL FEATURES
# ============================================================================

print("\n" + "="*80)
print(" COMBINING ALL FEATURES")
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

# Get numerical features
X_train_num = train[model_features].values
X_test_num = test[model_features].values

# Combine: Numerical + Image features
X_train_combined = np.hstack([X_train_num, train_img_features])
X_test_combined = np.hstack([X_test_num, test_img_features])

y = train[target]
y_log = train[target_log]
test_ids = test['sample_id']

print(f"\n[*] Multi-modal feature dimensions:")
print(f"   Numerical features: {X_train_num.shape[1]}")
print(f"   Image features: {train_img_features.shape[1]}")
print(f"   Combined features: {X_train_combined.shape[1]}")

# ============================================================================
# 6. TRAIN/VALIDATION SPLIT
# ============================================================================

X_train, X_val, y_train_log, y_val_log = train_test_split(
    X_train_combined, y_log, test_size=0.2, random_state=42
)

y_train = np.expm1(y_train_log)
y_val = np.expm1(y_val_log)

print(f"\n[*] Split: {len(X_train):,} train, {len(X_val):,} validation")

# ============================================================================
# 7. SMAPE METRIC
# ============================================================================

def calculate_smape(y_true, y_pred):
    """Calculate SMAPE metric"""
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape

# ============================================================================
# 8. TRAIN MODELS
# ============================================================================

predictions_val = {}
predictions_test = {}

# Load tuned hyperparameters if available
tuned_params_path = os.path.join(DATASET_FOLDER, 'tuned_hyperparameters.json')
if os.path.exists(tuned_params_path):
    with open(tuned_params_path, 'r') as f:
        tuned_params = json.load(f)
else:
    tuned_params = {}

# -------------------------
# 8.1 LIGHTGBM
# -------------------------
if HAS_LGB:
    print("\n" + "="*80)
    print(" TRAINING LIGHTGBM WITH IMAGE FEATURES")
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
    
    print("[*] Training LightGBM with multi-modal features...")
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train, y_train_log,
        eval_set=[(X_val, y_val_log)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
    )
    
    lgb_pred_val = np.expm1(lgb_model.predict(X_val))
    lgb_pred_test = np.expm1(lgb_model.predict(X_test_combined))
    
    lgb_smape = calculate_smape(y_val, lgb_pred_val)
    print(f"\n[+] LightGBM + Images SMAPE: {lgb_smape:.4f}%")
    
    predictions_val['lightgbm_images'] = lgb_pred_val
    predictions_test['lightgbm_images'] = lgb_pred_test

# -------------------------
# 8.2 CATBOOST
# -------------------------
if HAS_CAT:
    print("\n" + "="*80)
    print(" TRAINING CATBOOST WITH IMAGE FEATURES")
    print("="*80)
    
    # Enable GPU for CatBoost if available
    use_gpu_catboost = torch.cuda.is_available()
    if use_gpu_catboost:
        print("[+] Enabling GPU acceleration for CatBoost! 🔥")
    
    if 'catboost' in tuned_params:
        cat_params = tuned_params['catboost']['best_params'].copy()
        # Enable GPU if available
        if use_gpu_catboost:
            cat_params['task_type'] = 'GPU'
            cat_params['devices'] = '0'
        cat_params.update({
            'loss_function': 'RMSE',
            'random_seed': 42,
            'verbose': 200,
            'iterations': 2000,
            'early_stopping_rounds': 100
        })
    else:
        cat_params = {
            'iterations': 2000,
            'learning_rate': 0.03,
            'depth': 7,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': 200,
            'early_stopping_rounds': 100,
            'task_type': 'GPU' if use_gpu_catboost else 'CPU',
            'loss_function': 'RMSE'
        }
        if use_gpu_catboost:
            cat_params['devices'] = '0'
    
    print(f"[*] Training CatBoost on {cat_params.get('task_type', 'CPU')}...")
    cat_model = cb.CatBoostRegressor(**cat_params)
    cat_model.fit(
        X_train, y_train_log,
        eval_set=(X_val, y_val_log),
        verbose=200
    )
    
    cat_pred_val = np.expm1(cat_model.predict(X_val))
    cat_pred_test = np.expm1(cat_model.predict(X_test_combined))
    
    cat_smape = calculate_smape(y_val, cat_pred_val)
    print(f"\n[+] CatBoost + Images SMAPE: {cat_smape:.4f}%")
    
    predictions_val['catboost_images'] = cat_pred_val
    predictions_test['catboost_images'] = cat_pred_test

# ============================================================================
# 9. ENSEMBLE & CREATE SUBMISSION
# ============================================================================

print("\n" + "="*80)
print(" MODEL COMPARISON")
print("="*80)

print(f"\n[*] Validation SMAPE Scores:")
for model_name in predictions_val.keys():
    smape = calculate_smape(y_val, predictions_val[model_name])
    print(f"   {model_name}: {smape:.4f}%")

# Simple average ensemble
if len(predictions_val) > 0:
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
    
    submission_path = os.path.join(DATASET_FOLDER, 'test_out_images.csv')
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
    print(" IMAGE FEATURES COMPLETE!")
    print("="*80)
    
    print(f"\n[*] Expected improvement: 13.0% → 12.0-12.5%")
    print(f"[*] Files created:")
    print(f"   {submission_path}")
    print(f"   {train_cache_path}")
    print(f"   {test_cache_path}")
    print("\n[!] NEXT STEP: Share validation SMAPE for final ensemble")
    print("\n")

else:
    print("[!] ERROR: No predictions generated!")

