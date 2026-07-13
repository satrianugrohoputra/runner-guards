import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier

os.makedirs('models', exist_ok=True)
MODELS_DIR = os.path.abspath('models')


# ===========================================================
# 1. Weekly Injury Risk Model  (XGBoost)
#    Primary : Competitive Runners timeseries dataset
#    Augmented: run_ww_2019 + run_ww_2020 (world marathon)
# ===========================================================
print("=== [1/3] Training Weekly Injury Risk Model (XGBoost) ===")

week_path = "data/Injury Prediction for Competitive Runners/week_approach_maskedID_timeseries.csv"
df_week = pd.read_csv(week_path)

features_week = [
    'total kms', 'nr. sessions', 'nr. rest days', 'max km one day',
    'nr. strength trainings', 'total hours alternative training',
    'avg exertion', 'avg recovery', 'avg training success',
    'rel total kms week 0_1', 'rel total kms week 0_2', 'rel total kms week 1_2'
]

X_primary = df_week[features_week].copy().fillna(df_week[features_week].median())
y_primary  = df_week['injury'].copy()
print(f"  Primary dataset : {len(X_primary):,} rows")

# --- Augmentation: run_ww 2019 + 2020 ---
print("  Loading run_ww augmentation data...")
run19 = pd.read_csv("data/Long-distance running dataset.csv/run_ww_2019_m.csv")
run20 = pd.read_csv("data/Long-distance running dataset.csv/run_ww_2020_m.csv")
df_run = pd.concat([run19, run20], ignore_index=True)
df_run = df_run[df_run['distance'] >= 40.0].dropna(subset=['distance', 'duration'])

df_run['pace_min_per_km'] = df_run['duration'] / df_run['distance']
df_run['est_weekly_km']   = df_run['distance'] * np.random.default_rng(42).uniform(0.8, 1.3, size=len(df_run))
df_run['est_sessions']    = 5.0
df_run['est_rest_days']   = 2.0
df_run['est_max_km_day']  = df_run['est_weekly_km'] / 5.0

pace_max = df_run['pace_min_per_km'].quantile(0.95)
pace_min = df_run['pace_min_per_km'].quantile(0.05)
df_run['est_exertion'] = np.clip(
    (df_run['pace_min_per_km'] - pace_min) / (pace_max - pace_min + 1e-9), 0.1, 0.9
)
df_run['est_recovery']  = 1.0 - df_run['est_exertion']
df_run['est_success']   = df_run['est_recovery']

rng = np.random.default_rng(42)
df_run['est_rel_01'] = rng.uniform(0.90, 1.10, size=len(df_run))
df_run['est_rel_02'] = rng.uniform(0.85, 1.15, size=len(df_run))
df_run['est_rel_12'] = rng.uniform(0.90, 1.10, size=len(df_run))

pace_threshold       = df_run['pace_min_per_km'].quantile(0.80)
df_run['est_injury'] = (df_run['pace_min_per_km'] > pace_threshold).astype(int)

X_aug = pd.DataFrame({
    'total kms':                         df_run['est_weekly_km'].values,
    'nr. sessions':                      df_run['est_sessions'].values,
    'nr. rest days':                     df_run['est_rest_days'].values,
    'max km one day':                    df_run['est_max_km_day'].values,
    'nr. strength trainings':            1.0,
    'total hours alternative training':  0.5,
    'avg exertion':                      df_run['est_exertion'].values,
    'avg recovery':                      df_run['est_recovery'].values,
    'avg training success':              df_run['est_success'].values,
    'rel total kms week 0_1':            df_run['est_rel_01'].values,
    'rel total kms week 0_2':            df_run['est_rel_02'].values,
    'rel total kms week 1_2':            df_run['est_rel_12'].values,
})
y_aug = df_run['est_injury'].values

max_aug = min(len(X_aug), len(X_primary) * 3)
X_aug   = X_aug.sample(n=max_aug, random_state=42).reset_index(drop=True)
y_aug   = y_aug[:max_aug]
print(f"  Augmented rows  : {len(X_aug):,} (from {len(df_run):,} valid records)")

X_combined = pd.concat([X_primary, X_aug], ignore_index=True)
y_combined  = np.concatenate([y_primary.values, y_aug])
print(f"  Combined total  : {len(X_combined):,} rows")

X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_combined, y_combined, test_size=0.2, stratify=y_combined, random_state=42
)

# XGBoost handles imbalance via scale_pos_weight
neg_count  = int((y_train_w == 0).sum())
pos_count  = int((y_train_w == 1).sum())
spw        = neg_count / max(pos_count, 1)
print(f"  scale_pos_weight = {spw:.2f}  (neg={neg_count:,}, pos={pos_count:,})")

model_week = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw,
    eval_metric='auc',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
model_week.fit(
    X_train_w, y_train_w,
    eval_set=[(X_test_w, y_test_w)],
    verbose=False
)

y_pred_w = model_week.predict(X_test_w)
y_prob_w = model_week.predict_proba(X_test_w)[:, 1]
print("  Classification Report:")
print(classification_report(y_test_w, y_pred_w))
print(f"  ROC-AUC: {roc_auc_score(y_test_w, y_prob_w):.4f}")
print(f"  Best iteration: {model_week.best_iteration}")

# Feature importance (XGBoost gain)
fi_week = dict(zip(features_week, model_week.feature_importances_.tolist()))
fi_week_sorted = dict(sorted(fi_week.items(), key=lambda x: x[1], reverse=True))
print("  Top features:", list(fi_week_sorted.keys())[:5])

joblib.dump({
    'model':              model_week,
    'features':           features_week,
    'feature_importance': fi_week_sorted,
    'model_type':         'xgboost'
}, os.path.join(MODELS_DIR, 'weekly_injury_model.joblib'))
print("  [OK] Saved: models/weekly_injury_model.joblib\n")


# ===========================================================
# 2. Multimodal Injury Classification Model  (XGBoost)
#    Predicts: Healthy (0) / Low Risk (1) / Injured (2)
# ===========================================================
print("=== [2/3] Training Multimodal Injury Model (XGBoost) ===")

multi_path   = "data/multimodal-sports-injury/multimodal_sports_injury_dataset.csv"
df_multi     = pd.read_csv(multi_path)
df_multi_run = df_multi[df_multi['sport_type'].isin(['Track', 'Soccer', 'Other'])].copy()
print(f"  Filtered rows: {len(df_multi_run):,} / {len(df_multi):,}")

features_multi = [
    'heart_rate', 'body_temperature', 'hydration_level', 'sleep_quality',
    'recovery_score', 'stress_level', 'muscle_activity', 'joint_angles',
    'gait_speed', 'cadence', 'step_count', 'jump_height',
    'ground_reaction_force', 'range_of_motion', 'ambient_temperature',
    'humidity', 'altitude', 'training_intensity', 'training_duration',
    'training_load', 'fatigue_index'
]

X_multi = df_multi_run[features_multi].copy().fillna(df_multi_run[features_multi].median())
y_multi = df_multi_run['injury_occurred'].copy()

X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, stratify=y_multi, random_state=42
)

# Compute per-class weights for multi-class
class_counts = np.bincount(y_train_m.astype(int))
sample_weight = np.array([1.0 / class_counts[c] for c in y_train_m.astype(int)])
sample_weight = sample_weight / sample_weight.sum() * len(sample_weight)

model_multi = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    early_stopping_rounds=20,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
model_multi.fit(
    X_train_m, y_train_m,
    sample_weight=sample_weight,
    eval_set=[(X_test_m, y_test_m)],
    verbose=False
)

y_pred_m = model_multi.predict(X_test_m)
print("  Classification Report:")
print(classification_report(y_test_m, y_pred_m, target_names=['Healthy', 'Low Risk', 'Injured']))
print(f"  Best iteration: {model_multi.best_iteration}")

fi_multi = dict(zip(features_multi, model_multi.feature_importances_.tolist()))
fi_multi_sorted = dict(sorted(fi_multi.items(), key=lambda x: x[1], reverse=True))
print("  Top features:", list(fi_multi_sorted.keys())[:5])

joblib.dump({
    'model':              model_multi,
    'features':           features_multi,
    'feature_importance': fi_multi_sorted,
    'model_type':         'xgboost'
}, os.path.join(MODELS_DIR, 'multimodal_injury_model.joblib'))
print("  [OK] Saved: models/multimodal_injury_model.joblib\n")


# ===========================================================
# 3. Recovery Estimator Model  (XGBoost)
#    Predicts: Recovery_Success (0/1) from Athlete Recovery DS
# ===========================================================
print("=== [3/3] Training Athlete Recovery Model (XGBoost) ===")

rec_path = "data/Athlete_recovery_dataset/Athlete_recovery_dataset.csv"
df_rec   = pd.read_csv(rec_path)
print(f"  Recovery dataset: {len(df_rec):,} rows")
print(f"  Injury types    : {df_rec['Injury_Type'].unique().tolist()}")

le_injury   = LabelEncoder()
le_severity = LabelEncoder()
le_intensity= LabelEncoder()
le_therapy  = LabelEncoder()
le_muscle   = LabelEncoder()

df_rec['Injury_Type_enc']            = le_injury.fit_transform(df_rec['Injury_Type'])
df_rec['Injury_Severity_enc']        = le_severity.fit_transform(df_rec['Injury_Severity'])
df_rec['Training_Intensity_enc']     = le_intensity.fit_transform(df_rec['Training_Intensity'])
df_rec['Recovery_Therapy_Type_enc']  = le_therapy.fit_transform(df_rec['Recovery_Therapy_Type'])
df_rec['Muscle_Recovery_Status_enc'] = le_muscle.fit_transform(df_rec['Muscle_Recovery_Status'])

features_rec = [
    'Injury_Type_enc', 'Injury_Severity_enc', 'Heart_Rate',
    'POMS_Score', 'Training_Intensity_enc', 'Sleep_Hours',
    'Training_Days_per_Week', 'Recovery_Days_per_Week',
    'Muscle_Recovery_Status_enc', 'Confidence_Score'
]

X_rec = df_rec[features_rec].fillna(df_rec[features_rec].median())
y_rec = df_rec['Recovery_Success']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_rec, y_rec, test_size=0.2, stratify=y_rec, random_state=42
)

neg_r = int((y_train_r == 0).sum())
pos_r = int((y_train_r == 1).sum())
spw_r = neg_r / max(pos_r, 1)

model_rec = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=spw_r,
    eval_metric='auc',
    early_stopping_rounds=15,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)
model_rec.fit(
    X_train_r, y_train_r,
    eval_set=[(X_test_r, y_test_r)],
    verbose=False
)

y_pred_r = model_rec.predict(X_test_r)
print("  Classification Report:")
print(classification_report(y_test_r, y_pred_r, target_names=['Not Recovered', 'Recovered']))
print(f"  Best iteration: {model_rec.best_iteration}")

# Recovery lookup table (unchanged logic)
recovery_lookup = (
    df_rec.groupby(['Injury_Type', 'Injury_Severity'])
    .agg(
        median_recovery_days=('Recovery_Time', 'median'),
        best_therapy=('Recovery_Therapy_Type', lambda x: x.mode()[0])
    )
    .reset_index()
    .to_dict(orient='records')
)

joblib.dump({
    'model':      model_rec,
    'features':   features_rec,
    'label_encoders': {
        'injury': le_injury, 'severity': le_severity,
        'intensity': le_intensity, 'therapy': le_therapy, 'muscle': le_muscle,
    },
    'recovery_lookup': recovery_lookup,
    'raw_data': df_rec[[
        'Injury_Type', 'Injury_Severity', 'Recovery_Time',
        'Recovery_Therapy_Type', 'Recovery_Success'
    ]].to_dict(orient='records'),
    'model_type': 'xgboost'
}, os.path.join(MODELS_DIR, 'recovery_model.joblib'))
print("  [OK] Saved: models/recovery_model.joblib\n")

print("=" * 55)
print("[OK] All 3 XGBoost models trained & saved successfully!")
print("=" * 55)
