import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# -----------------
# 1. Train Injury Risk Model (Competitive Runners Dataset)
# -----------------
print("=== Training Weekly Injury Risk Model ===")
week_path = "data/Injury Prediction for Competitive Runners/week_approach_maskedID_timeseries.csv"
df_week = pd.read_csv(week_path)

# Select key features
features_week = [
    'total kms', 'nr. sessions', 'nr. rest days', 'max km one day', 
    'nr. strength trainings', 'total hours alternative training',
    'avg exertion', 'avg recovery', 'avg training success',
    'rel total kms week 0_1', 'rel total kms week 0_2', 'rel total kms week 1_2'
]

X_week = df_week[features_week].copy()
y_week = df_week['injury'].copy()

# Handle missing values
X_week = X_week.fillna(X_week.median())

# Train/Test Split
X_train_w, X_test_w, y_train_w, y_test_w = train_test_split(
    X_week, y_week, test_size=0.2, stratify=y_week, random_state=42
)

# Train Classifier
model_week = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', 
    max_depth=10,
    random_state=42, 
    n_jobs=-1
)
model_week.fit(X_train_w, y_train_w)

# Evaluate
y_pred_w = model_week.predict(X_test_w)
y_prob_w = model_week.predict_proba(X_test_w)[:, 1]
print("Classification Report:")
print(classification_report(y_test_w, y_pred_w))
print(f"ROC AUC Score: {roc_auc_score(y_test_w, y_prob_w):.4f}")

# Save Model
joblib.dump({
    'model': model_week,
    'features': features_week
}, 'weekly_injury_model.joblib')
print("Saved weekly_injury_model.joblib")


# -----------------
# 2. Train Multimodal Injury Classification Model
# -----------------
print("\n=== Training Multimodal Sports Injury Model ===")
multi_path = "data/multimodal-sports-injury/multimodal_sports_injury_dataset.csv"
df_multi = pd.read_csv(multi_path)

# Filter for Track and Soccer (most relevant to long-distance running/cardio)
df_multi_run = df_multi[df_multi['sport_type'].isin(['Track', 'Soccer', 'Other'])].copy()
print(f"Filtered rows: {len(df_multi_run)} from {len(df_multi)}")

features_multi = [
    'heart_rate', 'body_temperature', 'hydration_level', 'sleep_quality', 
    'recovery_score', 'stress_level', 'muscle_activity', 'joint_angles', 
    'gait_speed', 'cadence', 'step_count', 'jump_height', 
    'ground_reaction_force', 'range_of_motion', 'ambient_temperature', 
    'humidity', 'altitude', 'training_intensity', 'training_duration', 
    'training_load', 'fatigue_index'
]

X_multi = df_multi_run[features_multi].copy()
y_multi = df_multi_run['injury_occurred'].copy()

# Handle missing values
X_multi = X_multi.fillna(X_multi.median())

# Train/Test Split
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_multi, y_multi, test_size=0.2, stratify=y_multi, random_state=42
)

# Train Classifier
model_multi = RandomForestClassifier(
    n_estimators=100, 
    class_weight='balanced', 
    max_depth=12,
    random_state=42, 
    n_jobs=-1
)
model_multi.fit(X_train_m, y_train_m)

# Evaluate
y_pred_m = model_multi.predict(X_test_m)
print("Classification Report:")
print(classification_report(y_test_m, y_pred_m, target_names=['Healthy', 'Low Risk', 'Injured']))

# Save Model
joblib.dump({
    'model': model_multi,
    'features': features_multi
}, 'multimodal_injury_model.joblib')
print("Saved multimodal_injury_model.joblib")
