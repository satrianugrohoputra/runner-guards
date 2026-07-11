import os
import joblib
import numpy as np
import pandas as pd

# Load the models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
weekly_model_data = joblib.load(os.path.join(BASE_DIR, 'weekly_injury_model.joblib'))
multi_model_data = joblib.load(os.path.join(BASE_DIR, 'multimodal_injury_model.joblib'))

w_model = weekly_model_data['model']
w_features = weekly_model_data['features']

m_model = multi_model_data['model']
m_features = multi_model_data['features']

def calculate_acwr(weekly_loads):
    """
    Calculate Acute:Chronic Workload Ratio (ACWR).
    weekly_loads: list of floats, from oldest [week 3, week 2, week 1] to current [week 0].
    """
    if len(weekly_loads) < 4:
        # Pad with average if not enough weeks
        mean_val = np.mean(weekly_loads) if weekly_loads else 20.0
        while len(weekly_loads) < 4:
            weekly_loads.insert(0, mean_val)
            
    acute = weekly_loads[-1]  # current week
    chronic = np.mean(weekly_loads)  # average of past 4 weeks
    
    if chronic == 0:
        return 0.0
    return acute / chronic

def calculate_growth(current_load, previous_load):
    """Calculate percentage growth in weekly mileage."""
    if previous_load == 0:
        return 100.0 if current_load > 0 else 0.0
    return ((current_load - previous_load) / previous_load) * 100.0

def predict_injury_risk_score(user_features):
    """
    Predict binary injury risk probability (0-100) using the weekly load model.
    user_features: dict with keys matching features of weekly_injury_model
    """
    # Create a dataframe row with the correct feature order
    row = {}
    for feat in w_features:
        row[feat] = user_features.get(feat, 0.0)
        
    df_row = pd.DataFrame([row])
    
    # Predict probability of class 1 (injured)
    prob = w_model.predict_proba(df_row)[0][1]
    
    # Scale the raw model probability (which might be low due to imbalance) to a 0-100 risk score.
    # The baseline rate is around 1.3%. If probability is 20%, it is actually extremely high risk.
    # Let's map it: 0% -> 10, 1.3% (average) -> 35, 10% -> 70, 30%+ -> 95
    if prob < 0.013:
        scaled_score = 10 + (prob / 0.013) * 25
    elif prob < 0.10:
        scaled_score = 35 + ((prob - 0.013) / (0.10 - 0.013)) * 35
    else:
        scaled_score = 70 + min(((prob - 0.10) / (0.30 - 0.10)) * 25, 25)
        
    return float(np.clip(scaled_score, 0, 100))

def predict_injury_types(user_features, base_risk_score):
    """
    Predict probability of specific running injuries based on biomechanical rules and multimodal model predictions.
    Returns a sorted list of dicts with injury name and probability.
    """
    # Get model general prediction
    row = {}
    for feat in m_features:
        row[feat] = user_features.get(feat, 0.0)
    df_row = pd.DataFrame([row])
    df_row = df_row.fillna(0.0)
    
    # Predict probability for Healthy, Low Risk, Injured
    probs = m_model.predict_proba(df_row)[0] # [P(Healthy), P(Low Risk), P(Injured)]
    injured_prob = probs[2] # probability of class 2 (Injured)
    low_risk_prob = probs[1] # probability of class 1 (Low Risk)
    
    # Extract values for biomechanical heuristics
    grf = user_features.get('ground_reaction_force', 1500.0)
    cadence = user_features.get('cadence', 160.0)
    rom = user_features.get('range_of_motion', 120.0)
    joint_angles = user_features.get('joint_angles', 120.0)
    muscle_act = user_features.get('muscle_activity', 200.0)
    fatigue = user_features.get('fatigue_index', 40.0)
    hydration = user_features.get('hydration_level', 80.0)
    temp = user_features.get('body_temperature', 37.0)
    
    # Compute base injury probability based on model and weekly risk score
    p_injury_base = (base_risk_score / 100.0) * 0.7 + (injured_prob + 0.5 * low_risk_prob) * 0.3
    p_injury_base = np.clip(p_injury_base, 0.05, 0.95)
    
    # 1. Knee Pain (Patellofemoral Pain / Runner's Knee)
    # Trigger: Low ROM, high joint angle deviation, high impact
    knee_modifier = 1.0
    if rom < 110:
        knee_modifier += 0.3
    if joint_angles > 140 or joint_angles < 90:
        knee_modifier += 0.2
    if grf > 1800:
        knee_modifier += 0.1
    p_knee = p_injury_base * knee_modifier * 0.85
    
    # 2. Shin Splints (Medial Tibial Stress Syndrome)
    # Trigger: High ground reaction force, low cadence (heavy overstriding)
    shin_modifier = 1.0
    if grf > 1900:
        shin_modifier += 0.4
    if cadence < 155:
        shin_modifier += 0.3  # overstriding increases impact load on shins
    p_shin = p_injury_base * shin_modifier * 0.75
    
    # 3. Hamstring Strain
    # Trigger: High muscle activity under fatigue, high intensity
    ham_modifier = 1.0
    if muscle_act > 400:
        ham_modifier += 0.3
    if fatigue > 60:
        ham_modifier += 0.3
    p_ham = p_injury_base * ham_modifier * 0.65
    
    # 4. Muscle Cramps / Heat Exhaustion
    # Trigger: Low hydration, high body temp, high sweat
    cramp_modifier = 1.0
    if hydration < 65:
        cramp_modifier += 0.5
    if temp > 38.0:
        cramp_modifier += 0.3
    p_cramp = p_injury_base * cramp_modifier * 0.55
    
    # 5. Achilles Tendonitis
    # Trigger: Low range of motion, high training load/intensity
    achilles_modifier = 1.0
    if rom < 100:
        achilles_modifier += 0.3
    if grf > 1700:
        achilles_modifier += 0.2
    p_achilles = p_injury_base * achilles_modifier * 0.50
    
    injuries = [
        {"name": "🦵 Knee Pain (Runner's Knee)", "prob": int(np.clip(p_knee * 100, 5, 95))},
        {"name": "🦵 Shin Splints", "prob": int(np.clip(p_shin * 100, 5, 95))},
        {"name": "🦵 Hamstring Strain", "prob": int(np.clip(p_ham * 100, 5, 95))},
        {"name": "💪 Muscle Cramps", "prob": int(np.clip(p_cramp * 100, 5, 95))},
        {"name": "🦶 Achilles Tendonitis", "prob": int(np.clip(p_achilles * 100, 5, 95))},
    ]
    
    # Sort by probability descending
    return sorted(injuries, key=lambda x: x['prob'], reverse=True)
