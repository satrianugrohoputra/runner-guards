import streamlit as st
import pandas as pd
import numpy as np
import os
import time

# Import our engine and brain modules
import engine
import brain

def load_keys_from_secrets():
    """
    Load API keys from st.secrets supporting both flat and nested structures.
    """
    gemini = ""
    fireworks = ""
    
    # 1. Try nested structure st.secrets["api_keys"]["..."]
    try:
        if "api_keys" in st.secrets:
            gemini = st.secrets["api_keys"].get("GEMINI_API_KEY", "")
            fireworks = st.secrets["api_keys"].get("FIREWORKS_API_KEY", "")
    except Exception:
        pass
        
    # 2. Fallback to top-level st.secrets["..."]
    try:
        if not gemini and "GEMINI_API_KEY" in st.secrets:
            gemini = st.secrets["GEMINI_API_KEY"]
        if not fireworks and "FIREWORKS_API_KEY" in st.secrets:
            fireworks = st.secrets["FIREWORKS_API_KEY"]
    except Exception:
        pass
        
    return gemini, fireworks


# Set up page configurations
ICON_FLATICON_URL = "https://cdn-icons-png.flaticon.com/512/9233/9233133.png"
st.set_page_config(
    page_title="PaceGuard AI - AI Running Injury Coach",
    page_icon=ICON_FLATICON_URL,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom Styling (Rich dark glassmorphism theme)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 30px;
    border-radius: 15px;
    color: white;
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 10px 20px rgba(0,0,0,0.2);
}

.glow-card {
    background: #182030;
    border-radius: 15px;
    padding: 25px;
    border: 1px solid #2d3b55;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    margin-bottom: 20px;
    transition: all 0.3s ease;
    color: #f1f5f9;
}

.glow-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 10px 25px rgba(46, 204, 113, 0.2);
    border-color: #3b4e70;
}

.risk-circle {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 140px;
    height: 140px;
    border-radius: 50%;
    margin: 0 auto 15px auto;
    font-weight: 700;
}

.risk-low {
    background: radial-gradient(circle, #0e2b1b 0%, #1b5231 100%);
    border: 4px solid #2ecc71;
    color: #2ecc71;
    box-shadow: 0 0 15px rgba(46, 204, 113, 0.4);
}

.risk-medium {
    background: radial-gradient(circle, #332709 0%, #614a10 100%);
    border: 4px solid #f1c40f;
    color: #f1c40f;
    box-shadow: 0 0 15px rgba(241, 196, 15, 0.4);
}

.risk-high {
    background: radial-gradient(circle, #3a1515 0%, #6d2222 100%);
    border: 4px solid #e74c3c;
    color: #e74c3c;
    box-shadow: 0 0 15px rgba(231, 76, 60, 0.4);
}

.xai-item {
    padding: 10px 15px;
    margin-bottom: 10px;
    border-radius: 8px;
    background: #1d2636;
    border-left: 4px solid #3498db;
    font-size: 14px;
    color: #f1f5f9;
}

.xai-warn {
    border-left-color: #f1c40f;
}

.xai-danger {
    border-left-color: #e74c3c;
}

.xai-ok {
    border-left-color: #2ecc71;
}

.chat-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 8px;
    color: white;
}

.badge-deterministic {
    background: #9b59b6;
}

.badge-local {
    background: #34495e;
    border: 1px solid #7f8c8d;
}

.badge-cloud {
    background: #2980b9;
}

.chat-bubble-user {
    background-color: #2c3e50;
    padding: 12px 18px;
    border-radius: 15px 15px 0 15px;
    margin-bottom: 15px;
    align-self: flex-end;
    max-width: 80%;
    margin-left: auto;
}

.chat-bubble-user, .chat-bubble-user * {
    color: #f1f5f9 !important;
}

.chat-bubble-coach {
    background-color: #1a252f;
    padding: 12px 18px;
    border-radius: 15px 15px 15px 0;
    margin-bottom: 15px;
    border: 1px solid #2d3b55;
    max-width: 80%;
}

.chat-bubble-coach, .chat-bubble-coach .chat-text, .chat-bubble-coach .chat-text * {
    color: #f1f5f9 !important;
}

.chat-bubble-user code, .chat-bubble-coach code {
    background-color: #0f172a !important;
    color: #e2e8f0 !important;
    padding: 2px 6px !important;
    border-radius: 4px !important;
}

.chat-badge {
    color: #ffffff !important;
}

.chat-bubble-coach div[style*="font-size: 11px"] span {
    color: #a0aec0 !important;
}

.chat-bubble-coach div[style*="font-size: 11px"] span strong {
    color: #ffffff !important;
}

.chat-bubble-coach div[style*="font-size: 11px"] span[style*="color: #2ecc71"],
.chat-bubble-coach div[style*="font-size: 11px"] span[style*="color: #2ecc71"] * {
    color: #2ecc71 !important;
}
/* Force all Streamlit markdown text white with readable shadow */
[data-testid="stMarkdownContainer"] p, 
[data-testid="stMarkdownContainer"] h1, 
[data-testid="stMarkdownContainer"] h2, 
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] span {
    color: #FFFFFF !important;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9) !important;
}

/* =====================================================
   MOBILE RESPONSIVE — Applies ONLY on screens ≤ 768px
   Desktop styles above are completely untouched.
   ===================================================== */
@media (max-width: 768px) {

    /* ── Header: scale down font for small screens ── */
    .main-header {
        padding: 16px 12px;
        border-radius: 10px;
        margin-bottom: 14px;
    }
    .main-header h1 {
        font-size: 1.6rem !important;
        letter-spacing: -0.5px !important;
    }
    .main-header p {
        font-size: 0.85rem !important;
    }

    /* ── Cards: reduce padding to save vertical space ── */
    .glow-card {
        padding: 14px 12px;
        border-radius: 10px;
        margin-bottom: 12px;
    }
    /* Disable hover lift effect on touch — it can get "stuck" */
    .glow-card:hover {
        transform: none;
    }

    /* ── Risk Circle: shrink to fit narrow screens ── */
    .risk-circle {
        width: 100px;
        height: 100px;
    }

    /* ── XAI items: slightly smaller text on mobile ── */
    .xai-item {
        font-size: 13px;
        padding: 8px 12px;
    }

    /* ── Chat bubbles: go nearly full width on mobile ── */
    .chat-bubble-user,
    .chat-bubble-coach {
        max-width: 96%;
        padding: 10px 12px;
    }

    /* ── Force Streamlit horizontal column blocks to stack vertically ──
       This makes st.columns() responsive on mobile without changing Python code */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0px !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* ── Prevent iOS/Android auto-zoom on form input focus (min 16px) ── */
    input, textarea, select {
        font-size: 16px !important;
    }
    [data-testid="stTextInput"] input {
        font-size: 16px !important;
    }

    /* ── Suggestion chip buttons: bigger tap targets ── */
    [data-testid="stButton"] button {
        padding: 10px 14px !important;
        font-size: 13px !important;
        width: 100% !important;
        margin-bottom: 6px !important;
    }

    /* ── Subheaders: reduce size slightly ── */
    h2, h3 {
        font-size: 1.1rem !important;
    }

    /* ── Stats token row in chat: allow wrapping ── */
    .chat-bubble-coach div[style*="justify-content: space-between"] {
        flex-wrap: wrap !important;
        gap: 6px !important;
    }
}

/* Extra-small phones (≤ 480px, e.g. older Android) */
@media (max-width: 480px) {
    .main-header h1 {
        font-size: 1.3rem !important;
    }
    .glow-card {
        padding: 10px 10px;
    }
    .risk-circle {
        width: 85px;
        height: 85px;
    }
    .risk-circle span:first-child {
        font-size: 1.6rem !important;
    }
    .xai-item {
        font-size: 12px;
    }
}
</style>
""", unsafe_allow_html=True)
# Using your chosen Unsplash image ID
bg_img_url = "https://images.unsplash.com/photo-1549896869-ca27eeffe4fb?q=80&w=2070&auto=format&fit=crop"

page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
    /* Load background image from URL */
    background-image: url("{bg_img_url}");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* On mobile, fixed attachment causes flickering/blank in Android WebView */
@media (max-width: 768px) {{
    [data-testid="stAppViewContainer"] {{
        background-attachment: scroll;
    }}
}}

/* transparent black overlay for readability */
[data-testid="stAppViewContainer"]::before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(15, 23, 42, 0.85); /* Dark slate color with 85% opacity */
    z-index: -1;
}}
</style>
"""

# Injecting CSS to Streamlit
st.markdown(page_bg_img, unsafe_allow_html=True)
# -----------------
# SIDEBAR CONTROL
# -----------------
st.sidebar.title("PaceGuard Control")

# ── Mode Selector ──
st.sidebar.markdown("### 📊 Analysis Mode")
input_mode = st.sidebar.radio(
    "Choose how to enter data:",
    ["Preset Persona (Demo)", "Quick Mode (Fast Input)", "Detail Mode (Full Input)"],
    label_visibility="collapsed"
)

# =============================================================
# MODE 1: PRESET PERSONA
# =============================================================
if input_mode == "Preset Persona (Demo)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Select Runner Scenario:**")
    scenario = st.sidebar.selectbox(
        "Persona:",
        [
            "Andrew (Safe Runner)",
            "Stephen (Overtrained Runner)",
            "Claudia (Fatigued Runner)",
        ]
    )
    is_quick_mode = False

    if scenario == "Andrew (Safe Runner)":
        st.sidebar.info("🎯 **Andrew** trains regularly with slow mileage increases (ideal ACWR).")
        kms_weeks = [36.0, 38.0, 39.0, 40.0]
        sessions = 5; rest_days = 2; max_km_day = 10.0; strength = 2; alt_hours = 1.0
        exertion = 0.12; recovery = 0.72; success = 0.80
        cadence = 172; grf = 1400.0; rom = 125.0; joint_angles = 115.0
        hydration = 85.0; sleep = 8.2; stress = 0.20
        pace_start = "5:30"; pace_end = "5:25"; hr_start = 142; hr_end = 143

    elif scenario == "Stephen (Overtrained Runner)":
        st.sidebar.warning("⚠️ **Stephen** experienced an extreme spike in weekly mileage.")
        kms_weeks = [20.0, 22.0, 25.0, 65.0]
        sessions = 6; rest_days = 1; max_km_day = 22.0; strength = 0; alt_hours = 0.0
        exertion = 0.85; recovery = 0.22; success = 0.30
        cadence = 152; grf = 2100.0; rom = 95.0; joint_angles = 145.0
        hydration = 70.0; sleep = 4.8; stress = 0.75
        pace_start = "5:10"; pace_end = "4:55"; hr_start = 155; hr_end = 175

    else:  # Claudia
        st.sidebar.warning("⚠️ **Claudia** runs with constant mileage but has a sharp increase in heart rate (Aerobic Decoupling).")
        kms_weeks = [40.0, 41.0, 42.0, 42.0]
        sessions = 5; rest_days = 2; max_km_day = 12.0; strength = 1; alt_hours = 0.0
        exertion = 0.60; recovery = 0.35; success = 0.50
        cadence = 165; grf = 1650.0; rom = 115.0; joint_angles = 120.0
        hydration = 58.0; sleep = 5.2; stress = 0.60
        pace_start = "5:10"; pace_end = "5:35"; hr_start = 148; hr_end = 168

# =============================================================
# MODE 2: QUICK MODE
# =============================================================
elif input_mode == "Quick Mode (Fast Input)":
    st.sidebar.markdown("---")
    st.sidebar.markdown("**⚡ Fill in these 5 key metrics (~30 seconds):**")
    is_quick_mode = True

    col_w1, col_w0 = st.sidebar.columns(2)
    q_km_last = col_w1.number_input("Last Week's Km", min_value=0.0, max_value=200.0, value=30.0, step=1.0, format="%.0f")
    q_km_now  = col_w0.number_input("This Week's Km", min_value=0.0, max_value=200.0, value=40.0, step=1.0, format="%.0f")

    q_sessions   = st.sidebar.slider("Training Sessions / Week", 1, 14, 5, 1)
    q_rest_days  = st.sidebar.slider("Rest Days / Week", 0, 7, 2, 1)
    q_condition  = st.sidebar.selectbox(
        "Current Physical Condition:",
        ["💪 Fresh & Energetic", "😐 Neutral / Okay", "😴 Fatigued / Sore"]
    )

    st.sidebar.info("⚠️ **Quick Mode** uses average estimates for unprovided biomechanical data. Accuracy ~70%. Use Detail Mode for precise analysis.")

    # Auto-estimate missing biomechanical values based on condition
    if q_condition == "💪 Fresh & Energetic":
        exertion = 0.25; recovery = 0.80; success = 0.85; stress = 0.15
        hydration = 85.0; sleep = 8.0
        cadence = 172; grf = 1450.0; rom = 122.0; joint_angles = 115.0
        hr_start = 140; hr_end = 142
    elif q_condition == "😐 Neutral / Okay":
        exertion = 0.50; recovery = 0.50; success = 0.60; stress = 0.35
        hydration = 75.0; sleep = 6.5
        cadence = 165; grf = 1600.0; rom = 115.0; joint_angles = 118.0
        hr_start = 148; hr_end = 153
    else:  # Fatigued / Sore
        exertion = 0.78; recovery = 0.25; success = 0.35; stress = 0.70
        hydration = 60.0; sleep = 4.5
        cadence = 155; grf = 1900.0; rom = 100.0; joint_angles = 132.0
        hr_start = 155; hr_end = 172

    kms_weeks    = [q_km_last * 0.85, q_km_last * 0.92, q_km_last, q_km_now]
    sessions     = q_sessions
    rest_days    = q_rest_days
    max_km_day   = round(q_km_now / max(sessions, 1) * 1.5, 1)
    strength     = 1
    alt_hours    = 0.5
    pace_start   = "5:30"
    pace_end     = "5:40" if q_condition == "😴 Fatigued / Sore" else "5:30"

# =============================================================
# MODE 3: DETAIL MODE
# =============================================================
else:
    st.sidebar.markdown("---")
    is_quick_mode = False

    with st.sidebar.expander("📏 Training Load", expanded=True):
        col_w1, col_w0 = st.sidebar.columns(2)
        w1_kms = col_w1.slider("Last Week's Km", 0.0, 150.0, 20.0, 1.0)
        w0_kms = col_w0.slider("This Week's Km", 0.0, 150.0, 40.0, 1.0)
        kms_weeks  = [28.0, 29.0, w1_kms, w0_kms]
        sessions   = st.sidebar.slider("Weekly Running Sessions", 0, 14, 4, 1)
        rest_days  = st.sidebar.slider("Rest Days", 0, 7, 2, 1)
        max_km_day = st.sidebar.slider("Max Distance in 1 Day (km)", 0.0, 50.0, 15.0, 0.5)
        strength   = st.sidebar.slider("Strength Training Sessions", 0, 5, 1, 1)
        alt_hours  = st.sidebar.slider("Alternative Training Hours", 0.0, 10.0, 1.0, 0.5)

    with st.sidebar.expander("🏃 Intensity & Recovery"):
        exertion  = st.sidebar.slider("Average Exertion (0-1)", 0.0, 1.0, 0.4, 0.05)
        recovery  = st.sidebar.slider("Average Recovery (0-1)", 0.0, 1.0, 0.5, 0.05)
        success   = st.sidebar.slider("Average Training Success (0-1)", 0.0, 1.0, 0.6, 0.05)
        hydration = st.sidebar.slider("Hydration (%)", 40, 100, 80, 5)
        sleep     = st.sidebar.slider("Sleep Quality (0-10)", 0.0, 10.0, 7.5, 0.5)
        stress    = st.sidebar.slider("Stress Level (0-1)", 0.0, 1.0, 0.3, 0.05)

    with st.sidebar.expander("⚙️ Biomechanics"):
        cadence      = st.sidebar.slider("Cadence (spm)", 120, 210, 165, 2)
        grf          = st.sidebar.slider("Ground Reaction Force - GRF (N)", 800, 2800, 1500, 50)
        rom          = st.sidebar.slider("Range of Motion (°)", 60, 180, 120, 5)
        joint_angles = st.sidebar.slider("Knee Joint Angle (°)", 60, 180, 115, 5)

    with st.sidebar.expander("❤️ Pace & Heart Rate"):
        col_p1, col_p2 = st.sidebar.columns(2)
        pace_start_min = col_p1.number_input("Initial Pace (min)", 3.0, 10.0, 6.0, 0.1)
        pace_start_sec = col_p1.number_input("Initial Pace (sec)", 0, 59, 30, 1)
        pace_start = f"{int(pace_start_min)}:{pace_start_sec:02d}"
        pace_end_min = col_p2.number_input("Final Pace (min)", 3.0, 10.0, 6.0, 0.1)
        pace_end_sec = col_p2.number_input("Final Pace (sec)", 0, 59, 25, 1)
        pace_end = f"{int(pace_end_min)}:{pace_end_sec:02d}"
        col_h1, col_h2 = st.sidebar.columns(2)
        hr_start = col_h1.slider("Initial HR (bpm)", 100, 200, 140, 1)
        hr_end   = col_h2.slider("Final HR (bpm)", 100, 200, 160, 1)

# API Keys — Load directly from secrets.toml (no UI elements to keep layout clean)
gemini_key, fireworks_key = load_keys_from_secrets()

# Calculate math statistics
current_week_kms = kms_weeks[-1]
last_week_kms = kms_weeks[-2]
growth_pct = engine.calculate_growth(current_week_kms, last_week_kms)
acwr_val = engine.calculate_acwr(kms_weeks)

# Convert HR/Pace change into a drift percentage
# HR Drift calculation: ((hr_end - hr_start) / hr_start) * 100
hr_drift_val = ((hr_end - hr_start) / hr_start) * 100

# Assemble features dictionary for the models
user_metrics_weekly = {
    'total kms': current_week_kms,
    'nr. sessions': float(sessions),
    'nr. rest days': float(rest_days),
    'max km one day': max_km_day,
    'nr. strength trainings': float(strength),
    'total hours alternative training': alt_hours,
    'avg exertion': exertion,
    'avg recovery': recovery,
    'avg training success': success,
    'rel total kms week 0_1': current_week_kms / last_week_kms if last_week_kms > 0 else 0.0,
    'rel total kms week 0_2': current_week_kms / kms_weeks[-3] if kms_weeks[-3] > 0 else 0.0,
    'rel total kms week 1_2': last_week_kms / kms_weeks[-3] if kms_weeks[-3] > 0 else 0.0
}

user_metrics_multimodal = {
    'heart_rate': float((hr_start + hr_end) / 2),
    'body_temperature': 37.5 if hr_drift_val > 10 else 36.8,
    'hydration_level': hydration,
    'sleep_quality': sleep,
    'recovery_score': recovery * 100,
    'stress_level': stress,
    'muscle_activity': grf * 0.25, # rough mapping for model input
    'joint_angles': joint_angles,
    'gait_speed': 3.0,
    'cadence': float(cadence),
    'step_count': float(sessions * 8000),
    'jump_height': 0.35,
    'ground_reaction_force': grf,
    'range_of_motion': rom,
    'ambient_temperature': 25.0,
    'humidity': 60.0,
    'altitude': 100.0,
    'training_intensity': float(int(exertion * 10)),
    'training_duration': float((current_week_kms / 10.0) * 60.0), # estimated minutes
    'training_load': float(current_week_kms * exertion * 15),
    'fatigue_index': float(hr_drift_val * 4.0) if hr_drift_val > 0 else 15.0
}

# Run model predictions
risk_score = engine.predict_injury_risk_score(user_metrics_weekly)
injury_types = engine.predict_injury_types(user_metrics_multimodal, risk_score)

# Package all metrics for the AI coach context
metrics_context = {
    'acwr': acwr_val,
    'total_kms': current_week_kms,
    'risk_score': risk_score,
    'growth': growth_pct,
    'hr_drift': hr_drift_val,
    'pace_start': pace_start,
    'pace_end': pace_end,
    'sessions': sessions,
    'rest_days': rest_days,
    'recovery_score': recovery * 100,
    'hydration_level': hydration,
    'sleep_quality': sleep,
    'stress_level': stress,
    'ground_reaction_force': grf,
    'cadence': cadence,
    'data_mode': 'Quick' if is_quick_mode else 'Detail'
}


# -----------------
# MAIN DASHBOARD UI
# -----------------

FLATICON_URL = "https://cdn-icons-png.flaticon.com/512/94/94148.png"

# =====================================================================
# 1. SET UP PAGE CONFIGURATION (Untuk Ikon di TAB BROWSER)
# =====================================================================
st.set_page_config(
    page_title="PaceGuard AI - AI Running Injury Coach",
    page_icon=FLATICON_URL,
    layout="wide",
    initial_sidebar_state="collapsed"  
)

# =====================================================================
# 2. GLOWING HEADER BANNER (Untuk Ikon di UTAMA HALAMAN)
# =====================================================================
st.markdown(f"""
<div class="main-header">
    <h1 style="margin: 0; font-weight: 700; font-size: 2.8rem; letter-spacing: -1px; text-align: center;">
        <img src="{FLATICON_URL}" style="width: 50px; height: 50px; object-fit: contain; filter: brightness(0) invert(1);"/>
        PaceGuard AI
    </h1>
    <p style="margin: 5px 0 0 0; font-size: 1.1rem; opacity: 0.9;">"Train Smart. Run Far. Stay Injury-Free."</p>
</div>
""", unsafe_allow_html=True)

# Quick Mode notice banner
if is_quick_mode:
    condition_label = q_condition
    st.info(
        f" **Quick Mode Active** — Analysis based on minimal metrics. "
        f"This Week's Km: **{q_km_now:.0f} km** | Condition: **{condition_label}** | "
        f"Biomechanical data is automatically estimated. Accuracy ~70%.",
        icon="⚡"
    )

# Row 1: Three columns containing summary cards
col1, col2, col3 = st.columns(3)

# Card 1: Weekly Volume
with col1:
    growth_color = "red" if growth_pct > 15 else "green"
    st.markdown(f"""
    <div class="glow-card">
        <h3 style="margin-top:0; color:#3498db;">WEEKLY VOLUME</h3>
        <p style="font-size: 2.5rem; font-weight: 700; margin: 0;">{current_week_kms:.1f} <span style="font-size: 1.2rem; font-weight: 400; color:#95a5a6;">km</span></p>
        <div style="margin-top: 15px; font-size: 15px;">
            <div>Growth: <strong style="color: {growth_color};">{growth_pct:+.1f}%</strong> (Max ideal +10%)</div>
            <div style="margin-top: 5px;">Sessions: <strong>{sessions} sessions</strong> | Rest: <strong>{rest_days} days</strong></div>
            <div style="margin-top: 5px;">Max Distance in 1 Day: <strong>{max_km_day:.1f} km</strong></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Card 2: Injury Risk Score
with col2:
    if risk_score < 35:
        risk_class = "risk-low"
        risk_text = "LOW RISK"
        risk_color = "#2ecc71"
    elif risk_score < 60:
        risk_class = "risk-medium"
        risk_text = "MEDIUM RISK"
        risk_color = "#f1c40f"
    else:
        risk_class = "risk-high"
        risk_text = "HIGH RISK"
        risk_color = "#e74c3c"
        
    st.markdown(f"""
    <div class="glow-card" style="text-align: center;">
        <h3 style="margin-top:0; text-align: left; color:#a5b1c2;">INJURY RISK</h3>
        <div class="risk-circle {risk_class}">
            <span style="font-size: 2.2rem; line-height: 1.2;">{risk_score:.0f}</span>
            <span style="font-size: 0.8rem; font-weight: 400; opacity: 0.8;">/ 100</span>
        </div>
        <strong style="color: {risk_color}; font-size: 1.1rem; letter-spacing: 1px;">{risk_text}</strong>
    </div>
    """, unsafe_allow_html=True)

# Card 3: Sports Science Indices
with col3:
    acwr_color = "red" if acwr_val > 1.5 or acwr_val < 0.8 else ("yellow" if acwr_val > 1.3 else "green")
    acwr_text = "DANGER" if acwr_val > 1.5 else ("CAUTION" if acwr_val > 1.3 or acwr_val < 0.8 else "OPTIMAL")
    st.markdown(f"""
    <div class="glow-card">
        <h3 style="margin-top:0; color:#e67e22;">WORKLOAD INDICES</h3>
        <div style="margin-bottom: 12px;">
            <div style="font-size: 13px; color:#bdc3c7;">ACWR (Acute:Chronic Workload Ratio)</div>
            <div style="display: flex; align-items: baseline; gap: 10px;">
                <span style="font-size: 2rem; font-weight: 700; color:{'#e74c3c' if acwr_color == 'red' else ('#f1c40f' if acwr_color == 'yellow' else '#2ecc71')};">{acwr_val:.2f}</span>
                <span style="font-size: 0.9rem; padding: 2px 6px; border-radius: 4px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); font-weight: 600;">{acwr_text}</span>
            </div>
        </div>
        <div style="font-size: 14px; border-top: 1px solid #2d3b55; padding-top: 10px; display: flex; justify-content: space-between;">
            <span>Stress Level: <strong>{stress * 100:.0f}%</strong></span>
            <span>Sleep Quality: <strong>{sleep:.1f}/10</strong></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Row 2: Charts (past volume trend & likely injury probability bar charts)
chart_col, injury_col = st.columns([1, 1])

with chart_col:
    st.subheader("Weekly Running Volume (Trend Load)")
    # Render Bar chart for past 4 weeks
    weeks_df = pd.DataFrame({
        "Running Volume (km)": kms_weeks
    }, index=["Week 3", "Week 2", "Week 1", "Week 0 (Current)"])
    st.bar_chart(weeks_df, height=220)

with injury_col:
    st.subheader("🧬 Injury Type Probabilities (Multimodal)")
    # Renders the likely injury list as a progress bar chart
    for idx, inj in enumerate(injury_types):
        p_val = inj['prob']
        # Progress bar colors based on probability
        p_color = "error" if p_val > 60 else ("warning" if p_val > 35 else "success")
        st.write(f"**{inj['name']}** ({p_val}%)")
        st.progress(p_val / 100.0)

# Row 3: Biomechanical detail, Aerobic Decoupling & Explainable AI (XAI)
detail_col1, detail_col2 = st.columns([1.1, 1])

with detail_col1:
    st.subheader("📉 Pace & Heart Rate Drift Analysis")
    st.write("Aerobic decoupling (HR Drift) detects cardiovascular drift during running.")
    
    # Render a small UI simulation showing pace change and HR change
    drift_badge = "⚠️ Danger: Fatigue Detected" if hr_drift_val > 10 else ("⚡ Caution: Low Hydration" if hr_drift_val > 5 else "✅ Normal: Stable Condition")
    drift_color = "red" if hr_drift_val > 10 else ("orange" if hr_drift_val > 5 else "green")
    
    st.markdown(f"""
    <div class="glow-card" style="margin-top: 10px;">
        <div style="display:flex; justify-content: space-between; align-items:center; margin-bottom: 15px;">
            <span>Average running pace</span>
            <span style="font-size: 1.1rem; font-weight:600; color:#3498db;">{pace_start} /km ➔ {pace_end} /km</span>
        </div>
        <div style="display:flex; justify-content: space-between; align-items:center; margin-bottom: 15px;">
            <span>Average Heart Rate</span>
            <span style="font-size: 1.1rem; font-weight:600; color:#e74c3c;">{hr_start} bpm ➔ {hr_end} bpm</span>
        </div>
        <div style="display:flex; justify-content: space-between; align-items:center; border-top: 1px solid #2d3b55; padding-top: 12px;">
            <span>HR Drift Factor</span>
            <span style="font-size: 1.2rem; font-weight:700; color:{drift_color};">{hr_drift_val:+.1f}%</span>
        </div>
        <div style="text-align: right; margin-top: 8px; font-size: 13px; font-weight: 600; color: {drift_color};">{drift_badge}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="display: flex; gap: 15px; font-size: 13px; color: #f1f5f9;">
        <div style="background:#182030; border: 1px solid #2d3b55; padding: 10px; border-radius: 8px; flex: 1; color: #f1f5f9;">
            Cadence: <strong style="color: #ffffff;">{cadence} spm</strong>
        </div>
        <div style="background:#182030; border: 1px solid #2d3b55; padding: 10px; border-radius: 8px; flex: 1; color: #f1f5f9;">
            Impact Force (GRF): <strong style="color: #ffffff;">{grf:.0f} N</strong>
        </div>
        <div style="background:#182030; border: 1px solid #2d3b55; padding: 10px; border-radius: 8px; flex: 1; color: #f1f5f9;">
            Knee Joint Angle: <strong style="color: #ffffff;">{joint_angles:.0f}°</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

with detail_col2:
    st.subheader("💡 Explainable AI (Risk Analysis)")
    st.write("Key drivers of your running injury risk detected from biomechanics and workload patterns:")
    
    # Generate XAI list based on metrics
    xai_points = []
    
    # ACWR point
    if acwr_val > 1.5:
        xai_points.append(f"<div class='xai-item xai-danger'>🔴 **ACWR very high ({acwr_val:.2f}):** Acute workload increase is too aggressive, exceeding physical adaptation capacity (Danger Zone).</div>")
    elif acwr_val > 1.3:
        xai_points.append(f"<div class='xai-item xai-warn'>🟡 **ACWR slightly high ({acwr_val:.2f}):** You are slightly above the optimal training sweet spot. Exercise caution.</div>")
    elif acwr_val < 0.8:
        xai_points.append(f"<div class='xai-item xai-warn'>🟡 **ACWR too low ({acwr_val:.2f}):** Training volume dropped sharply, risking detraining or cardiorespiratory fitness loss.</div>")
    else:
        xai_points.append(f"<div class='xai-item xai-ok'>🟢 **ACWR Optimal ({acwr_val:.2f}):** Daily and weekly running volume distribution is perfectly balanced (Sweet Spot).</div>")
        
    # Growth point
    if growth_pct > 15.0:
        xai_points.append(f"<div class='xai-item xai-danger'>🔴 **Mileage Growth {growth_pct:+.1f}%:** Weekly volume spike exceeds the safe +10% threshold. Tendon tissues are not adapted to this sudden increase.</div>")
    elif growth_pct > 10.0:
        xai_points.append(f"<div class='xai-item xai-warn'>🟡 **Mileage Growth {growth_pct:+.1f}%:** Volume growth is slightly above the safe limit. Reduce volume next week.</div>")
        
    # HR Drift point
    if hr_drift_val > 10.0:
        xai_points.append(f"<div class='xai-item xai-danger'>🔴 **HR Drift High ({hr_drift_val:+.1f}%):** Heart rate rose significantly despite stable/slower pace. Indicates cardiovascular drift, systemic fatigue, or dehydration.</div>")
    elif hr_drift_val > 5.0:
        xai_points.append(f"<div class='xai-item xai-warn'>🟡 **HR Drift Moderate ({hr_drift_val:+.1f}%):** Initial signs of cardiovascular drift or running in high ambient temperatures.</div>")
        
    # Recovery & sleep point
    if sleep < 5.5:
        xai_points.append(f"<div class='xai-item xai-danger'>🔴 **Low Sleep Quality ({sleep:.1f}/10):** Growth Hormone release is disrupted, impairing muscle tissue recovery.</div>")
    elif recovery < 40.0:
        xai_points.append(f"<div class='xai-item xai-warn'>🟡 **Low Recovery Score ({recovery:.0f}%):** Post-run recovery is suboptimal. Add an extra rest day to your schedule.</div>")
        
    # Biomechanics
    if grf > 1900.0 and cadence < 160:
        xai_points.append(f"<div class='xai-item xai-danger'>🔴 **High Impact + Low Cadence:** Low cadence ({cadence} spm) combined with high impact force ({grf:.0f} N) triggers overstriding, putting excess stress on shins (Shin Splints).</div>")

    # Render points
    for pt in xai_points:
        st.markdown(pt, unsafe_allow_html=True)


# -----------------
# AI COACH ASSISTANT
# -----------------
st.markdown("---")
st.header("Ask PaceGuard AI Running Coach")
st.write("Get deep analysis, recovery training schedules, or explanations for running-related injuries.")

# Chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
for chat in st.session_state.chat_history:
    if chat['role'] == 'user':
        st.markdown(f'<div class="chat-bubble-user">\n\n{chat["text"]}\n\n</div>', unsafe_allow_html=True)
    else:
        # Build stats HTML if stats exist
        stats_html = ""
        if 'stats' in chat and chat['stats']:
            s = chat['stats']
            tokens = s.get('total_tokens', 0)
            cost = s.get('cost_usd', 0.0)
            savings_usd = s.get('savings_usd', 0.0)
            savings_pct = s.get('savings_pct', 0.0)
            
            if tokens > 0:
                stats_html = f'<div style="font-size: 11px; margin-top: 8px; color: #a0aec0; border-top: 1px dashed #2d3b55; padding-top: 6px; display: flex; justify-content: space-between;"><span>🎫 Tokens: <strong>{tokens}</strong></span><span>💰 Cost: <strong>&dollar;{cost:.6f}</strong></span><span style="color: #2ecc71;">⚡ Saved: <strong>{savings_pct:.1f}%</strong> (&dollar;{savings_usd:.5f})</span></div>'

        st.markdown(
            f'<div class="chat-bubble-coach">'
            f'<span class="chat-badge {chat["badge_class"]}">{chat["badge_text"]}</span>'
            f'<div class="chat-text">\n\n{chat["text"]}\n\n</div>'
            f'{stats_html}'
            f'</div>',
            unsafe_allow_html=True
        )

# Query form
with st.form("chat_form", clear_on_submit=True):
    selected_model = st.selectbox("AI Model Selection / Cost Routing:", 
        ["Automatic (Default)", "Gemini 3.5 Flash", "Gemini 3.1 Flash-Lite", "Fireworks DeepSeek V4 Pro", "Fireworks DeepSeek V4 Flash"]
    )
    user_query = st.text_input("Type your question to the Coach here (e.g., 'What should I do next week?' or 'How do I care for knee pain?'):")
    submit_chat = st.form_submit_button("Send Question")

if submit_chat and user_query:
    # Append user question
    st.session_state.chat_history.append({'role': 'user', 'text': user_query, 'model': selected_model})
    
    # Display user question instantly
    st.rerun()

# Rerun to compute AI response if there is a pending user question without coach answer
if len(st.session_state.chat_history) > 0 and st.session_state.chat_history[-1]['role'] == 'user':
    pending_query = st.session_state.chat_history[-1]['text']
    preferred_model = st.session_state.chat_history[-1].get('model', 'Automatic (Default)')
    
    with st.spinner("PaceGuard AI Coach is analyzing your metrics..."):
        # Get AI response
        coach_ans, badge_type, stats = brain.get_coach_response(
            pending_query, 
            metrics_context,
            gemini_key=gemini_key,
            fireworks_key=fireworks_key,
            preferred_model=preferred_model
        )
        
        # Decide badge class
        if "Error" in badge_type or "Guardrail" in badge_type:
            b_class = "badge-local"
            b_text = badge_type
        else:
            b_class = "badge-cloud"
            b_text = badge_type
            
        st.session_state.chat_history.append({
            'role': 'coach',
            'text': coach_ans,
            'badge_text': b_text,
            'badge_class': b_class,
            'stats': stats
        })
        
    st.rerun()

# Suggestion chips below chat input to help user pitch easily
st.markdown("💡 **Recommended Questions for Demo Pitching:**")
col_s1, col_s2, col_s3 = st.columns(3)
if col_s1.button("What is my total distance this week?"):
    st.session_state.chat_history.append({'role': 'user', 'text': "What is my total distance this week?"})
    st.rerun()
if col_s2.button("What does my current ACWR mean?"):
    st.session_state.chat_history.append({'role': 'user', 'text': "What does my current ACWR mean?"})
    st.rerun()
if col_s3.button("Analyze my running data this week and give training advice for next week."):
    st.session_state.chat_history.append({'role': 'user', 'text': "Analyze my running data this week and give training advice for next week."})
    st.rerun()

# Add a clear button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()
