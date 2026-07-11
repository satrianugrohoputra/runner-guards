import sys
import os
import tomllib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import brain

# 1. Load API Keys dari secrets.toml
try:
    with open("secrets.toml", "rb") as f:
        secrets = tomllib.load(f)
    gemini_key = secrets["api_keys"]["GEMINI_API_KEY"]
    fireworks_key = secrets["api_keys"]["FIREWORKS_API_KEY"]
except FileNotFoundError:
    print("Error: File secrets.toml tidak ditemukan!")
    sys.exit(1)
except KeyError:
    print("Error: Struktur [api_keys] atau key di dalam secrets.toml tidak sesuai!")
    sys.exit(1)

metrics_context = {
    'acwr': 1.97,
    'total_kms': 65.0,
    'risk_score': 71.0,
    'growth': 35.0,
    'hr_drift': 12.0,
    'pace_start': "5:10",
    'pace_end': "4:55",
    'sessions': 6,
    'rest_days': 1,
    'recovery_score': 22.0,
    'hydration_level': 70.0,
    'sleep_quality': 4.8,
    'stress_level': 0.75,
    'ground_reaction_force': 2100.0,
    'cadence': 152,
    'data_mode': 'Detail'
}

with open("scratch/output.txt", "w", encoding="utf-8") as f:
    f.write("--- Testing Gemini ---\n")
    try:
        ans_gem, badge_gem, stats_gem = brain.get_coach_response(
            "What does my current ACWR mean?",
            metrics_context,
            gemini_key=gemini_key,
            fireworks_key=fireworks_key,
            preferred_model="Gemini 3.5 Flash"
        )
        f.write(f"Badge: {badge_gem}\n")
        f.write(f"Answer:\n{ans_gem}\n")
    except Exception as e:
        f.write(f"Error: {e}\n")

    f.write("\n--- Testing Fireworks DeepSeek V4 Flash ---\n")
    try:
        ans_fw_f, badge_fw_f, stats_fw_f = brain.get_coach_response(
            "What does my current ACWR mean?",
            metrics_context,
            gemini_key=gemini_key,
            fireworks_key=fireworks_key,
            preferred_model="Fireworks DeepSeek V4 Flash"
        )
        f.write(f"Badge: {badge_fw_f}\n")
        f.write(f"Answer:\n{ans_fw_f}\n")
    except Exception as e:
        f.write(f"Error: {e}\n")

    f.write("\n--- Testing Fireworks DeepSeek V4 Pro ---\n")
    try:
        ans_fw_p, badge_fw_p, stats_fw_p = brain.get_coach_response(
            "What does my current ACWR mean?",
            metrics_context,
            gemini_key=gemini_key,
            fireworks_key=fireworks_key,
            preferred_model="Fireworks DeepSeek V4 Pro"
        )
        f.write(f"Badge: {badge_fw_p}\n")
        f.write(f"Answer:\n{ans_fw_p}\n")
    except Exception as e:
        f.write(f"Error: {e}\n")

print("Done writing results to scratch/output.txt")
