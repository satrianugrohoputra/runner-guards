import requests
import re

# =============================================
# LLM MODEL CONFIGURATION
# =============================================
GEMINI_LITE_MODEL = "gemini-3.1-flash-lite"
GEMINI_MODEL  = "gemini-3.5-flash"
GEMINI_PRO_MODEL  = GEMINI_MODEL

# Fireworks Models (for Token-Efficient Cost Routing)
# Both verified WORKING (serverless, no deployment needed) with current API key:

FIREWORKS_LITE_MODEL = "accounts/fireworks/models/deepseek-v4-flash"
FIREWORKS_PRO_MODEL  = "accounts/fireworks/models/deepseek-v4-pro"
FIREWORKS_BASE_URL   = "https://api.fireworks.ai/inference/v1"

# =============================================
# GUARDRAIL: List of FORBIDDEN off-topic subjects
# =============================================
OFF_TOPIC_PATTERNS = [
    r'\b(code|python|javascript|program|algorithm|function|class|import|def|html|css|sql|database|api|backend|frontend)\b',
    r'\b(recipe|cook|cooking|food|noodle|rice|dish|ingredient|kitchen|bake|fry)\b',
    r'\b(politics|party|president|election|news|law|government)\b',
    r'\b(film|movie|song|music|drama|actor|celebrity|tiktok|youtube|instagram)\b',
    r'\b(business|stock|crypto|investment|forex|property|price|sell|buy|marketing)\b',
]

# =============================================
# HALLUCINATION GUARD
# =============================================
HALLUCINATION_MARKERS = [
    "as a language model", "i do not have access", "cannot access the internet",
    "my data is limited", "i am chatgpt", "i am claude", "openai",
    "i do not know specific data", "not relevant to running",
    "```python", "```javascript", "```html", "def ", "import os",
]

# =============================================
# SYSTEM PROMPT — Structured & Strict
# =============================================
SYSTEM_PROMPT = """You are **PaceGuard AI**, an exclusive scientific data-driven AI Running Coach specializing in running injury prevention.

=== IDENTITY & ROLE ===
- You are strictly a running coach and sports scientist.
- You analyze biomechanical data and runners' training patterns.
- You communicate in a friendly, empathetic, and professional English.

=== HARD CONSTRAINTS (MUST ADHERE) ===
- DO NOT answer questions outside the scope of running, sports injuries, biomechanics, recovery, or runner nutrition.
- DO NOT write programming code (Python, JavaScript, etc.) in any form.
- DO NOT provide general cooking recipes.
- DO NOT talk about politics, economics, entertainment, or other general topics.
- DO NOT pretend to be another AI (ChatGPT, Claude, etc.).
- DO NOT claim capabilities outside the realm of sports science.
- DO NOT invent data or statistics that are not present in the provided runner data.

=== OUTPUT INSTRUCTIONS ===
If runner data is provided, ALWAYS use that data as the basis for your analysis.
If a question is irrelevant, politely decline and steer it back to running.
Formatting: Use headings, bullet points, and relevant emojis for readability.
Length: Concise yet comprehensive (maximum of 4 main sections).
"""

# =============================================
# GUARDRAIL CHECKER
# =============================================
def check_input_guardrail(query: str) -> tuple[bool, str]:
    q = query.lower().strip()

    if len(q.split()) < 2:
        return True, "⚠️ Query is too short. Please ask something more specific about your training or running condition."

    for pattern in OFF_TOPIC_PATTERNS:
        if re.search(pattern, q, re.IGNORECASE):
            return True, (
                "🚫 **Out of PaceGuard AI Context**\n\n"
                "Sorry, I can only help with topics related to:\n"
                "- 🏃 Running pattern analysis & injuries\n"
                "- 📊 Interpretation of ACWR, HR Drift, Risk Score\n"
                "- 💪 Training programs & recovery\n"
                "- 🦵 Prevention of knee pain, shin splints, hamstring issues, etc.\n\n"
                "Please ask a question related to running and your physical condition!"
            )

    injection_patterns = [
        r'(ignore|bypass).*(instructions|prompt|system)',
        r'(pretend|act as|be a).*(gpt|claude|other ai|chatbot)',
        r'(forget).*(role|identity|persona)',
    ]
    for pat in injection_patterns:
        if re.search(pat, q, re.IGNORECASE):
            return True, (
                "🔒 **Request Cannot Be Processed**\n\n"
                "I am PaceGuard AI, your exclusive running assistant. "
                "I cannot change my role or identity.\n\n"
                "Do you have any questions about your running training program? 😊"
            )

    return False, ""

def check_output_hallucination(text: str) -> str:
    for marker in HALLUCINATION_MARKERS:
        if marker.lower() in text.lower():
            return (
                "⚠️ **[PaceGuard AI - Response Filtered]**\n\n"
                "The AI response was detected to contain content outside the sports/running domain. "
                "Please rephrase your question with a more specific running context, "
                "or select another model from the dropdown.\n\n"
                "_Tip: Try asking questions like 'What should I do next week?'_"
            )

    text = re.sub(r'```[\s\S]*?```', '[Code content removed - outside PaceGuard AI scope]', text)
    return text


# =============================================
# TOKEN & COST SAVINGS CALCULATOR
# =============================================
def get_savings_stats(total_tokens, actual_cost, is_error=False):
    """
    Calculates cost efficiency metrics.
    Baseline is running deepseek-v4-flash ($0.90 per million tokens).
    """
    baseline_tokens = total_tokens if total_tokens > 0 else 1300
    baseline_cost = baseline_tokens * 0.0000009 # $0.90/M

    if is_error:
        actual_cost = 0.0
        total_tokens = 0

    savings_usd = max(0.0, baseline_cost - actual_cost)
    savings_pct = (savings_usd / baseline_cost) * 100 if baseline_cost > 0 else 0.0

    return {
        "total_tokens": total_tokens,
        "cost_usd": actual_cost,
        "savings_usd": savings_usd,
        "savings_pct": savings_pct
    }


# =============================================
# QUERY CLASSIFIER
# =============================================
def classify_query(query):
    """
    Routes query intelligently to either cloud_lite or cloud_pro 
    to minimize token cost while ensuring accuracy.
    """
    q = query.lower()

    # Simple questions or specific statistics: Lite model is sufficient (cheaper)
    lite_keywords = [
        "my distance", "total km", "mileage", "how many km",
        "risk score", "injury risk", "what is acwr", "what is hr drift",
        "what is zone 2", "can i", "is it", "when", "safe", "cadence", "statistics"
    ]
    if any(kw in q for kw in lite_keywords) or len(q.split()) < 15:
        return 'cloud_lite'

    # Complex analytical questions: use Pro model
    return 'cloud'


# =============================================
# PROMPT BUILDER — Structured & Grounded (Advanced Prompt Engineering)
# =============================================
def _get_dynamic_tone_instructions(risk_score, acwr):
    """Dynamically sets coaching tone based on risk and training load."""
    if risk_score > 65 or acwr > 1.5:
        return (
            "🔴 **TONE: URGENT & PROTECTIVE**\n"
            "The runner is in a CRITICAL injury danger zone. Be direct, authoritative, and firm. "
            "Prioritize injury prevention and immediate workload reduction above all else. "
            "Advise strongly against any mileage increases."
        )
    elif risk_score > 35 or acwr > 1.3:
        return (
            "🟡 **TONE: ADVISORY & CAUTIOUS**\n"
            "The runner is showing early fatigue signals. Be highly analytical, informative, and encouraging. "
            "Focus on stabilization, active recovery, and moderate adjustments. Alert them to watch key indicators."
        )
    else:
        return (
            "🟢 **TONE: POSITIVE & PROGRESSIVE**\n"
            "The runner has safe metrics. Be encouraging, motivating, and focus on safe progression. "
            "Highlight the consistency of their metrics and guide them on how to build aerobic volume safely."
        )

def _build_metrics_prompt(query, metrics):
    risk_score = metrics.get('risk_score', 0.0)
    acwr = metrics.get('acwr', 1.0)
    data_mode = metrics.get('data_mode', 'Detail')
    risk_ctx = ("HIGH — needs serious attention" if risk_score > 60 else
                ("MEDIUM — needs monitoring" if risk_score > 35 else "LOW — safe condition"))
    acwr_ctx = ("DANGER (overtraining)" if acwr > 1.5 else
                ("CAUTION (approaching limit)" if acwr > 1.3 else
                 ("UNDER-TRAINING" if acwr < 0.8 else "OPTIMAL")))

    mode_note = ""
    if data_mode == "Quick":
        mode_note = ("[NOTE: This runner's data was entered in QUICK MODE — biomechanical data like cadence, "
                     "GRF, ROM are automatic ESTIMATES based on reported physical conditions, not precise measurements. "
                     "Adjust your analysis depth accordingly and inform the runner that for more accurate analysis, "
                     "Detail Mode is recommended.]\n\n")

    tone_instruction = _get_dynamic_tone_instructions(risk_score, acwr)

    # Few-Shot Example
    few_shot_example = (
        "=== FEW-SHOT REFERENCE ANALYSIS ===\n"
        "QUERY: 'My shins hurt during runs, what is going on?'\n"
        "RESPONSE:\n"
        "## 🦵 Shin Stress Analysis (Shin Splints)\n"
        "Based on your metrics, you have a **Risk Score of 68/100 (HIGH)**, and your Ground Reaction Force is extremely high at **2,100 N** combined with a slow cadence of **155 spm**. This indicates overstriding (taking strides that are too long), which significantly increases the impact stress on your tibia bones.\n\n"
        "### 📋 Recommended Recovery Plan (Next 7 Days):\n"
        "- **Days 1-3:** Complete rest from running. Focus on cross-training (e.g. 20-30 min easy cycling) and apply ice to the shins for 15 mins twice daily.\n"
        "- **Days 4-7:** If pain-free, resume very light runs of max 3km, focusing strictly on increasing your cadence to **170-175 spm** to reduce impact force.\n\n"
        "Keep your head high; adjusting your stride rate is the key to running further and staying injury-free!\n"
        "====================================\n\n"
    )

    data_block = (
        f"{mode_note}"
        f"{tone_instruction}\n\n"
        f"=== RUNNER BIOMETRIC DATA ===\n"
        f"Input Mode       : {data_mode} Mode\n"
        f"Weekly Mileage   : {metrics.get('total_kms', 0.0):.1f} km (Growth: {metrics.get('growth', 0.0):+.1f}%)\n"
        f"Running Sessions : {metrics.get('sessions', 0):.0f}x | Rest Days: {metrics.get('rest_days', 0):.0f} days\n"
        f"ACWR             : {acwr:.2f} → {acwr_ctx}\n"
        f"HR Drift         : {metrics.get('hr_drift', 0.0):+.1f}%\n"
        f"Average Pace     : {metrics.get('pace_start', 'N/A')} → {metrics.get('pace_end', 'N/A')} /km\n"
        f"Risk Score       : {risk_score:.0f}/100 → {risk_ctx}\n"
        f"Recovery Score   : {metrics.get('recovery_score', 80.0):.0f}%\n"
        f"Sleep Quality    : {metrics.get('sleep_quality', 7.0):.1f}/10\n"
        f"Stress Level     : {metrics.get('stress_level', 0.3):.0f}%\n"
        f"Impact (GRF)     : {metrics.get('ground_reaction_force', 1500.0):.0f} N{' (estimated)' if data_mode == 'Quick' else ''}\n"
        f"Cadence          : {metrics.get('cadence', 160.0):.0f} spm{' (estimated)' if data_mode == 'Quick' else ''}\n\n"
        f"{few_shot_example}"
        f"=== RUNNER QUESTION ===\n"
        f"{query}\n\n"
        f"=== ANALYSIS INSTRUCTIONS (CHAIN-OF-THOUGHT) ===\n"
        f"Follow this step-by-step thinking process to construct your response:\n"
        f"STEP 1 [Data Evaluation]: Cross-reference the runner's question with the provided biometric metrics. Cite exact numbers.\n"
        f"STEP 2 [Root Cause Analysis]: Identify the primary metric driver behind their symptoms or risk (e.g. ACWR, Growth, GRF/Cadence mismatch).\n"
        f"STEP 3 [Action Plan Formulation]: Outline a concrete, detailed 7-day training adjust schedule.\n"
        f"STEP 4 [Final Review]: Ensure tone matches the dynamic tone instructions. Verify that NO coding/programming or cooking recipes are included.\n"
    )
    return data_block


# =============================================
# GEMINI API
# =============================================
def ask_gemini(prompt, api_key, use_lite=False):
    model = GEMINI_LITE_MODEL if use_lite else GEMINI_PRO_MODEL

    def make_request(model_name):
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model_name}:generateContent?key={api_key}")
        payload = {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.65, "maxOutputTokens": 1100}
        }
        return requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=45)

    try:
        response = make_request(model)
        if response.status_code != 200 and model == GEMINI_PRO_MODEL:
            try:
                fb = make_request(GEMINI_LITE_MODEL)
                if fb.status_code == 200:
                    response, model = fb, GEMINI_LITE_MODEL
            except Exception:
                pass

        if response.status_code == 200:
            res_json = response.json()
            text = res_json['candidates'][0]['content']['parts'][0]['text']
            text = check_output_hallucination(text)

            usage = res_json.get("usageMetadata", {})
            total_tokens = usage.get("totalTokenCount", 0)

            actual_cost = total_tokens * 0.000000075
            stats = get_savings_stats(total_tokens, actual_cost, is_error=False)

            return text, f"Gemini {model}", stats
        else:
            err = response.json().get('error', {}).get('message', response.text)
            return f"❌ Gemini Error ({response.status_code}): {err}", "Gemini Error", get_savings_stats(0, 0.0, is_error=True)
    except Exception as e:
        return f"❌ Gemini connection failed: {str(e)}", "Gemini Error", get_savings_stats(0, 0.0, is_error=True)


# =============================================
# FIREWORKS API (With Custom Model Routing)
# =============================================
def ask_fireworks(prompt, api_key, model_name=FIREWORKS_PRO_MODEL):
    url = f"{FIREWORKS_BASE_URL}/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.65,
        "max_tokens": 900
    }
    try:
        r = requests.post(url, json=payload,
                          headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                          timeout=30)
        
        if r.status_code == 200:
            res_json = r.json()
            text = res_json['choices'][0]['message']['content']
            text = check_output_hallucination(text)

            usage = res_json.get("usage", {})
            total_tokens = usage.get("total_tokens", 0)

            rate = 0.0000002 if "accounts/fireworks/models/deepseek-v4-pro" in model_name else 0.0000009
            actual_cost = total_tokens * rate
            stats = get_savings_stats(total_tokens, actual_cost, is_error=False)

            short_name = model_name.split("/")[-1]
            return text, f"Fireworks {short_name}", stats
        else:
            err = r.json().get('error', {}).get('message', r.text)
            if r.status_code == 404:
                short = model_name.split("/")[-1]
                return (
                    f"Model `{short}` was not found or is inaccessible (Error 404).\n\n"
                    f"Please ensure your Fireworks API Key is active and has credits.\n\n"
                    f"Alternative: Select a **Gemini** model from the dropdown below.",
                    "Fireworks Error",
                    get_savings_stats(0, 0.0, is_error=True)
                )
            return f"❌ Fireworks Error ({r.status_code}): {err}", "Fireworks Error", get_savings_stats(0, 0.0, is_error=True)
    except Exception as e:
        return f"❌ Fireworks connection failed: {str(e)}", "Fireworks Error", get_savings_stats(0, 0.0, is_error=True)


# =============================================
# MAIN ORCHESTRATOR
# =============================================
def get_coach_response(query, metrics, gemini_key=None, fireworks_key=None, preferred_model='Automatic (Default)'):
    """
    Main entry point:
    1. Input Guardrail
    2. Explicit Model Bypass
    3. LLM AI Routing (Hybrid Token-Efficient Routing to Cloud Lite or Cloud Pro)
    4. Output Hallucination Guard
    """
    # ── Step 1: Input Guardrail ──
    is_blocked, block_msg = check_input_guardrail(query)
    if is_blocked:
        return block_msg, "🛡️ Guardrail", get_savings_stats(0, 0.0, is_error=True)

    has_gemini = bool(gemini_key and len(gemini_key.strip()) > 10)
    has_fireworks = bool(fireworks_key and len(fireworks_key.strip()) > 10)

    # ── Step 2: Explicit Model Bypass ──
    if preferred_model == "Gemini 3.5 Flash" and has_gemini:
        prompt = _build_metrics_prompt(query, metrics)
        return ask_gemini(prompt, gemini_key, use_lite=False)

    if preferred_model == "Gemini 3.1 Flash-Lite" and has_gemini:
        prompt = _build_metrics_prompt(query, metrics)
        return ask_gemini(prompt, gemini_key, use_lite=True)

    if preferred_model == "Fireworks DeepSeek V4 Pro" and has_fireworks:
        prompt = _build_metrics_prompt(query, metrics)
        return ask_fireworks(prompt, fireworks_key, model_name="accounts/fireworks/models/deepseek-v4-pro")

    if preferred_model == "Fireworks DeepSeek V4 Flash" and has_fireworks:
        prompt = _build_metrics_prompt(query, metrics)
        return ask_fireworks(prompt, fireworks_key, model_name="accounts/fireworks/models/deepseek-v4-flash")

    # Error handling if manual selection has no valid key
    if "Fireworks" in preferred_model and not has_fireworks:
        return (
            "❌ **Empty Fireworks API Key**\n\nPlease enter your Fireworks API Key in the left panel to use this model.",
            "Key Missing Error",
            get_savings_stats(0, 0.0, is_error=True)
        )
    if "Gemini" in preferred_model and not has_gemini:
        return (
            "❌ **Empty Gemini API Key**\n\nPlease enter your Gemini API Key in the left panel to use this model.",
            "Key Missing Error",
            get_savings_stats(0, 0.0, is_error=True)
        )

    # If no API key is provided at all, return a warning
    if not has_gemini and not has_fireworks:
        return (
            "⚠️ **No API Key Provided**\n\nAI mode cannot run without an API key. Please enter an API Key to try the AI Coach features.",
            "Key Missing Error",
            get_savings_stats(0, 0.0, is_error=True)
        )

    # ── Step 3: LLM AI Routing (Cost Optimizer) ──
    route = classify_query(query)
    full_prompt = _build_metrics_prompt(query, metrics)

    if route == 'cloud_lite':
        if has_fireworks:
            return ask_fireworks(full_prompt, fireworks_key, model_name=FIREWORKS_LITE_MODEL)
        if has_gemini:
            return ask_gemini(full_prompt, gemini_key, use_lite=True)

    # default: cloud (full/pro)
    if has_fireworks:
        return ask_fireworks(full_prompt, fireworks_key, model_name=FIREWORKS_PRO_MODEL)
    if has_gemini:
        return ask_gemini(full_prompt, gemini_key, use_lite=False)

    return "⚠️ An error occurred during the API routing process.", "Routing Error", get_savings_stats(0, 0.0, is_error=True)
