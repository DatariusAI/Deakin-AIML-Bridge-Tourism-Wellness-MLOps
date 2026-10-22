"""Visit with Us · Wellness Retreat lead scorer (Streamlit).

Loads the model the GitHub Actions pipeline committed to this folder, collects one
customer's details into a DataFrame, and shows the purchase probability, a lead tier,
recommended next steps and the factors that moved the score.

Paths are resolved relative to this file, so the app runs the same way on
Streamlit Community Cloud, Hugging Face Spaces and locally.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import xgboost as xgb

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "best_tourism_model_v1.joblib"
META_PATH = HERE / "model_metadata.json"

CATEGORICAL = ["TypeofContact", "Occupation", "Gender", "ProductPitched", "MaritalStatus", "Designation"]
LABELS = {
    "Age": "Age", "TypeofContact": "Type of contact", "CityTier": "City tier",
    "DurationOfPitch": "Pitch duration", "Occupation": "Occupation", "Gender": "Gender",
    "NumberOfPersonVisiting": "Party size", "NumberOfFollowups": "Follow-ups",
    "ProductPitched": "Product pitched", "PreferredPropertyStar": "Preferred hotel stars",
    "MaritalStatus": "Marital status", "NumberOfTrips": "Trips per year", "Passport": "Passport",
    "PitchSatisfactionScore": "Pitch satisfaction", "OwnCar": "Owns a car",
    "NumberOfChildrenVisiting": "Children under 5", "Designation": "Designation",
    "MonthlyIncome": "Monthly income",
}

st.set_page_config(page_title="Visit with Us · Wellness Lead Scorer", page_icon="🌿", layout="wide")


# ----------------------------------------------------------------------------- model
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metadata():
    return json.loads(META_PATH.read_text()) if META_PATH.exists() else {}


def top_factors(model, row: pd.DataFrame, k: int = 5):
    """Per-feature contributions (log-odds) from XGBoost, folded back to the original columns."""
    pre, clf = model.steps[0][1], model.steps[-1][1]
    xt = pre.transform(row)
    contrib = clf.get_booster().predict(xgb.DMatrix(xt), pred_contribs=True)[0][:-1]
    names = [n.split("__", 1)[1] for n in pre.get_feature_names_out()]
    totals = {}
    for name, value in zip(names, contrib):
        base = next((c for c in CATEGORICAL if name.startswith(c + "_")), name)
        totals[base] = totals.get(base, 0.0) + float(value)
    return sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:k]


def tier(p: float):
    """Map a probability to a sales tier, a colour and next steps."""
    if p >= 0.70:
        return "Hot lead", "#1E7A55", [
            "Call within 24 hours with the Wellness Retreat offer.",
            "Lead with the package that was pitched and offer a spa or yoga add-on.",
            "Plan up to 5–6 follow-ups. Conversion keeps rising with persistence.",
        ]
    if p >= 0.40:
        return "Warm lead", "#C8871E", [
            "Add to this week's call list after the hot leads.",
            "Send a short wellness itinerary by email before calling.",
            "Try the Basic package as an entry offer if price is the blocker.",
        ]
    return "Cool lead", "#7A8B87", [
        "Keep in the newsletter and seasonal-offer nurture list.",
        "Do not spend a sales call now. Re-score after the next interaction.",
        "Watch for passport renewal or a change in party size.",
    ]


model = load_model()
meta = load_metadata()
threshold = float(meta.get("threshold", 0.5))
metrics = meta.get("metrics", {})

# ----------------------------------------------------------------------------- style
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"], .stMarkdown, label, input, button { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 3.2rem; padding-bottom: 2rem; max-width: 1320px; }
header[data-testid="stHeader"] { background: transparent; }
.hero { position: relative; border-radius: 22px; overflow: hidden; min-height: 230px;
        box-shadow: 0 10px 30px rgba(16, 60, 48, .18); margin-bottom: 1.3rem; }
.hero svg { position: absolute; inset: 0; width: 100%; height: 100%; }
.hero .copy { position: relative; padding: 34px 40px; color: #fff; max-width: 760px; }
.hero .brand { letter-spacing: .22em; text-transform: uppercase; font-size: .78rem; opacity: .92; font-weight: 600; }
.hero h1 { font-family: 'Playfair Display', serif; font-size: 2.55rem; line-height: 1.12; margin: .35rem 0 .55rem; color: #fff; padding: 0; }
.hero p { font-size: 1.02rem; opacity: .95; margin: 0; }
.chips { display: flex; gap: 10px; margin-top: 16px; flex-wrap: wrap; }
.chip { background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35);
        padding: 6px 12px; border-radius: 999px; font-size: .82rem; font-weight: 500; }
.section-title { font-family: 'Playfair Display', serif; font-size: 1.35rem; color: #173F35; margin: .2rem 0 .4rem; }
div[data-testid="stForm"] { background: #FFFFFF; border: 1px solid #DCE8E2; border-radius: 18px; padding: 1.2rem 1.4rem;
        box-shadow: 0 6px 20px rgba(23, 63, 53, .06); }
div[data-testid="stFormSubmitButton"] button { background: linear-gradient(90deg,#1E7A55,#2A9D8F); color: #fff; border: 0;
        padding: .6rem 1.6rem; border-radius: 12px; font-weight: 600; }
div[data-testid="stFormSubmitButton"] button p, div[data-testid="stFormSubmitButton"] button:hover p { color: #fff !important; font-weight: 600; }
div[data-testid="stFormSubmitButton"] button:hover { filter: brightness(1.08); }
.card { background: #FFFFFF; border: 1px solid #DCE8E2; border-radius: 18px; padding: 1.3rem 1.4rem;
        box-shadow: 0 6px 20px rgba(23, 63, 53, .06); margin-bottom: 1rem; }
.empty { text-align: center; color: #5C6F69; padding: 2.4rem 1.2rem; }
.empty .big { font-size: 2.6rem; }
.ring { width: 168px; height: 168px; border-radius: 50%; display: grid; place-items: center; margin: 4px auto 10px; }
.ring .inner { width: 128px; height: 128px; border-radius: 50%; background: #fff; display: grid; place-items: center; text-align: center; }
.ring .pct { font-family: 'Playfair Display', serif; font-size: 2.1rem; font-weight: 700; color: #173F35; line-height: 1; }
.ring .lbl { font-size: .72rem; color: #5C6F69; letter-spacing: .08em; text-transform: uppercase; margin-top: 4px; }
.badge { display: inline-block; color: #fff; font-weight: 600; padding: 6px 14px; border-radius: 999px; font-size: .9rem; }
.verdict { text-align: center; margin-bottom: .4rem; }
.verdict .sub { color: #5C6F69; font-size: .88rem; margin-top: 6px; }
.actions li { margin-bottom: 6px; color: #23362F; }
.factor { display: flex; align-items: center; gap: 10px; margin: 7px 0; font-size: .9rem; color: #23362F; }
.factor .name { width: 128px; flex-shrink: 0; }
.factor .bar { flex: 1; height: 10px; background: #EEF4F1; border-radius: 6px; position: relative; overflow: hidden; }
.factor .fill { position: absolute; top: 0; bottom: 0; border-radius: 6px; }
.factor .val { width: 54px; text-align: right; font-variant-numeric: tabular-nums; color: #5C6F69; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0F3B33 0%,#1E5E4E 100%); }
section[data-testid="stSidebar"] * { color: #F1F7F4 !important; }
.side-kpi { background: rgba(255,255,255,.08); border: 1px solid rgba(255,255,255,.15); border-radius: 12px; padding: 10px 12px; margin-bottom: 8px; }
.side-kpi .v { font-size: 1.35rem; font-weight: 600; }
.side-kpi .k { font-size: .75rem; opacity: .8; text-transform: uppercase; letter-spacing: .06em; }
.foot { color: #7A8B87; font-size: .8rem; text-align: center; margin-top: 1.6rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- hero banner (inline SVG landscape)
HERO_SVG = """
<svg viewBox="0 0 1200 260" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <defs>
    <linearGradient id="sky" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#0E4D64"/><stop offset=".55" stop-color="#2A8C8C"/><stop offset="1" stop-color="#F2B880"/>
    </linearGradient>
    <linearGradient id="lake" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#3A9E9A"/><stop offset="1" stop-color="#0F4C5C"/>
    </linearGradient>
    <linearGradient id="shade" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0" stop-color="#062C2A" stop-opacity=".75"/><stop offset=".65" stop-color="#062C2A" stop-opacity=".15"/>
      <stop offset="1" stop-color="#062C2A" stop-opacity="0"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="260" fill="url(#sky)"/>
  <circle cx="930" cy="150" r="58" fill="#FFD8A0" opacity=".95"/>
  <circle cx="930" cy="150" r="90" fill="#FFD8A0" opacity=".18"/>
  <path d="M0 175 L120 110 L210 150 L330 70 L450 140 L560 95 L690 160 L800 100 L930 165 L1050 90 L1200 150 L1200 260 L0 260Z" fill="#3E7C6B" opacity=".55"/>
  <path d="M0 195 L150 140 L260 180 L400 120 L520 175 L660 130 L790 185 L940 135 L1080 180 L1200 145 L1200 260 L0 260Z" fill="#1F5C4C"/>
  <rect y="205" width="1200" height="55" fill="url(#lake)"/>
  <rect x="860" y="212" width="140" height="3" rx="1.5" fill="#FFE2B5" opacity=".7"/>
  <rect x="885" y="222" width="90" height="2" rx="1" fill="#FFE2B5" opacity=".5"/>
  <g fill="#0F3B33">
    <path d="M1080 205 l18 -48 l18 48z"/><path d="M1105 205 l14 -38 l14 38z"/><path d="M1135 205 l20 -56 l20 56z"/>
    <path d="M40 205 l16 -42 l16 42z"/><path d="M66 205 l12 -32 l12 32z"/>
  </g>
  <path d="M760 120 q10 -8 20 0 q10 -8 20 0" stroke="#0F3B33" stroke-width="2.4" fill="none" opacity=".6"/>
  <path d="M700 92 q7 -6 14 0 q7 -6 14 0" stroke="#0F3B33" stroke-width="2" fill="none" opacity=".5"/>
  <rect width="1200" height="260" fill="url(#shade)"/>
</svg>
"""

chips = ""
if metrics:
    chips = (
        f'<div class="chips"><span class="chip">Model {meta.get("model_version", "v1")} · XGBoost</span>'
        f'<span class="chip">Test F1 {metrics.get("test_f1", 0):.2f}</span>'
        f'<span class="chip">ROC-AUC {metrics.get("test_roc_auc", 0):.2f}</span>'
        f'<span class="chip">Retrained by GitHub Actions</span></div>'
    )

st.markdown(
    f"""<div class="hero">{HERO_SVG}
<div class="copy"><div class="brand">Visit with Us · Wellness Tourism</div>
<h1>Who will book the Wellness Retreat?</h1>
<p>Score a customer before the sales call. The model ranks how likely they are to buy the new
Wellness Tourism Package, so your team spends its time on the right leads.</p>{chips}</div></div>""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------- layout: form left, result right
left, right = st.columns([1.45, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Customer profile</div>', unsafe_allow_html=True)
    with st.form("customer_form", border=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=90, value=35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital_status = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
            city_tier = st.selectbox("City tier", [1, 2, 3])
        with c2:
            occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
            designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
            monthly_income = st.number_input("Monthly income", min_value=0.0, max_value=100000.0, value=22000.0, step=500.0)
            passport = st.selectbox("Has passport", ["Yes", "No"])
        with c3:
            num_person = st.number_input("Persons visiting", min_value=1, max_value=10, value=3)
            num_children = st.number_input("Children under 5", min_value=0, max_value=5, value=1)
            num_trips = st.number_input("Trips per year", min_value=0, max_value=25, value=3)
            own_car = st.selectbox("Owns a car", ["Yes", "No"])

        st.markdown('<div class="section-title" style="margin-top:.6rem">Sales interaction</div>', unsafe_allow_html=True)
        s1, s2, s3 = st.columns(3)
        with s1:
            type_of_contact = st.selectbox("Type of contact", ["Self Enquiry", "Company Invited"])
            product_pitched = st.selectbox("Product pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
        with s2:
            duration_of_pitch = st.number_input("Pitch duration (min)", min_value=0, max_value=150, value=15)
            num_followups = st.number_input("Follow-ups", min_value=0, max_value=10, value=4)
        with s3:
            property_star = st.selectbox("Preferred hotel stars", [3, 4, 5])
            pitch_score = st.slider("Pitch satisfaction", 1, 5, 3)

        submitted = st.form_submit_button("Score this lead  →")

# One-row DataFrame with exactly the training column names; the pipeline does its own scaling/encoding.
input_df = pd.DataFrame([{
    "Age": float(age), "TypeofContact": type_of_contact, "CityTier": int(city_tier),
    "DurationOfPitch": float(duration_of_pitch), "Occupation": occupation, "Gender": gender,
    "NumberOfPersonVisiting": int(num_person), "NumberOfFollowups": float(num_followups),
    "ProductPitched": product_pitched, "PreferredPropertyStar": float(property_star),
    "MaritalStatus": marital_status, "NumberOfTrips": float(num_trips),
    "Passport": 1 if passport == "Yes" else 0, "PitchSatisfactionScore": int(pitch_score),
    "OwnCar": 1 if own_car == "Yes" else 0, "NumberOfChildrenVisiting": float(num_children),
    "Designation": designation, "MonthlyIncome": float(monthly_income),
}])

with right:
    st.markdown('<div class="section-title">Lead score</div>', unsafe_allow_html=True)
    if not submitted:
        st.markdown(
            '<div class="card empty"><div class="big">🌴</div>'
            "<b>No score yet.</b><br>Fill in the customer profile and press "
            "<b>Score this lead</b>. The result appears here.</div>",
            unsafe_allow_html=True,
        )
    else:
        proba = float(model.predict_proba(input_df)[0, 1])
        name, color, actions = tier(proba)
        decision = "Likely to buy" if proba >= threshold else "Unlikely to buy now"
        deg = int(round(proba * 360))
        st.markdown(
            f"""<div class="card">
<div class="ring" style="background: conic-gradient({color} {deg}deg, #E7EFEB {deg}deg);">
  <div class="inner"><div><div class="pct">{proba:.0%}</div><div class="lbl">purchase chance</div></div></div></div>
<div class="verdict"><span class="badge" style="background:{color}">{name}</span>
<div class="sub">{decision} · exact score {proba:.1%} · decision threshold {threshold:.0%}</div></div>
<div class="section-title" style="font-size:1.05rem;margin-top:.8rem">Recommended next steps</div>
<ul class="actions">{''.join(f'<li>{a}</li>' for a in actions)}</ul></div>""",
            unsafe_allow_html=True,
        )

        factors = top_factors(model, input_df)
        scale = max(abs(v) for _, v in factors) or 1.0
        rows = ""
        for feat, val in factors:
            width = abs(val) / scale * 50
            side = (f"left:50%;width:{width:.0f}%;background:#1E7A55" if val > 0
                    else f"right:50%;width:{width:.0f}%;background:#C2593F")
            arrow = "▲" if val > 0 else "▼"
            rows += (f'<div class="factor"><div class="name">{LABELS.get(feat, feat)}</div>'
                     f'<div class="bar"><div class="fill" style="{side}"></div></div>'
                     f'<div class="val">{arrow} {abs(val):.2f}</div></div>')
        st.markdown(
            f'<div class="card"><div class="section-title" style="font-size:1.05rem">What drove this score</div>'
            f'<div style="color:#5C6F69;font-size:.82rem;margin-bottom:6px">Green raises the chance to buy, '
            f'red lowers it (XGBoost contributions, log-odds).</div>{rows}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("Input sent to the model"):
            shown = input_df.T.rename(columns={0: "value"}).astype(str)
            shown.index = [LABELS.get(i, i) for i in shown.index]
            st.dataframe(shown, use_container_width=True, height=300)

# ----------------------------------------------------------------------------- sidebar model card
with st.sidebar:
    st.markdown("### 🌿 Model card")
    if meta:
        st.markdown(f"**Version {meta.get('model_version', '?')}** · XGBoost pipeline")
        st.caption(f"Trained {meta.get('trained_at_utc', '?').replace('T', ' ')[:16]} UTC by GitHub Actions")
        for key, label in [("test_f1", "Test F1"), ("test_recall", "Recall (buyers found)"),
                           ("test_precision", "Precision"), ("test_roc_auc", "ROC-AUC")]:
            st.markdown(f'<div class="side-kpi"><div class="v">{metrics.get(key, 0):.3f}</div>'
                        f'<div class="k">{label}</div></div>', unsafe_allow_html=True)
        st.caption("Calling the top 20% of scored leads reaches 86% of buyers in the test set.")

st.markdown(
    '<div class="foot">Visit with Us · Wellness Tourism Package lead scorer · Deakin Applied AI Master Bridge Program</div>',
    unsafe_allow_html=True,
)
