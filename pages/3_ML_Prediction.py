import streamlit as st
import pandas as pd
import numpy as np
import os, sys
import warnings
from sklearn.exceptions import InconsistentVersionWarning

# ── Relative path setup (works on any machine) ──
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_pipeline.pkl")
ENC_PATH   = os.path.join(BASE_DIR, "model", "oe_triage.pkl")
sys.path.append(BASE_DIR)

warnings.simplefilter("ignore", InconsistentVersionWarning)
st.set_page_config(page_title="ML Prediction — Smart Queue", page_icon="🤖", layout="wide")

st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Segoe UI',sans-serif;background:#F8FAFC;color:#1E293B;}
[data-testid="stAppViewContainer"]{background:#F8FAFC;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1628 0%,#1A3A5C 100%);border-right:3px solid #00C6FF;}
[data-testid="stSidebar"] *{color:#E2E8F0 !important;}
.stButton>button{background:linear-gradient(135deg,#0066CC,#0099FF);color:white!important;border:none;
    border-radius:8px;padding:8px 20px;font-weight:600;box-shadow:0 2px 8px rgba(0,102,204,0.3);}

/* Metric cards */
[data-testid="metric-container"]{
    background:#fff !important;border:1px solid #E2E8F0;
    border-left:4px solid #0066CC;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] { color: #0A1628 !important; font-weight: 700 !important; }
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] { color: #0066CC !important; font-weight: 800 !important; }

/* All field labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"],
.stSelectbox label, .stSlider label,
.stNumberInput label, .stCheckbox label, .stTextInput label {
    color: #0A1628 !important; font-weight: 600 !important; font-size: 13px !important;}

/* Section headings */
h3, h4 { color: #0066CC !important; border-bottom: 2px solid #E0F0FF; padding-bottom: 6px; }

/* Expander headers */
[data-testid="stExpander"] summary p {
    color: #0A1628 !important; font-weight: 700 !important; font-size: 14px !important;}

/* Alert/info boxes */
[data-testid="stAlert"] p { color: #1E293B !important; font-weight: 600 !important; }
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──
TRIAGE_PRIORITY   = {'Immediate':0,'Emergency':1,'Urgent':2,'Semi-urgent':3,'Non-urgent':4}
AGE_VULNERABILITY = {'Infant (0-1)':2,'Child (2-12)':2,'Senior (61+)':2,
                     'Teenager (13-17)':1,'Young Adult (18-35)':0,'Adult (36-60)':0}
TRIAGE_COLOR = {'Immediate':'#DC2626','Emergency':'#D97706','Urgent':'#CA8A04',
                'Semi-urgent':'#1D4ED8','Non-urgent':'#059669'}
TRIAGE_EMOJI = {'Immediate':'🔴','Emergency':'🟠','Urgent':'🟡','Semi-urgent':'🔵','Non-urgent':'🟢'}
CONSULT_DURATION = {'Immediate':30,'Emergency':35,'Urgent':25,'Semi-urgent':18,'Non-urgent':12}
TRIAGE_TARGET = {'Immediate':'< 5 min','Emergency':'< 15 min','Urgent':'< 45 min',
                 'Semi-urgent':'< 90 min','Non-urgent':'< 180 min'}

# ── Load model ──
@st.cache_resource
def load_model():
    try:
        import joblib
        if os.path.exists(MODEL_PATH) and os.path.exists(ENC_PATH):
            return joblib.load(MODEL_PATH), joblib.load(ENC_PATH), True
    except Exception:
        pass
    return None, None, False

model, oe_triage, model_loaded = load_model()

def rule_based_predict(triage, arrival_hour, providers, nurses, occupancy):
    base = {'Immediate':35,'Emergency':50,'Urgent':75,'Semi-urgent':112,'Non-urgent':159}
    wait = base.get(triage, 90)
    shortage = (providers / max(providers+nurses,1)) < 0.2
    if shortage:  wait += 15
    if occupancy > 0.75: wait += 12
    if arrival_hour in [8,9,10,11]: wait += 8
    if arrival_hour in [17,18,19]:  wait += 5
    return round(wait, 1)

def build_features(triage, age_group, dept, appt_type, insurance, arrival_method,
                   reason, tests, consultation, booking, arrival_hour, arrival_month,
                   is_weekend, providers, nurses, occupancy, delay_time,
                   is_registered, is_online, oe_triage):
    age_vuln  = AGE_VULNERABILITY.get(age_group, 0)
    triage_num= TRIAGE_PRIORITY.get(triage, 4)
    composite = (4 - triage_num) * 10 + age_vuln
    consult   = CONSULT_DURATION.get(triage, 20)
    ratio     = providers / max(providers+nurses, 1)
    shortage  = 1 if ratio < 0.2 else 0
    hi_occ    = 1 if occupancy > 0.75 else 0
    triage_score = float(oe_triage.transform([[triage]])[0][0])
    row = {
        'AgeGroup':age_group,'Department':dept,'AppointmentType':appt_type,
        'InsuranceType':insurance,'ArrivalMethod':arrival_method,'ReasonForVisit':reason,
        'TestsOrdered':tests,'ConsultationNeeded':consultation,'BookingType':booking,
        'ArrivalDayOfWeek':'Monday','TriageSeverityScore':triage_score,
        'AgeVulnerabilityScore':age_vuln,'CompositePriorityScore':composite,
        'ConsultationDuration_min':consult,'ArrivalHour':arrival_hour,
        'ArrivalMonth':arrival_month,'IsWeekend':int(is_weekend),
        'ProvidersOnShift':providers,'NursesOnShift':nurses,
        'StaffToPatientRatio':round(ratio,3),'StaffShortage':shortage,
        'FacilityOccupancyRate':occupancy,'IsHighOccupancy':hi_occ,
        'IsRegistered':int(is_registered),'IsOnlineBooking':int(is_online),
        'IsEmergencyDept':int(dept=='Emergency'),'ArrivalDelayTime':delay_time,
        'TriageCategory':triage
    }
    return pd.DataFrame([row]), composite, age_vuln

# ══════════════════════════════
# PAGE HEADER
# ══════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#3B0764,#6D28D9);
            padding:24px 28px;border-radius:12px;margin-bottom:24px;'>
    <h2 style='color:white;margin:0 0 4px 0;font-size:24px;'>🤖 ML Wait Time Prediction</h2>
    <p style='color:#DDD6FE;margin:0;font-size:13px;'>
        Gradient Boosting model &nbsp;|&nbsp; R² = 0.8265 &nbsp;|&nbsp; MAE = 16.24 min &nbsp;|&nbsp; 6 models compared
    </p>
</div>
""", unsafe_allow_html=True)

form_col, result_col = st.columns([1, 1], gap="large")

with form_col:
    st.markdown("### 🔧 Patient Scenario")

    with st.expander("🏥 Clinical Details", expanded=True):
        triage_cat = st.selectbox("Triage Category", ['Urgent','Semi-urgent','Non-urgent','Immediate','Emergency'])
        age_group  = st.selectbox("Age Group", ['Adult (36-60)','Young Adult (18-35)','Senior (61+)',
                                                 'Child (2-12)','Infant (0-1)','Teenager (13-17)'])
        department = st.selectbox("Department", ['General Medicine','Emergency','Cardiology',
                                                  'Orthopedics','General Surgery','Pediatrics','Neurology'])
        appt_type  = st.selectbox("Appointment Type", ['Walk-in','Urgent Care','New Patient','Follow-up','Specialist Referral'])

    with st.expander("⏰ Arrival Details", expanded=True):
        arrival_hour  = st.slider("Arrival Hour", 0, 23, 10, help="Peak hours: 9–11 AM and 4–6 PM")
        arrival_month = st.slider("Month", 1, 12, 3)
        is_weekend    = st.checkbox("Weekend?")
        delay_time    = st.slider("Arrival Delay (min)", -45, 40, 0)

    with st.expander("👥 Hospital Conditions", expanded=True):
        providers = st.slider("Doctors on Shift",  2, 9,  5)
        nurses    = st.slider("Nurses on Shift",   4, 17, 10)
        occupancy = st.slider("Occupancy Rate", 0.10, 0.97, 0.57, step=0.01, help="> 75% = High Occupancy")

    with st.expander("📋 Admin Details"):
        insurance      = st.selectbox("Insurance",      ['Public','Private','Medicare','Self-pay'])
        arrival_method = st.selectbox("Arrival",        ['Walk-in','Ambulance','Scheduled','Referral'])
        reason         = st.selectbox("Reason",         ['Chest pain','High fever','Joint pain','Routine check','Injury','Other'])
        tests          = st.selectbox("Tests",          ['None','Blood test','X-Ray','ECG','CT Scan','MRI'])
        consultation   = st.selectbox("Consultation?",  ['TRUE','FALSE'])
        booking        = st.selectbox("Booking",        ['Walk-in','Online','Phone','Referral'])
        is_registered  = st.checkbox("Registered Patient?", True)
        is_online      = st.checkbox("Online Booking?", False)

    predict_btn = st.button("🤖 Predict Wait Time", use_container_width=True)

# ══════════════════════════════
# RESULTS
# ══════════════════════════════
with result_col:
    st.markdown("### 📊 Prediction Result")

    if predict_btn or True:
        if model_loaded and predict_btn:
            try:
                df_input, composite, age_vuln = build_features(
                    triage_cat, age_group, department, appt_type, insurance,
                    arrival_method, reason, tests, consultation, booking,
                    arrival_hour, arrival_month, is_weekend, providers, nurses,
                    occupancy, delay_time, is_registered, is_online, oe_triage)
                pred_wait = round(float(model.predict(df_input)[0]), 1)
                pred_wait = max(1.0, pred_wait)
                mode = "ML Model (Gradient Boosting)"
            except Exception as e:
                pred_wait = rule_based_predict(triage_cat, arrival_hour, providers, nurses, occupancy)
                mode = f"Rule-based (model error: {e})"
                composite = (4-TRIAGE_PRIORITY.get(triage_cat,4))*10 + AGE_VULNERABILITY.get(age_group,0)
                age_vuln  = AGE_VULNERABILITY.get(age_group,0)
        else:
            pred_wait = rule_based_predict(triage_cat, arrival_hour, providers, nurses, occupancy)
            mode = "Rule-based estimate (load model for ML)"
            composite = (4-TRIAGE_PRIORITY.get(triage_cat,4))*10 + AGE_VULNERABILITY.get(age_group,0)
            age_vuln  = AGE_VULNERABILITY.get(age_group,0)

        tc      = TRIAGE_COLOR[triage_cat]
        em      = TRIAGE_EMOJI[triage_cat]
        target  = TRIAGE_TARGET[triage_cat]
        shortage= (providers / max(providers+nurses,1)) < 0.2
        hi_occ  = occupancy > 0.75

        if pred_wait <= 30:
            wait_color='#059669'; wait_label='Low Wait'
        elif pred_wait <= 90:
            wait_color='#D97706'; wait_label='Moderate Wait'
        else:
            wait_color='#DC2626'; wait_label='High Wait'

        # ── MAIN RESULT CARD ──
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#F8FAFC,#EFF6FF);
                    border:2px solid {tc};border-radius:14px;padding:24px;margin-bottom:16px;
                    box-shadow:0 4px 20px rgba(0,0,0,0.08);'>
            <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                <div>
                    <div style='font-size:12px;color:#374151;margin-bottom:4px;
                                text-transform:uppercase;letter-spacing:1px;font-weight:600;'>
                        Predicted Wait Time
                    </div>
                    <div style='font-size:52px;font-weight:900;color:{wait_color};line-height:1;'>
                        {pred_wait:.0f}
                        <span style='font-size:20px;font-weight:400;color:#374151;'>min</span>
                    </div>
                    <div style='margin-top:6px;'>
                        <span style='background:{wait_color};color:white;padding:3px 10px;
                                     border-radius:10px;font-size:12px;font-weight:600;'>
                            {wait_label}
                        </span>
                    </div>
                </div>
                <div style='text-align:right;'>
                    <div style='font-size:28px;font-weight:800;color:{tc};'>
                        {em} {triage_cat}
                    </div>
                    <div style='font-size:13px;color:#374151;margin-top:4px;font-weight:600;'>Target: {target}</div>
                    <div style='font-size:11px;color:#374151;margin-top:8px;'>{mode}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── SCORE CARDS ──
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Priority Score", composite)
        sc2.metric("Age Vuln.",      f"{age_vuln} {'🔴' if age_vuln==2 else '✅'}")
        sc3.metric("Staff Shortage", "⚠️ Yes" if shortage else "✅ No")
        sc4.metric("Occupancy",      f"{occupancy:.0%} {'🔴' if hi_occ else '✅'}")

        # ── SCENARIO BREAKDOWN ──
        st.markdown("---")
        st.markdown("#### 📋 Why This Prediction?")

        factors = []
        base_vals = {'Immediate':35,'Emergency':50,'Urgent':75,'Semi-urgent':112,'Non-urgent':159}
        base_w = base_vals.get(triage_cat, 90)
        factors.append(("Triage base wait",            f"+{base_w} min",   "#0066CC", base_w))
        if shortage:
            factors.append(("Staff shortage penalty",  "+15 min",           "#DC2626", 15))
        if hi_occ:
            factors.append(("High occupancy penalty",  "+12 min",           "#D97706", 12))
        if arrival_hour in [9,10,11]:
            factors.append(("Peak morning hours",      "+8 min",            "#7C3AED", 8))
        if age_vuln == 2:
            factors.append(("Vulnerable patient priority", "Priority boost","#059669", 0))

        for factor, val, color, _ in factors:
            st.markdown(f"""
            <div style='display:flex;justify-content:space-between;align-items:center;
                        background:white;border:1px solid #E2E8F0;border-left:4px solid {color};
                        border-radius:8px;padding:10px 16px;margin-bottom:8px;'>
                <span style='font-size:13px;color:#1E293B;font-weight:600;'>{factor}</span>
                <span style='font-size:14px;font-weight:700;color:{color};'>{val}</span>
            </div>
            """, unsafe_allow_html=True)

        # ── WHAT-IF ANALYSIS ──
        st.markdown("---")
        st.markdown("#### 🔍 What-If Comparison")
        scenarios = [
            ("Current scenario",             pred_wait, tc),
            ("If 2 more doctors added",
             rule_based_predict(triage_cat, arrival_hour, providers+2, nurses, occupancy), "#059669"),
            ("If occupancy drops to 50%",
             rule_based_predict(triage_cat, arrival_hour, providers, nurses, 0.50), "#0066CC"),
            ("If arrived 2 hrs earlier",
             rule_based_predict(triage_cat, max(0,arrival_hour-2), providers, nurses, occupancy), "#7C3AED"),
        ]
        for label, val, color in scenarios:
            diff = val - pred_wait
            diff_str = f"({'−' if diff<0 else '+' if diff>0 else ''}{abs(diff):.0f} min)" if diff != 0 else "(baseline)"
            bar_w = min(int((val / 200)*100), 100)
            st.markdown(f"""
            <div style='margin-bottom:10px;'>
                <div style='display:flex;justify-content:space-between;margin-bottom:3px;'>
                    <span style='font-size:12px;color:#1E293B;font-weight:600;'>{label}</span>
                    <span style='font-size:13px;font-weight:700;color:{color};'>
                        {val:.0f} min &nbsp;<span style='font-size:11px;color:#374151;font-weight:600;'>{diff_str}</span>
                    </span>
                </div>
                <div style='background:#F1F5F9;border-radius:4px;height:10px;overflow:hidden;'>
                    <div style='width:{bar_w}%;background:{color};height:100%;border-radius:4px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── MODEL COMPARISON TABLE ──
        st.markdown("---")
        st.markdown("#### 🏆 Model Comparison (Training Results)")
        model_data = {
            'Model':['Gradient Boosting ⭐','Random Forest','XGBoost','Ridge Regression','SVR','Linear Regression'],
            'R² Score':[0.8265,0.7841,0.8102,0.5213,0.4987,0.4801],
            'MAE (min)':[16.24,18.91,17.33,28.74,31.20,32.88],
            'Status':['✅ Best','','','','','']
        }
        df_models = pd.DataFrame(model_data)
        st.dataframe(df_models, use_container_width=True, hide_index=True)