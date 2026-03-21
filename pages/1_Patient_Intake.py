import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys, os

# ── Path setup (relative — works on any machine) ──
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_pipeline.pkl")
ENC_PATH   = os.path.join(BASE_DIR, "model", "oe_triage.pkl")
sys.path.append(BASE_DIR)

st.set_page_config(page_title="Patient Intake — Smart Queue", page_icon="🎫", layout="wide")

st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Segoe UI',sans-serif;background:#F8FAFC;color:#1E293B;}
[data-testid="stAppViewContainer"]{background:#F8FAFC;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1628 0%,#1A3A5C 100%);border-right:3px solid #00C6FF;}
[data-testid="stSidebar"] *{color:#E2E8F0 !important;}
.stButton>button{background:linear-gradient(135deg,#0066CC,#0099FF);color:white!important;border:none;
    border-radius:8px;padding:10px 24px;font-weight:600;font-size:14px;
    box-shadow:0 2px 8px rgba(0,102,204,0.3);}
.stButton>button:hover{transform:translateY(-1px);}
[data-testid="metric-container"]{background:#fff;border:1px solid #E2E8F0;
    border-left:4px solid #0066CC;border-radius:8px;padding:12px 16px;
    box-shadow:0 2px 8px rgba(0,0,0,0.06);}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] { color: #0A1628 !important; font-weight: 700 !important; }
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] { color: #0066CC !important; font-weight: 800 !important; }
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"],
.stSelectbox label, .stSlider label,
.stNumberInput label, .stCheckbox label, .stTextInput label {
    color: #0A1628 !important; font-weight: 600 !important; font-size: 13px !important;}
h3 { color: #0066CC !important; border-bottom: 2px solid #E0F0FF; padding-bottom: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──
OPD_START = 9
OPD_END   = 17
EMERGENCY_TRIAGE  = ['Immediate', 'Emergency']
TRIAGE_PRIORITY   = {'Immediate':0,'Emergency':1,'Urgent':2,'Semi-urgent':3,'Non-urgent':4}
AGE_VULNERABILITY = {'Infant (0-1)':2,'Child (2-12)':2,'Senior (61+)':2,
                     'Teenager (13-17)':1,'Young Adult (18-35)':0,'Adult (36-60)':0}
TOKEN_PREFIX      = {'Immediate':'IMM','Emergency':'EMG','Urgent':'URG',
                     'Semi-urgent':'SEM','Non-urgent':'NOU'}
TRIAGE_TARGET     = {'Immediate':'< 5 min','Emergency':'< 15 min',
                     'Urgent':'< 45 min','Semi-urgent':'< 90 min','Non-urgent':'< 180 min'}
TRIAGE_COLOR      = {'Immediate':'#DC2626','Emergency':'#D97706','Urgent':'#CA8A04',
                     'Semi-urgent':'#1D4ED8','Non-urgent':'#059669'}
TRIAGE_EMOJI      = {'Immediate':'🔴','Emergency':'🟠','Urgent':'🟡',
                     'Semi-urgent':'🔵','Non-urgent':'🟢'}
CONSULT_DURATION  = {'Immediate':30,'Emergency':35,'Urgent':25,'Semi-urgent':18,'Non-urgent':12}

# ── Session state ──
if 'token_counters'  not in st.session_state:
    st.session_state.token_counters = {k:0 for k in TOKEN_PREFIX}
if 'intake_history'  not in st.session_state:
    st.session_state.intake_history = []
if 'patient_counter' not in st.session_state:
    st.session_state.patient_counter = 0

# ── DB ──
try:
    from db import insert_patient, fetch_all_patients, count_patients_today, avg_wait_today, test_connection
    DB_LIVE = test_connection()
except Exception:
    DB_LIVE = False

# ── Model loader (relative paths) ──
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

def is_opd_eligible(hour, triage):
    return triage in EMERGENCY_TRIAGE or OPD_START <= hour < OPD_END

def generate_token(triage):
    st.session_state.token_counters[triage] += 1
    return f"{TOKEN_PREFIX[triage]}-{st.session_state.token_counters[triage]:03d}"

# ── PAGE HEADER ──
st.markdown("""
<div style='background:linear-gradient(135deg,#0A1628,#1A3A5C);
            padding:24px 28px;border-radius:12px;margin-bottom:24px;'>
    <h2 style='color:white;margin:0 0 4px 0;font-size:24px;'>🎫 Patient Intake & Token Generation</h2>
    <p style='color:#93C5FD;margin:0;font-size:13px;'>
        Register patient → OPD hours check → ML wait prediction → Token issued → Saved to MySQL
    </p>
</div>
""", unsafe_allow_html=True)

# ── DB + Model status ──
c1, c2, c3, c4 = st.columns(4)
c1.metric("🗄️ MySQL",        "🟢 Connected"   if DB_LIVE     else "🔴 Offline")
c2.metric("🤖 ML Model",     "🟢 Loaded"      if model_loaded else "🟡 Rule-based")
if DB_LIVE:
    c3.metric("👥 Today's Patients", count_patients_today())
    aw = avg_wait_today()
    c4.metric("⏱ Avg Wait Today",   f"{aw} min" if aw else "—")
else:
    c3.metric("📋 Session Patients", len(st.session_state.intake_history))
    c4.metric("OPD Hours",           "09:00 – 17:00")

# ── OPD Status Banner ──
now_hour = datetime.now().hour
opd_open = OPD_START <= now_hour < OPD_END
if opd_open:
    st.markdown(f"""<div style='background:#D1FAE5;border:1px solid #6EE7B7;border-radius:8px;
        padding:10px 16px;margin-bottom:16px;font-weight:600;color:#065F46;font-size:14px;'>
        ✅ OPD is currently OPEN &nbsp;|&nbsp; Hours: 09:00 AM – 05:00 PM &nbsp;|&nbsp;
        Current time: {datetime.now().strftime('%H:%M')}
    </div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div style='background:#FEE2E2;border:1px solid #FCA5A5;border-radius:8px;
        padding:10px 16px;margin-bottom:16px;font-weight:600;color:#991B1B;font-size:14px;'>
        ⛔ OPD is currently CLOSED &nbsp;|&nbsp; Hours: 09:00 AM – 05:00 PM &nbsp;|&nbsp;
        Current time: {datetime.now().strftime('%H:%M')} &nbsp;|&nbsp;
        Emergency/Immediate cases: Emergency Dept open 24/7
    </div>""", unsafe_allow_html=True)

# ── INTAKE FORM ──
with st.form("intake_form", clear_on_submit=False):
    st.markdown("<div style='font-size:16px;font-weight:700;color:#1E293B;margin-bottom:16px;'>📋 Patient Information</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("<p style='color:#0066CC;font-weight:700;font-size:15px;background:#E0F0FF;padding:6px 10px;border-radius:6px;'>🏥 Clinical Details</p>", unsafe_allow_html=True)
        triage_cat     = st.selectbox("Triage Category ⭐", ['Immediate','Emergency','Urgent','Semi-urgent','Non-urgent'])
        age_group      = st.selectbox("Age Group", ['Infant (0-1)','Child (2-12)','Teenager (13-17)',
                                                     'Young Adult (18-35)','Adult (36-60)','Senior (61+)'])
        department     = st.selectbox("Department", ['Emergency','General Medicine','Cardiology',
                                                      'Orthopedics','General Surgery','Pediatrics','Neurology'])
        arrival_method = st.selectbox("Arrival Method", ['Walk-in','Ambulance','Scheduled','Referral'])

    with col2:
        st.markdown("<p style='color:#0066CC;font-weight:700;font-size:15px;background:#E0F0FF;padding:6px 10px;border-radius:6px;'>📋 Appointment Details</p>", unsafe_allow_html=True)
        appt_type  = st.selectbox("Appointment Type", ['Urgent Care','Walk-in','New Patient','Follow-up','Specialist Referral'])
        booking    = st.selectbox("Booking Type",     ['Walk-in','Online','Phone','Referral'])
        insurance  = st.selectbox("Insurance Type",   ['Public','Private','Medicare','Medicaid','Self-pay','Unknown'])
        reason     = st.selectbox("Reason for Visit", ['Chest pain','Breathing difficulty','High fever',
                                                        'Joint pain','Routine check','Injury','Other'])

    with col3:
        st.markdown("<p style='color:#0066CC;font-weight:700;font-size:15px;background:#E0F0FF;padding:6px 10px;border-radius:6px;'>🏨 Arrival & Hospital</p>", unsafe_allow_html=True)
        arrival_hour  = st.slider("Arrival Hour",  0, 23, datetime.now().hour)
        arrival_month = st.slider("Arrival Month", 1, 12, datetime.now().month)
        is_weekend    = st.checkbox("Is Weekend?", value=datetime.now().weekday() >= 5)
        consultation  = st.selectbox("Consultation Needed?", ['TRUE','FALSE'])
        tests         = st.selectbox("Tests Ordered", ['None','Blood test','X-Ray','ECG','CT Scan','MRI','Ultrasound'])

    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    with col4:
        providers    = st.slider("Doctors on Shift",   2, 9,   5)
        nurses       = st.slider("Nurses on Shift",    4, 17, 10)
    with col5:
        occupancy    = st.slider("Occupancy Rate", 0.10, 0.97, 0.57, step=0.01)
        delay_time   = st.slider("Arrival Delay (min)", -45, 40, 0)
    with col6:
        is_registered = st.checkbox("Registered Patient?", True)
        is_online     = st.checkbox("Online Booking?",     False)
        pid_input     = st.text_input("Patient Name / ID (optional)", placeholder="e.g. Ravi Kumar")

    submitted = st.form_submit_button("🎫 Register Patient & Generate Token", use_container_width=True)

if submitted:
    st.session_state.patient_counter += 1
    pid = pid_input.strip() if pid_input.strip() else f"P-{st.session_state.patient_counter:03d}"

    if not is_opd_eligible(arrival_hour, triage_cat):
        st.markdown(f"""
        <div style='background:#FEF2F2;border:2px solid #EF4444;border-radius:12px;padding:20px;margin-top:16px;'>
            <div style='font-size:18px;font-weight:800;color:#DC2626;'>⛔ OPD Closed — Cannot Register</div>
            <div style='font-size:14px;color:#374151;margin-top:8px;'>
                <b>{triage_cat}</b> patients are only accepted during OPD hours (09:00 – 17:00).<br>
                Current arrival hour selected: <b>{arrival_hour:02d}:00</b>
            </div>
            <div style='margin-top:12px;background:#FEE2E2;border-radius:8px;padding:10px;color:#991B1B;font-weight:600;'>
                <b>Action:</b> Please return between 09:00 AM – 05:00 PM<br>
                <b>Emergency cases:</b> Go directly to Emergency Department (open 24/7)
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        token_id   = generate_token(triage_cat)
        ratio      = providers / max((providers + nurses), 1)
        shortage   = 1 if ratio < 0.2 else 0
        hi_occ     = 1 if occupancy > 0.75 else 0
        age_vuln   = AGE_VULNERABILITY.get(age_group, 0)
        triage_num = TRIAGE_PRIORITY.get(triage_cat, 4)
        composite  = (4 - triage_num) * 10 + age_vuln
        consult_min= CONSULT_DURATION.get(triage_cat, 20)

        # ── ML Prediction ──
        pred_wait = None
        if model_loaded:
            try:
                row = {
                    'AgeGroup':age_group,'Department':department,'AppointmentType':appt_type,
                    'InsuranceType':insurance,'ArrivalMethod':arrival_method,'ReasonForVisit':reason,
                    'TestsOrdered':tests,'ConsultationNeeded':consultation,'BookingType':booking,
                    'ArrivalDayOfWeek':'Monday',
                    'TriageSeverityScore':float(oe_triage.transform([[triage_cat]])[0][0]),
                    'AgeVulnerabilityScore':age_vuln,'CompositePriorityScore':composite,
                    'ConsultationDuration_min':consult_min,'ArrivalHour':arrival_hour,
                    'ArrivalMonth':arrival_month,'IsWeekend':int(is_weekend),
                    'ProvidersOnShift':providers,'NursesOnShift':nurses,
                    'StaffToPatientRatio':round(providers/(providers+nurses+0.01),3),
                    'StaffShortage':shortage,'FacilityOccupancyRate':occupancy,
                    'IsHighOccupancy':hi_occ,'IsRegistered':int(is_registered),
                    'IsOnlineBooking':int(is_online),'IsEmergencyDept':int(department=='Emergency'),
                    'ArrivalDelayTime':delay_time,'TriageCategory':triage_cat
                }
                pred_wait = float(model.predict(pd.DataFrame([row]))[0])
                pred_wait = round(max(1.0, pred_wait), 1)
            except Exception:
                pred_wait = None

        if pred_wait is None:
            base      = {'Immediate':35,'Emergency':50,'Urgent':75,'Semi-urgent':112,'Non-urgent':159}
            pred_wait = base.get(triage_cat, 90) + (15 if shortage else 0) + (10 if hi_occ else 0)

        tc        = TRIAGE_COLOR[triage_cat]
        target    = TRIAGE_TARGET[triage_cat]
        mode_note = '(ML Model)' if model_loaded else '(Rule-based estimate)'

        # ── Save to MySQL ──
        db_saved = False
        if DB_LIVE:
            db_saved = insert_patient({
                'token_id':      token_id,
                'patient_id':    pid,
                'triage':        triage_cat,
                'age_group':     age_group,
                'department':    department,
                'appt_type':     appt_type,
                'insurance':     insurance,
                'arrival_method':arrival_method,
                'reason':        reason,
                'tests':         tests,
                'consultation':  consultation,
                'booking_type':  booking,
                'arrival_hour':  arrival_hour,
                'arrival_month': arrival_month,
                'is_weekend':    int(is_weekend),
                'providers':     providers,
                'nurses':        nurses,
                'occupancy':     occupancy,
                'composite_score': composite,
                'age_vuln':      age_vuln,
                'pred_wait_min': pred_wait,
                'staff_shortage':shortage,
                'hi_occupancy':  hi_occ,
            })

        db_badge = (
            "🗄️ <span style='color:#10B981;font-weight:700;'>Saved to MySQL</span>"
            if db_saved else
            "💾 <span style='color:#F59E0B;font-weight:700;'>Session only (MySQL offline)</span>"
        )

        # ── TOKEN RESULT CARD ──
        st.markdown(f"""
        <div style='background:#F0FDF4;border:2px solid #10B981;border-radius:12px;padding:24px;margin-top:16px;'>
            <div style='font-size:18px;font-weight:800;color:#065F46;margin-bottom:4px;'>
                ✅ Token Issued Successfully &nbsp;&nbsp; {db_badge}
            </div>
            <div style='display:flex;align-items:center;gap:20px;margin:16px 0;flex-wrap:wrap;'>
                <div style='background:{tc};color:white;padding:10px 28px;border-radius:24px;
                            font-size:26px;font-weight:800;letter-spacing:3px;
                            box-shadow:0 4px 12px rgba(0,0,0,0.2);'>
                    {token_id}
                </div>
                <div>
                    <div style='font-size:13px;color:#374151;font-weight:600;'>Queue Priority</div>
                    <div style='font-size:18px;font-weight:700;color:{tc};'>
                        {TRIAGE_EMOJI[triage_cat]} {triage_cat} — Target: {target}
                    </div>
                </div>
                <div>
                    <div style='font-size:13px;color:#374151;font-weight:600;'>Predicted Wait {mode_note}</div>
                    <div style='font-size:28px;font-weight:800;color:#0066CC;'>{pred_wait:.0f} min</div>
                </div>
            </div>
            <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;'>
                <div style='background:white;border-radius:8px;padding:12px;border:1px solid #E2E8F0;'>
                    <div style='font-size:11px;color:#374151;font-weight:700;'>Patient ID</div>
                    <div style='font-size:14px;font-weight:700;color:#1E293B;'>{pid}</div>
                </div>
                <div style='background:white;border-radius:8px;padding:12px;border:1px solid #E2E8F0;'>
                    <div style='font-size:11px;color:#374151;font-weight:700;'>Age Group</div>
                    <div style='font-size:14px;font-weight:700;color:#1E293B;'>{age_group}</div>
                </div>
                <div style='background:white;border-radius:8px;padding:12px;border:1px solid #E2E8F0;'>
                    <div style='font-size:11px;color:#374151;font-weight:700;'>Department</div>
                    <div style='font-size:14px;font-weight:700;color:#1E293B;'>{department}</div>
                </div>
                <div style='background:white;border-radius:8px;padding:12px;border:1px solid #E2E8F0;'>
                    <div style='font-size:11px;color:#374151;font-weight:700;'>Composite Score</div>
                    <div style='font-size:14px;font-weight:700;color:{tc};'>{composite}</div>
                </div>
                <div style='background:white;border-radius:8px;padding:12px;border:1px solid #E2E8F0;'>
                    <div style='font-size:11px;color:#374151;font-weight:700;'>Age Vuln. Score</div>
                    <div style='font-size:14px;font-weight:700;color:#1E293B;'>{age_vuln} {"(Vulnerable)" if age_vuln==2 else "(Standard)"}</div>
                </div>
                <div style='background:white;border-radius:8px;padding:12px;border:1px solid #E2E8F0;'>
                    <div style='font-size:11px;color:#374151;font-weight:700;'>Staff Shortage</div>
                    <div style='font-size:14px;font-weight:700;color:{"#DC2626" if shortage else "#059669"}'>
                        {"⚠️ Yes" if shortage else "✅ No"}
                    </div>
                </div>
                <div style='background:white;border-radius:8px;padding:12px;border:1px solid #E2E8F0;'>
                    <div style='font-size:11px;color:#374151;font-weight:700;'>Occupancy</div>
                    <div style='font-size:14px;font-weight:700;color:#1E293B;'>{occupancy:.0%}
                        {"🔴 High" if hi_occ else "✅ Normal"}
                    </div>
                </div>
                <div style='background:white;border-radius:8px;padding:12px;border:1px solid #E2E8F0;'>
                    <div style='font-size:11px;color:#374151;font-weight:700;'>OPD Access</div>
                    <div style='font-size:14px;font-weight:700;color:#1D4ED8;'>
                        {"24/7 Emergency" if triage_cat in EMERGENCY_TRIAGE else "OPD Hours"}
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Session history ──
        st.session_state.intake_history.append({
            'Token': token_id, 'Patient': pid, 'Triage': triage_cat,
            'Age Group': age_group, 'Department': department,
            'Pred Wait (min)': pred_wait, 'Composite Score': composite,
            'Saved to DB': '✅' if db_saved else '💾 Session',
            'Time': datetime.now().strftime('%H:%M:%S')
        })

# ── TODAY'S LOG ──
st.markdown("---")
st.markdown("### 📜 Today's Intake Log")

tab1, tab2 = st.tabs(["💾 Session (This Run)", "🗄️ MySQL (All Time)"])

with tab1:
    if st.session_state.intake_history:
        df_log = pd.DataFrame(st.session_state.intake_history)
        st.dataframe(df_log, use_container_width=True, hide_index=True)
        col_a, col_b = st.columns([1,4])
        with col_a:
            if st.button("🗑️ Clear Session Log"):
                st.session_state.intake_history = []
                st.rerun()
    else:
        st.info("No patients registered this session yet.")

with tab2:
    if DB_LIVE:
        db_rows = fetch_all_patients(limit=100)
        if db_rows:
            df_db = pd.DataFrame(db_rows)
            cols_show = ['token_id','patient_id','triage','age_group','department',
                         'pred_wait_min','composite_score','registered_at']
            df_db = df_db[[c for c in cols_show if c in df_db.columns]]
            st.dataframe(df_db, use_container_width=True, hide_index=True)
            st.caption(f"Showing last {len(df_db)} records from MySQL")
        else:
            st.info("No records in MySQL yet.")
    else:
        st.warning("MySQL not connected. Set credentials in `db.py` to see persistent records.")