import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

st.set_page_config(page_title="Live Queue — Smart Queue", page_icon="📋", layout="wide")

st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Segoe UI',sans-serif;background:#F8FAFC;color:#1E293B;}
[data-testid="stAppViewContainer"]{background:#F8FAFC;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1628 0%,#1A3A5C 100%);border-right:3px solid #00C6FF;}
[data-testid="stSidebar"] *{color:#E2E8F0 !important;}
.stButton>button{background:linear-gradient(135deg,#0066CC,#0099FF);color:white!important;border:none;
    border-radius:8px;padding:8px 20px;font-weight:600;box-shadow:0 2px 8px rgba(0,102,204,0.3);}
[data-testid="metric-container"]{
    background:#fff !important;border:1px solid #E2E8F0;
    border-left:4px solid #0066CC;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] { color: #0A1628 !important; font-weight: 700 !important; }
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] { color: #0066CC !important; font-weight: 800 !important;
    font-size: 1.2rem !important; white-space: nowrap !important; }
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"],
.stSelectbox label, .stSlider label,
.stNumberInput label, .stCheckbox label, .stTextInput label {
    color: #0A1628 !important; font-weight: 600 !important; font-size: 13px !important;}
h3 { color: #0066CC !important; border-bottom: 2px solid #E0F0FF; padding-bottom: 6px; }
[data-testid="stAlert"] p,
.stSuccess p, .stError p, .stWarning p, .stInfo p {
    color: #1E293B !important; font-weight: 600 !important;}
</style>
""", unsafe_allow_html=True)

# ── Constants ──
OPD_START=9; OPD_END=17
EMERGENCY_TRIAGE=['Immediate','Emergency']
TRIAGE_PRIORITY={'Immediate':0,'Emergency':1,'Urgent':2,'Semi-urgent':3,'Non-urgent':4}
AGE_VULNERABILITY={'Infant (0-1)':2,'Child (2-12)':2,'Senior (61+)':2,
                   'Teenager (13-17)':1,'Young Adult (18-35)':0,'Adult (36-60)':0}
TOKEN_PREFIX={'Immediate':'IMM','Emergency':'EMG','Urgent':'URG','Semi-urgent':'SEM','Non-urgent':'NOU'}
CONSULT_DURATION={'Immediate':30,'Emergency':35,'Urgent':25,'Semi-urgent':18,'Non-urgent':12}
TRIAGE_TARGET={'Immediate':'< 5 min','Emergency':'< 15 min','Urgent':'< 45 min',
               'Semi-urgent':'< 90 min','Non-urgent':'< 180 min'}
TRIAGE_COLOR={'Immediate':'#DC2626','Emergency':'#D97706','Urgent':'#CA8A04',
              'Semi-urgent':'#1D4ED8','Non-urgent':'#059669'}
TRIAGE_EMOJI={'Immediate':'🔴','Emergency':'🟠','Urgent':'🟡','Semi-urgent':'🔵','Non-urgent':'🟢'}
TRIAGE_BG={'Immediate':'#FEF2F2','Emergency':'#FFFBEB','Urgent':'#FEFCE8',
           'Semi-urgent':'#EFF6FF','Non-urgent':'#F0FDF4'}

# ── DB ──
try:
    from db import insert_served_patient, fetch_queue_history, count_served_today, test_connection
    DB_LIVE = test_connection()
except Exception:
    DB_LIVE = False

def is_opd_eligible(hour, triage):
    return triage in EMERGENCY_TRIAGE or OPD_START <= hour < OPD_END

# ── Session state ──
if 'queue_tokens'    not in st.session_state: st.session_state.queue_tokens    = []
if 'served_tokens'   not in st.session_state: st.session_state.served_tokens   = []
if 'rejected_tokens' not in st.session_state: st.session_state.rejected_tokens = []
if 'token_counters'  not in st.session_state: st.session_state.token_counters  = {k:0 for k in TOKEN_PREFIX}
if 'doctor_free_at'  not in st.session_state:
    st.session_state.doctor_free_at = datetime.now().replace(hour=9,minute=0,second=0,microsecond=0)
if 'clinic_date'     not in st.session_state:
    st.session_state.clinic_date = datetime.now().replace(hour=9,minute=0,second=0,microsecond=0)
if 'patient_counter' not in st.session_state: st.session_state.patient_counter = 0

def recalculate_dynamic_waits():
    q = sorted(st.session_state.queue_tokens,
               key=lambda t:(TRIAGE_PRIORITY.get(t['triage'],5),
                              -AGE_VULNERABILITY.get(t['age_group'],0),
                              t['ml_wait'], t['arrival_time']))
    cumulative = 0.0
    call_time  = st.session_state.doctor_free_at
    for p in q:
        p['dynamic_wait']    = round(cumulative, 1)
        p['expected_call_at']= call_time.strftime('%H:%M')
        cumulative += CONSULT_DURATION.get(p['triage'], 20)
        call_time  += timedelta(minutes=CONSULT_DURATION.get(p['triage'],20))
    st.session_state.queue_tokens = q

def add_patient(triage, age_group, dept, ml_wait, arrival_hour):
    today    = st.session_state.clinic_date.date()
    arr_time = st.session_state.clinic_date.replace(hour=arrival_hour, minute=0)
    if arr_time.date() != today:
        return False, "Date mismatch"
    if not is_opd_eligible(arrival_hour, triage):
        st.session_state.patient_counter += 1
        st.session_state.token_counters[triage] += 1
        tok_id = f"{TOKEN_PREFIX[triage]}-{st.session_state.token_counters[triage]:03d}"
        st.session_state.rejected_tokens.append({
            'token':tok_id,'triage':triage,'age_group':age_group,
            'dept':dept,'arrival_hour':arrival_hour
        })
        return False, f"OPD CLOSED at {arrival_hour:02d}:00 — {triage} not accepted outside OPD hours"
    st.session_state.patient_counter += 1
    st.session_state.token_counters[triage] += 1
    tok_id   = f"{TOKEN_PREFIX[triage]}-{st.session_state.token_counters[triage]:03d}"
    age_vuln  = AGE_VULNERABILITY.get(age_group, 0)
    composite = (4 - TRIAGE_PRIORITY.get(triage,4)) * 10 + age_vuln
    token = {
        'token':tok_id, 'patient':f'P-{st.session_state.patient_counter:03d}',
        'triage':triage, 'age_group':age_group, 'dept':dept,
        'ml_wait':float(ml_wait), 'age_vuln':age_vuln, 'composite':composite,
        'arrival_time':arr_time, 'dynamic_wait':0.0, 'expected_call_at':'—', 'status':'WAITING'
    }
    st.session_state.queue_tokens.append(token)
    recalculate_dynamic_waits()
    return True, tok_id

def serve_next_patient(actual_duration):
    if not st.session_state.queue_tokens:
        return None, "Queue is empty"
    token = st.session_state.queue_tokens.pop(0)
    token['status'] = 'SERVED'
    diff  = CONSULT_DURATION.get(token['triage'],20) - actual_duration
    st.session_state.doctor_free_at += timedelta(minutes=actual_duration)
    st.session_state.served_tokens.append(token)
    # ── Save to MySQL ──
    if DB_LIVE:
        insert_served_patient(token)
    recalculate_dynamic_waits()
    return token, diff

# ── PAGE HEADER ──
st.markdown("""
<div style='background:linear-gradient(135deg,#0A1628,#1A3A5C);
            padding:24px 28px;border-radius:12px;margin-bottom:24px;'>
    <h2 style='color:white;margin:0 0 4px 0;font-size:24px;'>📋 Live Smart Priority Queue</h2>
    <p style='color:#93C5FD;margin:0;font-size:13px;'>
        4-level ordering: Triage → Age Vulnerability → ML Wait → Arrival Time &nbsp;|&nbsp;
        Served patients auto-saved to MySQL
    </p>
</div>
""", unsafe_allow_html=True)

# ── TOP METRICS ──
doctor_free_str = st.session_state.doctor_free_at.strftime('%H:%M')
clinic_date_str = st.session_state.clinic_date.strftime('%d-%b-%Y')

m1, m2, m3, m4, m5, m6 = st.columns(6)

def metric_card(col, label, value, color="#0066CC"):
    col.markdown(f"""
    <div style='background:white;border:1px solid #E2E8F0;border-left:4px solid {color};
                border-radius:8px;padding:14px 16px;box-shadow:0 2px 8px rgba(0,0,0,0.05);'>
        <div style='font-size:11px;font-weight:700;color:#374151;margin-bottom:4px;'>{label}</div>
        <div style='font-size:18px;font-weight:800;color:{color};white-space:nowrap;'>{value}</div>
    </div>
    """, unsafe_allow_html=True)

metric_card(m1, "🩺 In Queue",        len(st.session_state.queue_tokens),    "#0066CC")
metric_card(m2, "✅ Served (session)", len(st.session_state.served_tokens),   "#059669")
metric_card(m3, "⛔ Rejected",         len(st.session_state.rejected_tokens), "#DC2626")
metric_card(m4, "🕐 Doctor Free At",   doctor_free_str,                       "#7C3AED")
metric_card(m5, "📅 Clinic Date",       clinic_date_str,                       "#D97706")
metric_card(m6, "🗄️ MySQL",           "🟢 ON" if DB_LIVE else "🔴 OFF",      "#059669" if DB_LIVE else "#EF4444")

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

left, right = st.columns([1, 2])

with left:
    st.markdown("### ➕ Add Patient to Queue")
    with st.form("add_patient_form"):
        t_triage = st.selectbox("Triage Category", ['Immediate','Emergency','Urgent','Semi-urgent','Non-urgent'])
        t_age    = st.selectbox("Age Group", ['Adult (36-60)','Young Adult (18-35)','Senior (61+)',
                                               'Child (2-12)','Infant (0-1)','Teenager (13-17)'])
        t_dept   = st.selectbox("Department", ['Emergency','General Medicine','Cardiology',
                                                'Orthopedics','General Surgery','Pediatrics','Neurology'])
        t_hour   = st.slider("Arrival Hour", 0, 23, datetime.now().hour)
        t_wait   = st.number_input("ML Predicted Wait (min)", min_value=1.0, max_value=220.0, value=90.0, step=1.0)
        add_btn  = st.form_submit_button("➕ Add to Queue", use_container_width=True)

    if add_btn:
        success, msg = add_patient(t_triage, t_age, t_dept, t_wait, t_hour)
        if success:
            st.success(f"✅ Token **{msg}** issued! — {t_triage}")
        else:
            st.error(f"⛔ {msg}")

    st.markdown("---")
    st.markdown("### 🔔 Serve Next Patient")
    with st.form("serve_form"):
        default_triage = st.session_state.queue_tokens[0]['triage'] if st.session_state.queue_tokens else 'Urgent'
        actual_dur = st.number_input("Actual Consultation (min)", min_value=1, max_value=120,
                                      value=CONSULT_DURATION.get(default_triage, 25))
        serve_btn = st.form_submit_button("🔔 Call Next Patient", use_container_width=True)

    if serve_btn:
        token, diff = serve_next_patient(actual_dur)
        if token:
            tc = TRIAGE_COLOR[token['triage']]
            db_note = " &nbsp;|&nbsp; <b style='color:#10B981;'>🗄️ Saved to MySQL</b>" if DB_LIVE else ""
            st.markdown(f"""
            <div style='background:{TRIAGE_BG[token["triage"]]};border:2px solid {tc};
                        border-radius:10px;padding:16px;margin-top:8px;'>
                <div style='font-size:18px;font-weight:800;color:{tc};'>
                    {TRIAGE_EMOJI[token["triage"]]} NOW SERVING: {token["token"]}
                </div>
                <div style='font-size:13px;color:#374151;margin-top:8px;'>
                    Patient: {token["patient"]} &nbsp;|&nbsp; {token["triage"]} &nbsp;|&nbsp;
                    {token["age_group"]}<br>Department: {token["dept"]}{db_note}
                </div>
                {"<div style='margin-top:8px;color:#059669;font-weight:600;'>⚡ Doctor finished "+str(abs(diff))+" min EARLY</div>" if diff>0 else ""}
                {"<div style='margin-top:8px;color:#DC2626;font-weight:600;'>⏳ Doctor ran "+str(abs(diff))+" min LATE</div>" if diff<0 else ""}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Queue is empty")

    st.markdown("---")
    if st.button("🔄 Reset Queue", use_container_width=True):
        for key in ['queue_tokens','served_tokens','rejected_tokens']:
            st.session_state[key] = []
        st.session_state.token_counters = {k:0 for k in TOKEN_PREFIX}
        st.session_state.doctor_free_at = datetime.now().replace(hour=9,minute=0,second=0,microsecond=0)
        st.session_state.patient_counter = 0
        st.rerun()

with right:
    st.markdown("### 📊 Current Queue")
    if not st.session_state.queue_tokens:
        st.info("Queue is empty. Add patients using the form on the left.")
    else:
        for i, p in enumerate(st.session_state.queue_tokens, 1):
            tc  = TRIAGE_COLOR[p['triage']]
            bg  = TRIAGE_BG[p['triage']]
            em  = TRIAGE_EMOJI[p['triage']]
            tgt = TRIAGE_TARGET[p['triage']]
            vuln_txt = " 👶" if p['age_group'] in ['Infant (0-1)','Child (2-12)'] else " 👴" if p['age_group']=='Senior (61+)' else ""
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {tc};border-left:5px solid {tc};
                        border-radius:10px;padding:14px 18px;margin-bottom:10px;
                        display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;'>
                <div style='display:flex;align-items:center;gap:14px;'>
                    <div style='font-size:20px;font-weight:800;color:#64748B;min-width:24px;'>#{i}</div>
                    <div style='background:{tc};color:white;padding:5px 14px;border-radius:16px;
                                font-size:15px;font-weight:700;letter-spacing:1px;'>{p["token"]}</div>
                    <div>
                        <div style='font-size:14px;font-weight:700;color:{tc};'>{em} {p["triage"]}{vuln_txt}</div>
                        <div style='font-size:12px;color:#6B7280;'>{p["age_group"]} &nbsp;|&nbsp; {p["dept"]} &nbsp;|&nbsp; Score: {p["composite"]}</div>
                    </div>
                </div>
                <div style='display:flex;gap:20px;align-items:center;flex-wrap:wrap;'>
                    <div style='text-align:center;'>
                        <div style='font-size:11px;color:#6B7280;'>ML Wait</div>
                        <div style='font-size:15px;font-weight:700;color:#1D4ED8;'>{p["ml_wait"]:.0f}m</div>
                    </div>
                    <div style='text-align:center;'>
                        <div style='font-size:11px;color:#6B7280;'>Dynamic Wait</div>
                        <div style='font-size:15px;font-weight:700;color:#7C3AED;'>{p["dynamic_wait"]:.0f}m</div>
                    </div>
                    <div style='text-align:center;'>
                        <div style='font-size:11px;color:#6B7280;'>Call At</div>
                        <div style='font-size:15px;font-weight:700;color:#059669;'>{p["expected_call_at"]}</div>
                    </div>
                    <div style='text-align:center;'>
                        <div style='font-size:11px;color:#6B7280;'>Target</div>
                        <div style='font-size:12px;font-weight:600;color:{tc};'>{tgt}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Priority Summary")
    from collections import Counter
    counts = Counter(p['triage'] for p in st.session_state.queue_tokens)
    for level in ['Immediate','Emergency','Urgent','Semi-urgent','Non-urgent']:
        cnt     = counts.get(level, 0)
        tc      = TRIAGE_COLOR[level]
        em      = TRIAGE_EMOJI[level]
        note    = '(24/7)' if level in EMERGENCY_TRIAGE else '(OPD hrs only)'
        bar_pct = min(cnt * 20, 100)
        st.markdown(f"""
        <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;'>
            <div style='width:130px;font-size:13px;font-weight:600;color:{tc};'>{em} {level}</div>
            <div style='flex:1;background:#F1F5F9;border-radius:4px;height:16px;overflow:hidden;'>
                <div style='width:{bar_pct}%;background:{tc};height:100%;border-radius:4px;'></div>
            </div>
            <div style='width:70px;font-size:13px;color:#374151;font-weight:600;text-align:right;'>
                {cnt} patient{"s" if cnt!=1 else ""}
            </div>
            <div style='width:80px;font-size:11px;color:#6B7280;'>{note}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── MySQL History Tab ──
    st.markdown("---")
    tab1, tab2 = st.tabs(["✅ Served (Session)", "🗄️ MySQL History"])

    with tab1:
        if st.session_state.served_tokens:
            df_served = pd.DataFrame([{
                'Token':p['token'],'Patient':p['patient'],'Triage':p['triage'],
                'Age Group':p['age_group'],'Department':p['dept'],'ML Wait':p['ml_wait']
            } for p in st.session_state.served_tokens])
            st.dataframe(df_served, use_container_width=True, hide_index=True)
        else:
            st.info("No patients served this session yet.")

    with tab2:
        if DB_LIVE:
            db_rows = fetch_queue_history(limit=100)
            if db_rows:
                df_db = pd.DataFrame(db_rows)
                cols_show = ['token_id','patient_id','triage','age_group','department',
                             'ml_wait_min','dynamic_wait','status','served_at']
                df_db = df_db[[c for c in cols_show if c in df_db.columns]]
                st.dataframe(df_db, use_container_width=True, hide_index=True)
                st.caption(f"Total served in MySQL: {len(db_rows)}")
            else:
                st.info("No records in MySQL yet.")
        else:
            st.warning("MySQL not connected. Set credentials in `db.py`.")

    if st.session_state.rejected_tokens:
        st.markdown("---")
        st.markdown("### ⛔ Turned Away (OPD Closed)")
        for r in st.session_state.rejected_tokens:
            st.markdown(f"""
            <div style='background:#FEF2F2;border:1px solid #FCA5A5;border-radius:8px;
                        padding:10px 14px;margin-bottom:6px;font-size:13px;'>
                <b style='color:#DC2626;'>{r["token"]}</b> &nbsp;|&nbsp;
                {TRIAGE_EMOJI[r["triage"]]} {r["triage"]} &nbsp;|&nbsp;
                {r["age_group"]} &nbsp;|&nbsp; Arrived: {r["arrival_hour"]:02d}:00 &nbsp;|&nbsp;
                <i style='color:#6B7280;'>Please return 09:00–17:00</i>
            </div>
            """, unsafe_allow_html=True)