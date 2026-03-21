import streamlit as st
import warnings
import sys, os
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(__file__))

st.set_page_config(
    page_title="Smart Queue Management System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Setup DB on first load ──
try:
    from db import setup_database, test_connection
    setup_database()
    DB_LIVE = test_connection()
except Exception:
    DB_LIVE = False

# ── Global CSS ──
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
    background-color: #F8FAFC;
    color: #1E293B;
}
[data-testid="stAppViewContainer"] { background-color: #F8FAFC; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #1A3A5C 100%);
    border-right: 3px solid #00C6FF;
}
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="metric-container"] {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-left: 4px solid #0066CC;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.stButton > button {
    background: linear-gradient(135deg, #0066CC, #0099FF);
    color: white !important;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 14px;
    transition: all 0.2s;
    box-shadow: 0 2px 8px rgba(0,102,204,0.3);
}
.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(0,102,204,0.4);
}
.sq-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
}
hr { border: none; border-top: 2px solid #E2E8F0; margin: 20px 0; }
[data-testid="stSidebarNav"] a {
    color: #CBD5E1 !important;
    font-size: 14px;
    padding: 8px 12px;
    border-radius: 6px;
    margin: 2px 0;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(0,198,255,0.15) !important;
    color: #00C6FF !important;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:48px;'>🏥</div>
        <div style='font-size:17px; font-weight:700; color:#00C6FF; margin-top:6px;'>Smart Queue</div>
        <div style='font-size:12px; color:#94A3B8; margin-top:2px;'>Management System</div>
        <div style='font-size:10px; color:#64748B; margin-top:8px; background:rgba(0,198,255,0.1);
                    border:1px solid rgba(0,198,255,0.2); border-radius:6px; padding:4px 8px;'>
            Project 2482454
        </div>
    </div>
    <hr style='border-color:#1E3A5C; margin:12px 0;'>
    """, unsafe_allow_html=True)

    db_color  = "#10B981" if DB_LIVE else "#EF4444"
    db_label  = "🟢 MySQL Connected" if DB_LIVE else "🔴 MySQL Offline (session only)"

    st.markdown(f"""
    <hr style='border-color:#1E3A5C; margin:16px 0 8px 0;'>
    <div style='padding:8px 4px; font-size:11px; color:#475569;'>
        <div style='margin-bottom:4px;'>📊 Dataset: 4,000 records</div>
        <div style='margin-bottom:4px;'>🤖 Best Model: Gradient Boosting</div>
        <div style='margin-bottom:4px;'>📈 R² Score: 0.8265</div>
        <div style='margin-bottom:4px;'>⏱ MAE: 16.24 min</div>
        <div style='color:#00C6FF; margin-top:8px;'>OPD Hours: 09:00 – 17:00</div>
        <div style='margin-top:8px; color:{db_color}; font-weight:700;'>{db_label}</div>
    </div>
    """, unsafe_allow_html=True)

# ── HOME PAGE HERO ──
st.markdown("""
<div style='background:linear-gradient(135deg,#0A1628 0%,#1A3A5C 60%,#0066CC 100%);
            padding:48px 40px; border-radius:16px; margin-bottom:32px;
            box-shadow:0 8px 32px rgba(0,102,204,0.2);'>
    <div style='font-size:48px; margin-bottom:8px;'>🏥</div>
    <h1 style='color:white; font-size:32px; font-weight:800; margin:0 0 8px 0;'>
        Smart Queue Management System
    </h1>
    <p style='color:#93C5FD; font-size:16px; margin:0 0 16px 0;'>
        AI-powered OPD patient queue system — Gradient Boosting + LSTM forecasting
    </p>
    <div style='display:flex; gap:12px; flex-wrap:wrap;'>
        <span style='background:rgba(0,198,255,0.15); border:1px solid rgba(0,198,255,0.3);
                     color:#7DD3FC; padding:6px 14px; border-radius:20px; font-size:13px;'>
            ✅ 4,000 Patient Records
        </span>
        <span style='background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.3);
                     color:#6EE7B7; padding:6px 14px; border-radius:20px; font-size:13px;'>
            ✅ R² = 0.8265
        </span>
        <span style='background:rgba(245,158,11,0.15); border:1px solid rgba(245,158,11,0.3);
                     color:#FCD34D; padding:6px 14px; border-radius:20px; font-size:13px;'>
            ✅ LSTM MAE = 8.7 min
        </span>
        <span style='background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.3);
                     color:#FCA5A5; padding:6px 14px; border-radius:20px; font-size:13px;'>
            ✅ 22.4% Wait Reduction
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── DB STATUS BANNER ──
if DB_LIVE:
    st.success("🗄️ MySQL database connected — all patient data will be saved permanently.")
else:
    st.warning("⚠️ MySQL not connected — app works with session memory only. Open `db.py` and set your MySQL credentials to enable persistence.")

# ── PAGE CARDS ──
col1, col2, col3, col4 = st.columns(4)
cards = [
    ("1_Patient_Intake",  "🎫", "Patient Intake",       "Register patients, check OPD hours, generate tokens",       "#0066CC"),
    ("2_Live_Queue",      "📋", "Live Smart Queue",     "Real-time 4-level priority queue with dynamic wait times",  "#059669"),
    ("3_ML_Prediction",   "🤖", "ML Prediction",        "Predict wait time using Gradient Boosting model",           "#7C3AED"),
    ("4_LSTM_Forecast",   "🔮", "LSTM Forecast",        "7-day hourly OPD load and wait time forecast",              "#D97706"),
]
for col, (page, icon, title, desc, color) in zip([col1,col2,col3,col4], cards):
    with col:
        st.markdown(f"""
        <div class='sq-card' style='border-left:5px solid {color}; text-align:center; min-height:160px;'>
            <div style='font-size:36px; margin-bottom:8px;'>{icon}</div>
            <div style='font-size:15px; font-weight:700; color:#1E293B; margin-bottom:6px;'>{title}</div>
            <div style='font-size:12px; color:#64748B; line-height:1.5;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div style='background:#EFF6FF; border:1px solid #BFDBFE; border-radius:10px;
            padding:16px 20px; margin-top:8px;'>
    <div style='font-size:14px; font-weight:600; color:#1E40AF; margin-bottom:8px;'>
        👈 Use the sidebar to navigate between pages
    </div>
    <div style='font-size:13px; color:#3B82F6;'>
        Start with <b>Patient Intake</b> → <b>Live Queue</b> → <b>ML Prediction</b> → <b>LSTM Forecast</b>
    </div>
</div>
""", unsafe_allow_html=True)