import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="LSTM Forecast — Smart Queue", page_icon="🔮", layout="wide")

st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Segoe UI',sans-serif;background:#F8FAFC;color:#1E293B;}
[data-testid="stAppViewContainer"]{background:#F8FAFC;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0A1628 0%,#1A3A5C 100%);border-right:3px solid #00C6FF;}
[data-testid="stSidebar"] *{color:#E2E8F0 !important;}
.stButton>button{background:linear-gradient(135deg,#D97706,#F59E0B);color:white!important;border:none;
    border-radius:8px;padding:8px 20px;font-weight:600;box-shadow:0 2px 8px rgba(217,119,6,0.3);}

/* Metric cards */
[data-testid="metric-container"]{
    background:#fff !important;border:1px solid #E2E8F0;
    border-left:4px solid #D97706;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] { color: #0A1628 !important; font-weight: 700 !important; }
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] { color: #D97706 !important; font-weight: 800 !important; }

/* All field labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"],
.stSelectbox label, .stSlider label,
.stNumberInput label, .stCheckbox label, .stTextInput label {
    color: #0A1628 !important; font-weight: 600 !important; font-size: 13px !important;}

/* Section headings */
h3, h4 { color: #D97706 !important; border-bottom: 2px solid #FEF3C7; padding-bottom: 6px; }

/* Bold text in markdown */
strong { color: #1E293B !important; }
</style>
""", unsafe_allow_html=True)

# ── VERIFIED FORECAST DATA ──
FORECAST_DATA = [
    {'day':'Saturday',  'date':'2024-06-29','patients':20,'wait':92.3,'load':'🟡 Moderate'},
    {'day':'Sunday',    'date':'2024-06-30','patients':20,'wait':91.7,'load':'🟡 Moderate'},
    {'day':'Monday',    'date':'2024-07-01','patients':19,'wait':91.7,'load':'🟡 Moderate'},
    {'day':'Tuesday',   'date':'2024-07-02','patients':20,'wait':92.2,'load':'🟡 Moderate'},
    {'day':'Wednesday', 'date':'2024-07-03','patients':20,'wait':92.1,'load':'🟡 Moderate'},
    {'day':'Thursday',  'date':'2024-07-04','patients':20,'wait':92.5,'load':'🟡 Moderate'},
    {'day':'Friday',    'date':'2024-07-05','patients':20,'wait':92.6,'load':'🟡 Moderate'},
]

FCFS_AVG   = 92.2
SMART_AVG  = 71.5
SAVING_MIN = 20.7
SAVING_PCT = 22.4

HOURLY_PROFILE = {
    'Saturday':  [1,0,0,0,0,0,0,1,2,4,5,4,3,3,3,3,2,2,1,1,0,0,0,0],
    'Sunday':    [1,0,0,0,0,0,0,1,2,3,5,4,3,3,3,2,2,2,1,1,0,0,0,0],
    'Monday':    [0,0,0,0,0,0,1,2,3,5,5,4,4,4,3,3,2,2,1,0,0,0,0,0],
    'Tuesday':   [0,0,0,0,0,0,1,2,3,5,5,4,4,4,3,3,2,2,1,0,0,0,0,0],
    'Wednesday': [0,0,0,0,0,0,1,2,3,5,5,4,4,4,3,3,2,2,1,0,0,0,0,0],
    'Thursday':  [0,0,0,0,0,0,1,2,3,5,5,4,4,4,3,3,2,2,1,0,0,0,0,0],
    'Friday':    [0,0,0,0,0,0,1,2,3,4,5,4,3,3,3,3,3,2,1,1,0,0,0,0],
}

STAFF_ADVICE = [
    {'segment':'07:00 – 09:00','intensity':'Low',   'color':'#059669','bg':'#F0FDF4','advice':'Pre-opening prep. 2–3 staff. Ready for emergency walk-ins.'},
    {'segment':'09:00 – 12:00','intensity':'High',  'color':'#DC2626','bg':'#FEF2F2','advice':'Peak morning rush. 5–6 doctors + 10 nurses required. Triage at entrance.'},
    {'segment':'12:00 – 14:00','intensity':'Medium','color':'#D97706','bg':'#FFFBEB','advice':'Moderate. 3–4 staff. Overlap lunch breaks in rotation.'},
    {'segment':'14:00 – 17:00','intensity':'Medium','color':'#D97706','bg':'#FFFBEB','advice':'Steady afternoon flow. 4 doctors. Monitor occupancy.'},
    {'segment':'17:00 – 19:00','intensity':'Low',   'color':'#1D4ED8','bg':'#EFF6FF','advice':'OPD closing. Emergency only. 1–2 on-call doctors.'},
    {'segment':'19:00 – 07:00','intensity':'Night', 'color':'#6B7280','bg':'#F9FAFB','advice':'Emergency/Immediate only (24/7). Minimal staff 1–2 on duty.'},
]

# ══════════════════════════════
# PAGE HEADER
# ══════════════════════════════
st.markdown("""
<div style='background:linear-gradient(135deg,#78350F,#D97706);
            padding:24px 28px;border-radius:12px;margin-bottom:24px;'>
    <h2 style='color:white;margin:0 0 4px 0;font-size:24px;'>🔮 LSTM 7-Day Forecast</h2>
    <p style='color:#FDE68A;margin:0;font-size:13px;'>
        Bidirectional LSTM · 122,210 parameters · Look-back: 7 days ·
        PatientCount MAE: 4.0 · AvgWaitTime MAE: 8.7 min
    </p>
</div>
""", unsafe_allow_html=True)

# ── LSTM MODEL STATS ──
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("Architecture",    "LSTM(128→64)")
m2.metric("Parameters",      "122,210")
m3.metric("Patient MAE",     "4.0 patients")
m4.metric("Wait Time MAE",   "8.7 min")
m5.metric("Training Epochs", "18 (early stop)")

st.markdown("---")

# ══════════════════════════════
# 7-DAY FORECAST CARDS
# ══════════════════════════════
st.markdown("### 📅 7-Day Forecast (2024-06-29 → 2024-07-05)")

cols = st.columns(7)
for col, day in zip(cols, FORECAST_DATA):
    is_weekend = day['day'] in ['Saturday','Sunday']
    bg    = '#FEF3C7' if is_weekend else '#FFFFFF'
    bdr   = '#F59E0B' if is_weekend else '#E2E8F0'
    bdr_l = '#D97706' if is_weekend else '#0066CC'
    with col:
        st.markdown(f"""
        <div style='background:{bg};border:1px solid {bdr};border-top:4px solid {bdr_l};
                    border-radius:10px;padding:14px 10px;text-align:center;
                    box-shadow:0 2px 8px rgba(0,0,0,0.06);'>
            <div style='font-size:12px;font-weight:700;color:#1E293B;'>{day["day"][:3].upper()}</div>
            <div style='font-size:10px;color:#374151;margin-bottom:8px;font-weight:600;'>{day["date"][5:]}</div>
            <div style='font-size:24px;font-weight:900;color:#0066CC;'>{day["patients"]}</div>
            <div style='font-size:10px;color:#374151;margin-bottom:6px;font-weight:600;'>patients</div>
            <div style='font-size:18px;font-weight:800;color:#D97706;'>{day["wait"]:.1f}</div>
            <div style='font-size:10px;color:#374151;margin-bottom:6px;font-weight:600;'>avg wait (min)</div>
            <div style='font-size:11px;font-weight:700;color:#D97706;'>{day["load"]}</div>
            {"<div style='margin-top:6px;font-size:10px;color:#D97706;font-weight:600;'>📅 Weekend</div>" if is_weekend else ""}
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# ══════════════════════════════
# DAY DRILL-DOWN + HOURLY CHART
# ══════════════════════════════
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### 🔍 Day Drill-Down")

    day_names = [d['day'] for d in FORECAST_DATA]
    sel_day   = st.selectbox("Select forecast day:", day_names)
    sel_data  = next(d for d in FORECAST_DATA if d['day']==sel_day)
    hourly    = HOURLY_PROFILE.get(sel_day, HOURLY_PROFILE['Monday'])

    is_wknd = sel_day in ['Saturday','Sunday']
    bg_card = '#FFFBEB' if is_wknd else '#EFF6FF'
    bdr_clr = '#D97706' if is_wknd else '#0066CC'

    st.markdown(f"""
    <div style='background:{bg_card};border:2px solid {bdr_clr};border-radius:12px;
                padding:20px;margin-bottom:16px;'>
        <div style='font-size:20px;font-weight:800;color:#1E293B;margin-bottom:12px;'>
            {sel_day} — {sel_data["date"]}
            {"&nbsp; 📅 Weekend" if is_wknd else ""}
        </div>
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;'>
            <div style='background:white;border-radius:8px;padding:12px;text-align:center;'>
                <div style='font-size:11px;color:#374151;font-weight:600;'>Expected Patients</div>
                <div style='font-size:28px;font-weight:800;color:#0066CC;'>{sel_data["patients"]}</div>
            </div>
            <div style='background:white;border-radius:8px;padding:12px;text-align:center;'>
                <div style='font-size:11px;color:#374151;font-weight:600;'>Avg Wait Time</div>
                <div style='font-size:28px;font-weight:800;color:#D97706;'>{sel_data["wait"]:.1f}m</div>
            </div>
            <div style='background:white;border-radius:8px;padding:12px;text-align:center;'>
                <div style='font-size:11px;color:#374151;font-weight:600;'>Load Status</div>
                <div style='font-size:16px;font-weight:800;color:#D97706;'>{sel_data["load"]}</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    try:
        import plotly.graph_objects as go
        hours  = list(range(24))
        labels = [f"{h:02d}:00" for h in hours]
        colors = ['#DC2626' if hourly[h]>=5 else
                  '#D97706' if hourly[h]>=3 else
                  '#0066CC' if hourly[h]>=1 else '#E2E8F0'
                  for h in hours]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=labels, y=hourly, marker_color=colors,
            text=[str(v) if v>0 else '' for v in hourly],
            textposition='outside',
            hovertemplate='%{x}<br>Patients: %{y}<extra></extra>'
        ))
        fig.add_vrect(x0=9, x1=17, fillcolor='rgba(0,200,100,0.06)', layer='below', line_width=0)
        fig.add_vline(x=9,  line_dash='dash', line_color='#059669',
                      annotation_text='OPD Open',  annotation_position='top')
        fig.add_vline(x=17, line_dash='dash', line_color='#DC2626',
                      annotation_text='OPD Close', annotation_position='top')
        fig.update_layout(
            title=dict(text=f"Hourly Patient Arrivals — {sel_day}", font=dict(size=14, color='#1E293B')),
            xaxis=dict(tickangle=45, tickfont=dict(size=9, color='#1E293B')),
            yaxis=dict(title='Patients', gridcolor='#F1F5F9', tickfont=dict(color='#1E293B')),
            plot_bgcolor='white', paper_bgcolor='white',
            height=300, margin=dict(l=40,r=20,t=50,b=60),
            showlegend=False, font=dict(color='#1E293B')
        )
        st.plotly_chart(fig, width="stretch")
    except ImportError:
        st.markdown("**Hourly Patient Distribution**")
        peak_hours = [(h, hourly[h]) for h in range(24) if hourly[h] > 0]
        for h, v in peak_hours:
            bar = "█" * v
            opd = " ← OPD" if 9 <= h < 17 else ""
            st.markdown(f"`{h:02d}:00` {bar} {v}{opd}")

with right_col:
    st.markdown("### ⚡ FCFS vs Smart Queue")

    st.markdown(f"""
    <div style='background:#F0FDF4;border:2px solid #10B981;border-radius:12px;
                padding:20px;margin-bottom:16px;'>
        <div style='font-size:18px;font-weight:800;color:#065F46;margin-bottom:16px;'>
            ✅ Smart Queue saves <span style='color:#059669;font-size:24px;'>{SAVING_PCT}%</span>
            = {SAVING_MIN} min per patient
        </div>
        <div style='margin-bottom:16px;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                <span style='font-size:13px;font-weight:600;color:#1E293B;'>FCFS (First Come First Serve)</span>
                <span style='font-size:15px;font-weight:800;color:#DC2626;'>{FCFS_AVG} min</span>
            </div>
            <div style='background:#F1F5F9;border-radius:6px;height:20px;overflow:hidden;'>
                <div style='width:100%;background:#EF4444;height:100%;border-radius:6px;
                            display:flex;align-items:center;justify-content:flex-end;padding-right:8px;'>
                    <span style='font-size:11px;color:white;font-weight:600;'>{FCFS_AVG} min</span>
                </div>
            </div>
        </div>
        <div style='margin-bottom:12px;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                <span style='font-size:13px;font-weight:600;color:#1E293B;'>Smart Queue (4-Level Priority)</span>
                <span style='font-size:15px;font-weight:800;color:#059669;'>{SMART_AVG} min</span>
            </div>
            <div style='background:#F1F5F9;border-radius:6px;height:20px;overflow:hidden;'>
                <div style='width:{int(SMART_AVG/FCFS_AVG*100)}%;background:#10B981;height:100%;border-radius:6px;
                            display:flex;align-items:center;justify-content:flex-end;padding-right:8px;'>
                    <span style='font-size:11px;color:white;font-weight:600;'>{SMART_AVG} min</span>
                </div>
            </div>
        </div>
        <div style='background:white;border-radius:8px;padding:12px;text-align:center;'>
            <span style='font-size:13px;color:#374151;font-weight:600;'>Time saved: </span>
            <span style='font-size:22px;font-weight:900;color:#059669;'>{SAVING_MIN} min</span>
            <span style='font-size:13px;color:#059669;font-weight:600;'> per patient ({SAVING_PCT}% reduction)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size:14px;font-weight:700;color:#1E293B;'>Per-Day Comparison (Forecast Week)</p>", unsafe_allow_html=True)
    try:
        import plotly.graph_objects as go
        days_short = [d['day'][:3] for d in FORECAST_DATA]
        fcfs_vals  = [d['wait'] for d in FORECAST_DATA]
        smart_vals = [round(d['wait'] * (1 - SAVING_PCT/100), 1) for d in FORECAST_DATA]

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name='FCFS', x=days_short, y=fcfs_vals,
                               marker_color='#EF4444', text=fcfs_vals,
                               textposition='outside', textfont=dict(size=10, color='#1E293B')))
        fig2.add_trace(go.Bar(name='Smart Queue', x=days_short, y=smart_vals,
                               marker_color='#10B981', text=smart_vals,
                               textposition='outside', textfont=dict(size=10, color='#1E293B')))
        fig2.update_layout(
            barmode='group',
            yaxis=dict(title='Avg Wait (min)', gridcolor='#F1F5F9', tickfont=dict(color='#1E293B')),
            xaxis=dict(tickfont=dict(color='#1E293B')),
            plot_bgcolor='white', paper_bgcolor='white',
            height=260, margin=dict(l=40,r=20,t=20,b=40),
            legend=dict(orientation='h', yanchor='bottom', y=1.0, xanchor='right', x=1,
                        font=dict(color='#1E293B')),
            font=dict(color='#1E293B')
        )
        st.plotly_chart(fig2, use_container_width=True)
    except ImportError:
        for d in FORECAST_DATA:
            smart = round(d['wait'] * (1-SAVING_PCT/100), 1)
            st.markdown(f"**{d['day'][:3]}**: FCFS {d['wait']}m → Smart {smart}m (save {d['wait']-smart:.1f}m)")

    st.markdown("---")
    st.markdown("### 👥 Staff Planning Guide")
    for s in STAFF_ADVICE:
        st.markdown(f"""
        <div style='background:{s["bg"]};border-left:4px solid {s["color"]};
                    border-radius:8px;padding:10px 14px;margin-bottom:8px;'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span style='font-size:13px;font-weight:700;color:#1E293B;'>{s["segment"]}</span>
                <span style='background:{s["color"]};color:white;padding:2px 10px;
                             border-radius:10px;font-size:11px;font-weight:600;'>
                    {s["intensity"]}
                </span>
            </div>
            <div style='font-size:12px;color:#1E293B;margin-top:4px;font-weight:500;'>{s["advice"]}</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════
# LSTM ARCHITECTURE SUMMARY
# ══════════════════════════════
st.markdown("---")
st.markdown("### 🧠 LSTM Architecture Summary")

arch_col, train_col, feat_col = st.columns(3)

with arch_col:
    st.markdown("""
    <div style='background:white;border:1px solid #E2E8F0;border-top:4px solid #D97706;
                border-radius:10px;padding:16px;'>
        <div style='font-size:14px;font-weight:700;color:#1E293B;margin-bottom:12px;'>🏗️ Model Layers</div>
    """, unsafe_allow_html=True)
    layers = [
        ("Input",        "9 features × 7 days",    "#374151"),
        ("LSTM Layer 1", "128 units (return_seq)",  "#D97706"),
        ("Dropout",      "0.2",                     "#6B7280"),
        ("LSTM Layer 2", "64 units",                "#D97706"),
        ("Dropout",      "0.2",                     "#6B7280"),
        ("Dense",        "32 units (ReLU)",          "#0066CC"),
        ("Output",       "2: [PatientCount, Wait]",  "#059669"),
    ]
    for name, detail, color in layers:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:6px 0;
                    border-bottom:1px solid #F1F5F9;'>
            <span style='font-size:12px;font-weight:600;color:{color};'>{name}</span>
            <span style='font-size:12px;color:#374151;font-weight:500;'>{detail}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with train_col:
    st.markdown("""
    <div style='background:white;border:1px solid #E2E8F0;border-top:4px solid #0066CC;
                border-radius:10px;padding:16px;'>
        <div style='font-size:14px;font-weight:700;color:#1E293B;margin-bottom:12px;'>⚙️ Training Config</div>
    """, unsafe_allow_html=True)
    configs = [
        ("Optimizer",    "Adam"),
        ("Loss",         "MSE"),
        ("Epochs",       "18 (early stop at 3)"),
        ("Patience",     "15"),
        ("Batch Size",   "32"),
        ("Look-back",    "7 days"),
        ("Train/Test",   "80% / 20%"),
        ("Total Params", "122,210"),
    ]
    for k, v in configs:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:6px 0;
                    border-bottom:1px solid #F1F5F9;'>
            <span style='font-size:12px;color:#1E293B;font-weight:600;'>{k}</span>
            <span style='font-size:12px;font-weight:700;color:#0066CC;'>{v}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with feat_col:
    st.markdown("""
    <div style='background:white;border:1px solid #E2E8F0;border-top:4px solid #059669;
                border-radius:10px;padding:16px;'>
        <div style='font-size:14px;font-weight:700;color:#1E293B;margin-bottom:12px;'>📊 Input Features (9)</div>
    """, unsafe_allow_html=True)
    features = [
        ("PatientCount",   "Target variable"),
        ("AvgWaitTime",    "Target variable"),
        ("AvgConsultMin",  "Daily avg"),
        ("DoctorsOnShift", "Staff"),
        ("NursesOnShift",  "Staff"),
        ("StaffShortage",  "Binary flag"),
        ("OccupancyRate",  "0.0–1.0"),
        ("DayOfWeek",      "0–6"),
        ("IsWeekend",      "Binary flag"),
    ]
    for feat, note in features:
        st.markdown(f"""
        <div style='display:flex;justify-content:space-between;padding:6px 0;
                    border-bottom:1px solid #F1F5F9;'>
            <span style='font-size:12px;font-weight:600;color:#059669;'>{feat}</span>
            <span style='font-size:11px;color:#374151;font-weight:500;'>{note}</span>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)