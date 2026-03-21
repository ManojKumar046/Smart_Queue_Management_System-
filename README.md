# 🏥 Smart Queue Management System — Streamlit App

## Folder Structure
```
SmartQueue_App/
│
├── app.py                        ← Home page (run this)
├── requirements.txt
├── pages/
│   ├── 1_Patient_Intake.py       ← Token generation + OPD check
│   ├── 2_Live_Queue.py           ← Real-time priority queue
│   ├── 3_ML_Prediction.py        ← Gradient Boosting wait prediction
│   └── 4_LSTM_Forecast.py        ← 7-day LSTM forecast
├── model/
│   ├── best_pipeline.pkl         ← (you provide — from your notebook)
│   └── oe_triage.pkl             ← (you provide — from your notebook)
└── data/
    └── Hospital_SmartQueue_Dataset.csv  ← (optional — your dataset)
```

---

## Step 1 — Save your model files (run in Jupyter)

```python
import joblib
joblib.dump(best_pipeline, 'SmartQueue_App/model/best_pipeline.pkl')
joblib.dump(oe_triage,     'SmartQueue_App/model/oe_triage.pkl')
```

---

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 3 — Run the app

```bash
cd SmartQueue_App
streamlit run app.py
```

Then open: **http://localhost:8501**

---

## Pages

| Page | What it does |
|------|-------------|
| 🏠 Home | Overview and navigation |
| 🎫 Patient Intake | Fill patient form → OPD check → token issued → ML wait prediction |
| 📋 Live Queue | Add patients → priority queue table → serve next → dynamic wait updates |
| 🤖 ML Prediction | Adjust any scenario → instant GB prediction → what-if analysis |
| 🔮 LSTM Forecast | 7-day forecast cards → hourly drill-down → FCFS vs Smart comparison → staff planning |

---

## Notes

- Without `best_pipeline.pkl` the app uses rule-based estimates (still works, just less accurate)
- The Live Queue page keeps state during the session (resets on browser refresh)
- LSTM forecast page uses the verified output from your trained model (hardcoded from notebook run)
