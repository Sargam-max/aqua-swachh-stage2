# Aqua-Swachh: Machine Learning Filter Condition Prototype & Web Dashboard (Stage 2)

> **Predictive Filter Health Monitoring for Smart Water Purifiers**  
> *Developed for atomQUEST Track 2 - Team Aqua-Swachh (RGIPT)*

---

## 1. Overview & Stage 2 Objectives

This project contains the complete machine learning and interactive web dashboard system designed for **Stage 2** of the **Aqua-Swachh** smart water purifier. 

As recommended for Stage 2 development:
- **Distinguish a working ML prototype from a validated lifetime prediction**: The objective of Stage 2 is to demonstrate a robust, physics-informed data logging and classification pipeline, rather than claiming unvalidated days-of-life or drinking safety guarantees.
- **Model Task**: Classify current operational filter condition into three operational states:
  1. `Normal`: Filter operating within nominal parameters (high salt rejection $\eta > 88\%$, normal permeate flow rate).
  2. `Monitor`: Early signs of membrane scaling or moderate fouling (salt rejection $74\%-88\%$, slightly degraded flow). Cartridge functional, but preparing for replacement.
  3. `Service Required`: Membrane breakdown ($\eta < 70\%$) or severe clogging ($Q < 0.9$ L/min). Immediate maintenance/swap needed.

---

## 2. Complete Folder Structure

```
c:\Users\chris\Documents\antigravity\vibrant-bardeen\
├── data/
│   ├── generate_dataset.py          # Physics-based synthetic dataset generator
│   ├── data_logging_esp32.py        # USB serial logger from ESP32 to CSV
│   └── sensor_data.csv              # 1,500 labeled sample records
├── firmware/
│   └── esp32_sensor_node.ino        # Complete Arduino C++ sketch for ESP32-S3
├── model/
│   ├── features.py                  # Physics feature extraction (η, Temp Comp, Delta TDS)
│   ├── train.py                     # Training pipeline, baseline comparison, 5-fold CV
│   └── saved_models/
│       ├── random_forest_filter_model.joblib # Serialized model pipeline
│       └── metrics_summary.json     # Exported evaluation metrics & confusion matrix
├── inference/
│   └── predict.py                   # Real-time inference engine + safety guardrails
├── web/
│   ├── app.py                       # FastAPI application (REST API & static server)
│   └── static/
│       └── index.html               # Responsive interactive web dashboard (Chart.js + Tailwind)
├── demo.py                          # CLI interactive live demonstration
├── run_dashboard.bat                # 1-Click launcher for Windows
├── stop_dashboard.bat               # 1-Click server stop utility
├── tests/
│   └── test_backend.py              # Automated 16-test backend verification suite
├── Dockerfile                       # Container definition for cloud deployment
├── Procfile                         # Render / Railway process definition
├── requirements.txt                 # Frozen Python dependencies
└── README.md
```

---

## 3. How to Run the Code

### Step 1: Open Terminal in Project Directory
```powershell
cd c:\Users\chris\Documents\antigravity\vibrant-bardeen
```

### Step 2: Activate Virtual Environment
```powershell
# On Windows PowerShell:
.venv\Scripts\activate

# On Windows Command Prompt (cmd):
.venv\Scripts\activate.bat
```

### Step 3: Run the Web Frontend & Dashboard
You can double-click **`run_dashboard.bat`** or run:
```powershell
.venv\Scripts\python.exe -m uvicorn web.app:app --host 127.0.0.1 --port 8000 --reload
```
Then open your browser at:
👉 **`http://127.0.0.1:8000`**

### Step 4: Run the Command-Line Demo (CLI)
```powershell
.venv\Scripts\python.exe -u demo.py
```

### Step 5: Retrain or Benchmark the Models
```powershell
.venv\Scripts\python.exe -u -m model.train
```

### Step 6: Stream Live Data from ESP32 Hardware via USB
Flash `firmware/esp32_sensor_node.ino` to your ESP32-S3 and run:
```powershell
.venv\Scripts\python.exe -u data/data_logging_esp32.py --port COM3
```

---

## 4. How to Host It Online

You have 3 easy options to share your dashboard with teammates, mentors, and competition judges:

### Option A: Free Cloud Hosting on Render.com (Recommended - 5 Minutes)
1. Push this folder to a GitHub repository.
2. Sign in to [Render.com](https://render.com) (free).
3. Click **New +** → **Web Service** → Connect your GitHub repository.
4. Set the following settings:
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt && python data/generate_dataset.py && python -m model.train`
   - **Start Command**: `uvicorn web.app:app --host 0.0.0.0 --port $PORT`
5. Click **Deploy Web Service**. Render gives you a public URL like `https://aqua-swachh.onrender.com`!

### Option B: Free Cloud Hosting on Railway.app
1. Push to GitHub.
2. Go to [Railway.app](https://railway.app), click **New Project** → **Deploy from GitHub repo**.
3. Railway automatically detects the included `Dockerfile` and `Procfile` and deploys it with a live HTTPS URL.

### Option C: Instant Public Tunnel via Cloudflare (Zero Deployment, 30 Seconds)
If you want to demo live from your laptop to anyone worldwide without pushing code:
1. In one terminal, run the local dashboard:
   ```powershell
   .venv\Scripts\python.exe -m uvicorn web.app:app --port 8000
   ```
2. In a second terminal, run Cloudflare Tunnel:
   ```powershell
   # If cloudflared or ngrok is installed:
   cloudflared tunnel --url http://localhost:8000
   # Or using ngrok:
   ngrok http 8000
   ```
   This gives you an instant public HTTPS link that forwards live to your local machine!

### Option D: Host on Your Local Network (For Phone / Tablet Demo)
To view the dashboard on your phone or team laptops connected to the same Wi-Fi router:
```powershell
.venv\Scripts\python.exe -m uvicorn web.app:app --host 0.0.0.0 --port 8000
```
Then open `http://<YOUR_LAPTOP_IP>:8000` on your mobile phone browser.

---

## 5. Web Dashboard Features

- **Live Sensor Telemetry Gauges**: Real-time readings for Feed TDS, Permeate TDS, Flow Rate, and Water Temperature.
- **Physics Indicators**: Computed Salt Rejection ($\eta\%$), Temperature-Compensated Rejection (25°C base), Delta TDS removed, and Flow-Rejection Index.
- **Live Classification Badge**: Prominent dynamic badges (`NORMAL`, `MONITOR`, `SERVICE REQUIRED`) with confidence score.
- **Probability Distribution Gauges**: Real-time progress bars showing class probabilities from the Random Forest ensemble.
- **Preset Test Bench**: Instant 1-click test scenarios for Fresh Filter, Hard Water, Scaling, Clogged Filter, and Membrane Tear.
- **Live Sensor Sliders**: Interactive sliders allowing custom manual tuning of TDS In, TDS Out, Temp, and Flow Rate with immediate live ML inference.
- **Real-Time Telemetry Graphs**: Rolling dual-axis Chart.js graphs displaying TDS trends and Salt Rejection / Flow trends.
- **Safety Guardrail Indicator**: Visual indicator when deterministic fail-safe checks override the ML model.
