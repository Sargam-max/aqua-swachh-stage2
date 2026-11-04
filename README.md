# 💧 Aqua-Swachh: Smart Water Purifier Predictive Health Intelligence

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32--S3-E7352C?logo=espressif&logoColor=white)](https://www.espressif.com/)
[![Track](https://img.shields.io/badge/atomQUEST-Track%202%20%28RGIPT%29-0284c7)](https://rgipt.ac.in)
[![Tests](https://img.shields.io/badge/Tests-16%20Passing-10b981?logo=checkmarx&logoColor=white)](tests/test_backend.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Predictive Filter Health Monitoring for Smart Water Purifiers**  
> An edge-to-cloud machine learning system that diagnoses RO membrane health, detects early scaling, and enforces deterministic safety guardrails on drinking water quality.
>
> 🌐 **Live Public Demo:** [https://aqua-swachh.onrender.com/](https://aqua-swachh.onrender.com/)  
> 📦 **GitHub Repository:** [https://github.com/Sargam-max/aqua-swachh-stage2](https://github.com/Sargam-max/aqua-swachh-stage2)

---

## 📌 The Problem We Are Solving

Millions of households and commercial facilities rely on Reverse Osmosis (RO) drinking water purifiers. Yet virtually all consumer purifiers operate on **blind, timer-based maintenance**:

1. **Premature Filter Swaps:** Technicians routinely replace functional cartridges every 6 to 12 months based purely on the calendar, wasting thousands of rupees and generating unnecessary plastic and activated carbon waste.
2. **Undetected Membrane Breakthrough:** If an RO membrane develops microscopic tears or scales up prematurely due to aggressive local borewell water, consumers continue drinking contaminated water without knowing it.
3. **Misleading Single-Sensor Alarms:** A simple TDS spike doesn't necessarily mean a filter is broken—if the city supply or tanker water becomes harder, a healthy membrane will naturally have a higher absolute permeate TDS while still rejecting 93%+ of impurities.

**Aqua-Swachh** solves this by pairing an **ESP32-S3 multi-sensor IoT edge array** with **physics-informed machine learning** and **deterministic safety guardrails** to continuously diagnose the true physical condition of the filtration membrane.

---

## 🔬 How It Works: The 5-Stage Purifier Architecture

Aqua-Swachh monitors the water journey across critical checkpoints in the filtration process:

```
[ 1. Raw Tap Inlet ] ────► [ 2. Sediment & Carbon ] ────► [ 3. RO Membrane Core ] ────► [ 4. Permeate Output ] ────► [ 5. Flow Meter ] ────► Pure Tap
   Xcluma TDS-1                5-Micron Spun Yarn              75 GPD Thin-Film             Xcluma TDS-2               YF-S201 Turbine
   GPIO 1 (ADC1)               Particulate Trap                Salt Separation η            GPIO 2 (ADC1)              GPIO 5 (Pulse ISR)
   Measures Feed TDS           Turbidity < 1 NTU               Fouling & Scaling            Measures Pure TDS          Measures Permeate Q
```

### Sensor Array Specification

| Sensor Channel | Hardware Component | Interface | Measurement Range | Role in Diagnostics |
| :--- | :--- | :--- | :--- | :--- |
| **Feed TDS (TDS-1)** | Xcluma Analog TDS V1.0 | GPIO 1 (ADC1) | $0 - 1500\text{ ppm}$ | Baseline feed mineral concentration |
| **Permeate TDS (TDS-2)** | Xcluma Analog TDS V1.0 | GPIO 2 (ADC1) | $0 - 500\text{ ppm}$ | Purified permeate solute concentration |
| **Water Temperature** | Dallas DS18B20 1-Wire | GPIO 4 | $-10 - 85^\circ\text{C}$ | Conductivity normalization to 25°C baseline |
| **Permeate Flow Rate** | YF-S201 1/2" Hall Turbine | GPIO 5 (ISR) | $0.2 - 10\text{ L/min}$ | Detects particulate head-loss & flow choking |

---

## 🧠 Physics-Informed Machine Learning (The Science)

Instead of treating machine learning like a black box, Aqua-Swachh embeds **electrochemical water treatment equations** into the feature pipeline (`model/features.py`):

### 1. Salt Rejection Efficiency ($\eta$)
$$\eta = 1 - \left(\frac{\text{TDS}_{out}}{\text{TDS}_{in}}\right)$$
*Why it matters:* Rejection isolates membrane health from input water swings. A feed TDS jump from 400 to 800 ppm filtered to 50 ppm represents a **healthy 93.8% rejection**, proving the membrane is working nominally.

### 2. 25°C Temperature Compensation
$$\text{TDS}_{\text{norm}} = \frac{\text{TDS}}{1.0 + 0.02 \times (T - 25.0)}$$
*Why it matters:* Water electrical conductivity drifts by approximately $2\%$ per $^\circ\text{C}$. This formula normalizes winter ($15^\circ\text{C}$) and summer ($35^\circ\text{C}$) readings to a shared 25°C calibration baseline.

### 3. Flow-Rejection Composite Index
$$\text{Index} = Q \times \max(\eta, 0)$$
*Why it matters:* Distinguishes **pre-filter clogging** (water remains clean, but flow plummets to a trickle) from **membrane rupture** (flow stays fast, but salt rejection crashes).

---

## 🎯 Model Benchmarks & Evaluation

We benchmarked three model tiers using stratified 5-fold cross-validation on 1,500 physics-grounded operational samples (`model/train.py`):

| Model Architecture | Cross-Val F1 (Macro) | Test Accuracy | Role in Pipeline |
| :--- | :---: | :---: | :--- |
| **Dummy Classifier** (Majority Class) | $0.2366$ | $55.0\%$ | Baseline lower bound |
| **Logistic Regression** (Linear Baseline) | $0.9962 \pm 0.007$ | $99.7\%$ | Linear benchmark comparison |
| **Random Forest Classifier** (Primary) | **$1.0000 \pm 0.000$** | **$100.0\%$** | **Primary production model** (100 estimators) |

### Three Diagnostic States:
* 🟢 **NORMAL:** Salt rejection $\eta > 88\%$, nominal permeate flow ($1.5 - 2.2\text{ L/min}$). Water quality is optimal.
* 🟡 **MONITOR:** Early mineral scaling or foulant deposition ($\eta = 74\% - 88\%$, flow $1.0 - 1.4\text{ L/min}$). Plan cartridge swap in upcoming cycles.
* 🔴 **SERVICE REQUIRED:** Membrane barrier compromised or particulate blockages ($Q < 0.7\text{ L/min}$ or $\text{TDS}_{out} > 200\text{ ppm}$).

---

## 🛡️ Deterministic Safety Guardrails

In consumer drinking water safety, statistical AI should never have unchecked authority. Aqua-Swachh implements **deterministic safety guardrails** (`inference/predict.py`) that run concurrently with the ML model:

$$\text{Trigger Condition} = (\text{TDS}_{out} \ge 250\text{ ppm}) \lor (\eta < 55\%) \lor (Q < 0.40\text{ L/min})$$

*If any trigger condition is met, the system **automatically overrides** the ML prediction to `Service Required`, alerts the user, and triggers an emergency indicator.*

---

## 📁 Repository Structure

```
├── web/
│   ├── app.py                      # FastAPI REST API & static server
│   └── static/
│       └── index.html              # Clean modern web dashboard (Tailwind + Chart.js)
├── model/
│   ├── features.py                 # Physics feature extraction (η, Temp Comp, Delta TDS)
│   ├── train.py                    # Model training, 5-fold cross validation, benchmark
│   └── saved_models/
│       ├── random_forest_filter_model.joblib  # Serialized production model pipeline
│       └── metrics_summary.json    # Exported evaluation metrics & confusion matrix
├── data/
│   ├── generate_dataset.py         # Physics-based synthetic dataset generator
│   ├── data_logging_esp32.py       # USB serial bridge logger for physical ESP32
│   └── sensor_data.csv             # 1,500 labeled benchmark records
├── firmware/
│   └── esp32_sensor_node.ino       # C++ Arduino firmware for ESP32-S3 (ADC filtering + ISR)
├── inference/
│   └── predict.py                  # High-level inference engine + safety guardrails
├── tests/
│   └── test_backend.py             # Automated 16-test backend verification suite
├── demo.py                         # Interactive terminal test bench (5 scenarios)
├── run_dashboard.bat               # 1-Click Windows launcher (auto-recovers ports & venv)
├── stop_dashboard.bat              # 1-Click clean shutdown utility for Windows
├── Dockerfile                      # Production container image
├── Procfile                        # Cloud process definition (Render / Railway)
├── requirements.txt                # Cross-platform dependencies
└── README.md
```

---

## 🚀 Quick Start Guide

### Option 1: 1-Click Launch on Windows (Easiest)
Simply double-click:
👉 **`run_dashboard.bat`**

*The launcher automatically detects Python, verifies model weights, cleans up any occupied ports, and opens your default browser at `http://127.0.0.1:8000`.*

---

### Option 2: Terminal Setup (Windows / macOS / Linux)

```bash
# 1. Clone the repository
git clone https://github.com/Sargam-max/aqua-swachh-stage2.git
cd aqua-swachh-stage2

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the web dashboard
uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at **`http://localhost:8000`**.

---

### Option 3: Run the CLI Test Bench
```bash
python demo.py
```
*Evaluates the 5 real-world filter scenarios in your terminal with physics indicators, probabilities, and recommendations.*

---

### Option 4: Run the Automated Test Suite
```bash
python -m unittest tests/test_backend.py -v
```
*Runs all 16 verification tests across physics calculations, ML inference, safety guardrails, and live REST API endpoints.*

---

## 🧪 Testing with Real Sensor Data

1. Open the **Batch Sensor Lab** tab in the web dashboard.
2. Click **"Analyze Bundled Stage 2 Dataset (1,000 Rows)"** to instantly run the Random Forest model on 1,000 real-world sensor records without needing to hunt for files.
3. Or drop any `.csv` file recorded from your purifier containing the columns:  
   `tds_in, tds_out, temperature, flow_rate`
4. Use the interactive table filters (`All`, `Normal`, `Monitor`, `Service`) to inspect individual predictions, confidence scores, and safety flags.

---

## 🌐 Free Cloud Deployment (Render.com)

1. Fork or push this repository to GitHub.
2. Sign in to [Render.com](https://render.com) and click **New +** → **Web Service**.
3. Select your repository (`Sargam-max/aqua-swachh-stage2`).
4. Configure:
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt && python data/generate_dataset.py && python -m model.train`
   * **Start Command:** `uvicorn web.app:app --host 0.0.0.0 --port $PORT`
5. Click **Deploy**. Render provides a free, public 24/7 HTTPS URL.

---

## 👥 Team & Institutional Acknowledgments

* **Project:** Aqua-Swachh (Predictive Filter Health Monitoring for Smart Water Purifiers)
* **Team:** Team Aqua-Swachh
* **Competition:** **atomQUEST Track 2**
* **Institution:** **Rajiv Gandhi Institute of Petroleum Technology (RGIPT)**

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and build upon this work for academic, commercial, and research purposes.
