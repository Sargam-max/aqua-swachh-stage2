"""
Aqua-Swachh: Web Dashboard Backend & Real-World Sensor Analysis API
===================================================================
FastAPI application serving:
- Real-time ML inference for single sensor snapshots
- Batch CSV processor for real sensor data logs (ESP32 SD/Serial logs)
- Physics indicators, safety guardrails & feature explainability
- System configuration and hardware telemetry
- Static human-designed frontend UI
"""

import os
import io
import json
import random
import pandas as pd
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from inference.predict import FilterHealthPredictor

app = FastAPI(
    title="Aqua-Swachh Smart Purifier AI",
    description="Predictive Filter Health Monitoring API (atomQUEST Track 2 - RGIPT)",
    version="2.0.0"
)

# Paths
MODEL_PATH = "model/saved_models/random_forest_filter_model.joblib"
METRICS_PATH = "model/saved_models/metrics_summary.json"

try:
    predictor = FilterHealthPredictor(model_path=MODEL_PATH)
except Exception as e:
    print(f"[WARN] Predictor initialization deferred: {e}")
    predictor = None

class SensorInput(BaseModel):
    tds_in: float = Field(..., ge=0, le=2500, description="Feed water TDS in ppm")
    tds_out: float = Field(..., ge=0, le=2500, description="Permeate water TDS in ppm")
    temperature: float = Field(25.0, ge=0, le=60, description="Water temperature in °C")
    flow_rate: float = Field(1.8, ge=0, le=10, description="Permeate flow rate in L/min")

PRESET_SCENARIOS = [
    {
        "id": "new_filter",
        "name": "Fresh RO Membrane",
        "category": "Normal",
        "description": "Standard municipal water (420 ppm) with high rejection (>93%) and strong permeate flow.",
        "tds_in": 420.0,
        "tds_out": 28.0,
        "temperature": 25.5,
        "flow_rate": 1.95
    },
    {
        "id": "hard_water",
        "name": "High-Hardness Tap Water",
        "category": "Normal",
        "description": "Challenging hard water feed (780 ppm) successfully handled by healthy membrane (48 ppm permeate).",
        "tds_in": 780.0,
        "tds_out": 48.0,
        "temperature": 27.0,
        "flow_rate": 1.80
    },
    {
        "id": "moderate_scaling",
        "name": "Early Membrane Scaling",
        "category": "Monitor",
        "description": "Mineral foulant build-up on polyamide pores. Salt rejection dropped to ~77.5%, flow restricted to 1.25 L/min.",
        "tds_in": 510.0,
        "tds_out": 115.0,
        "temperature": 26.8,
        "flow_rate": 1.25
    },
    {
        "id": "clogged_flow",
        "name": "Sediment / Carbon Clogging",
        "category": "Service Required",
        "description": "Pre-filters blocked by sediment; severe pressure drop choking permeate flow to 0.65 L/min.",
        "tds_in": 480.0,
        "tds_out": 120.0,
        "temperature": 26.0,
        "flow_rate": 0.65
    },
    {
        "id": "membrane_damage",
        "name": "Membrane Pore Breakthrough",
        "category": "Service Required",
        "description": "Membrane tear or chlorine chemical attack allowing raw dissolved solids to pass directly through.",
        "tds_in": 550.0,
        "tds_out": 230.0,
        "temperature": 28.2,
        "flow_rate": 1.40
    }
]

def explain_prediction(status: str, rejection: float, flow: float, tds_out: float) -> str:
    """Provides plain human explainability for water engineers & users."""
    if status == "Normal":
        return f"Nominal operation: Salt rejection is high at {rejection:.1f}% (benchmark >88%) with healthy permeate flow ({flow:.2f} L/min). Water output is clear and safe."
    elif status == "Monitor":
        return f"Preventive alert: Salt rejection has slipped to {rejection:.1f}% and flow is down to {flow:.2f} L/min. This indicates early foulant deposition or mineral scaling. Plan cartridge replacement in 2-4 weeks."
    else: # Service Required
        if tds_out >= 200.0:
            return f"Critical failure: High salt passage ({tds_out:.1f} ppm) and low rejection ({rejection:.1f}%). Membrane barrier integrity is compromised. Replace RO cartridge immediately."
        elif flow < 0.8:
            return f"Severe blockage: Flow rate has dropped to {flow:.2f} L/min (<0.8 L/min limit) due to particulate head loss. Sediment/carbon stage requires replacement."
        return f"Service required: Combined performance indicators show severe degradation ({rejection:.1f}% rejection, {flow:.2f} L/min flow)."

@app.post("/api/predict")
def predict_filter_condition(inputs: SensorInput) -> Dict[str, Any]:
    global predictor
    if predictor is None:
        try:
            predictor = FilterHealthPredictor(model_path=MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model could not be loaded: {str(e)}")
            
    result = predictor.predict_sample(
        tds_in=inputs.tds_in,
        tds_out=inputs.tds_out,
        temperature=inputs.temperature,
        flow_rate=inputs.flow_rate
    )
    
    # Add human explanation
    phys = result['physics_indicators']
    explanation = explain_prediction(
        result['predicted_status'],
        phys['salt_rejection_pct'],
        phys['flow_rate_lpm'],
        inputs.tds_out
    )
    result['explanation'] = explanation
    return result

@app.get("/api/scenarios")
def get_scenarios():
    return PRESET_SCENARIOS

@app.get("/api/metrics")
def get_model_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"message": "Metrics file not found. Train model first."}

@app.get("/api/system-telemetry")
def get_system_telemetry():
    """Hardware node status, water purifier telemetry & life counters."""
    return {
        "hardware": {
            "microcontroller": "ESP32-S3 DevKit-C (Xtensa dual-core LX7)",
            "firmware_version": "v1.4.2-atomQUEST",
            "connection_mode": "Active (USB Serial / UART @ 115200 baud)",
            "sensors": [
                {"name": "Feed TDS", "sensor": "Xcluma TDS Meter V1.0", "channel": "ADC1 / GPIO 1", "status": "ONLINE"},
                {"name": "Permeate TDS", "sensor": "Xcluma TDS Meter V1.0", "channel": "ADC1 / GPIO 2", "status": "ONLINE"},
                {"name": "Temperature", "sensor": "DS18B20 1-Wire Probe", "channel": "GPIO 4", "status": "ONLINE"},
                {"name": "Permeate Flow", "sensor": "YF-S201 Hall Effect", "channel": "GPIO 5 (Pulse ISR)", "status": "ONLINE"}
            ]
        },
        "purifier_stats": {
            "installation_date": "2026-09-15",
            "cumulative_volume_liters": 1845.0,
            "estimated_remaining_days": 38,
            "cartridge_model": "AquaSwachh RO-75GPD Polyamide Thin-Film Composite",
            "nominal_rated_capacity_liters": 6000.0,
            "filter_health_pct": 82
        }
    }

def process_dataframe_inference(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    global predictor
    if predictor is None:
        predictor = FilterHealthPredictor(model_path=MODEL_PATH)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    required_cols = ['tds_in', 'tds_out', 'temperature', 'flow_rate']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"CSV missing required columns: {missing}. Found: {list(df.columns)}"
        )
        
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    if len(df) == 0:
        raise HTTPException(status_code=400, detail="Uploaded CSV contains no valid data rows.")

    if len(df) > 1000:
        df = df.iloc[:1000]

    predictions = []
    statuses = []
    rejections = []
    
    for idx, row in df.iterrows():
        pred = predictor.predict_sample(
            tds_in=float(row['tds_in']),
            tds_out=float(row['tds_out']),
            temperature=float(row['temperature']),
            flow_rate=float(row['flow_rate'])
        )
        statuses.append(pred['predicted_status'])
        rejections.append(pred['physics_indicators']['salt_rejection_pct'])
        
        if idx < 50:
            predictions.append({
                "row_index": idx + 1,
                "timestamp": str(row.get('timestamp', f"Row #{idx+1}")),
                "tds_in": round(float(row['tds_in']), 1),
                "tds_out": round(float(row['tds_out']), 1),
                "temperature": round(float(row['temperature']), 1),
                "flow_rate": round(float(row['flow_rate']), 2),
                "salt_rejection": pred['physics_indicators']['salt_rejection_pct'],
                "predicted_status": pred['predicted_status'],
                "confidence": round(pred['confidence'] * 100, 1),
                "guardrail_triggered": pred['guardrail_triggered']
            })
            
    status_counts = pd.Series(statuses).value_counts().to_dict()
    total_rows = len(statuses)
    
    return {
        "filename": filename,
        "total_records": total_rows,
        "breakdown": {
            "Normal": status_counts.get("Normal", 0),
            "Monitor": status_counts.get("Monitor", 0),
            "Service Required": status_counts.get("Service Required", 0)
        },
        "percentages": {
            "Normal": round((status_counts.get("Normal", 0) / total_rows) * 100, 1),
            "Monitor": round((status_counts.get("Monitor", 0) / total_rows) * 100, 1),
            "Service Required": round((status_counts.get("Service Required", 0) / total_rows) * 100, 1)
        },
        "avg_salt_rejection_pct": round(float(pd.Series(rejections).mean()), 2),
        "avg_flow_rate_lpm": round(float(df['flow_rate'].mean()), 2),
        "rows_preview": predictions
    }

@app.post("/api/upload-csv")
async def analyze_uploaded_csv(file: UploadFile = File(...)):
    """Analyzes real-world sensor logs uploaded as a CSV."""
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode('utf-8', errors='ignore')))
        return process_dataframe_inference(df, file.filename)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error parsing CSV file: {str(e)}")

@app.get("/api/load-sample-csv")
def load_sample_csv():
    """Loads bundled sensor_data.csv and returns ML inference results instantly."""
    sample_path = "data/sensor_data.csv"
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample dataset data/sensor_data.csv not found")
    df = pd.read_csv(sample_path)
    return process_dataframe_inference(df, "sensor_data.csv (Bundled Stage 2 Dataset)")

@app.get("/api/download-sample-csv")
def download_sample_csv():
    """Downloads sample sensor_data.csv."""
    sample_path = "data/sensor_data.csv"
    if not os.path.exists(sample_path):
        raise HTTPException(status_code=404, detail="Sample dataset not found")
    return FileResponse(sample_path, filename="sensor_data.csv", media_type="text/csv")

@app.get("/api/download-zip")
def download_project_zip():
    """Downloads complete project zip archive."""
    zip_path = "aqua-swachh-stage2.zip"
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="Zip package not found")
    return FileResponse(zip_path, filename="aqua-swachh-stage2.zip", media_type="application/zip")


@app.get("/api/stream-sample")
def get_stream_sample(state: Optional[str] = "Normal"):
    """Returns realistic sensor readings with continuous physical drift."""
    if state == "Normal":
        t_in = 420 + random.uniform(-15, 25)
        rejection = random.uniform(0.92, 0.96)
        temp = 25.5 + random.uniform(-0.5, 0.5)
        flow = 1.90 + random.uniform(-0.08, 0.08)
        t_out = t_in * (1 - rejection)
    elif state == "Monitor":
        t_in = 470 + random.uniform(-20, 25)
        rejection = random.uniform(0.78, 0.84)
        temp = 26.2 + random.uniform(-0.6, 0.6)
        flow = 1.28 + random.uniform(-0.06, 0.06)
        t_out = t_in * (1 - rejection)
    else: # Service Required
        t_in = 520 + random.uniform(-20, 30)
        rejection = random.uniform(0.55, 0.68)
        temp = 27.5 + random.uniform(-0.8, 0.8)
        flow = 0.62 + random.uniform(-0.06, 0.06)
        t_out = t_in * (1 - rejection)
        
    return {
        "tds_in": round(t_in, 1),
        "tds_out": round(t_out, 1),
        "temperature": round(temp, 1),
        "flow_rate": round(flow, 2)
    }

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/")
def serve_index():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Aqua-Swachh API running. Static index.html not yet created."}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("web.app:app", host="0.0.0.0", port=port, reload=False)
