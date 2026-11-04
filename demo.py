"""
Aqua-Swachh: Stage 2 ML Pipeline Live Demonstration
===================================================
Demonstrates the full end-to-end pipeline:
1. Loading or generating sensor dataset
2. Physics feature processing
3. Machine Learning model inference (Random Forest Classifier)
4. Visual test console for simulated scenarios
"""

import os
import sys
import time
import pandas as pd
import numpy as np

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from data.generate_dataset import generate_water_sensor_dataset
from model.train import train_and_evaluate
from inference.predict import FilterHealthPredictor


def run_demo():
    print("\n" + "=" * 70)
    print(" AQUA-SWACHH: SMART WATER PURIFIER PREDICTIVE MONITORING ")
    print(" Stage 2 ML Prototype: Filter Condition Classification")
    print("=" * 70)
    
    csv_path = "data/sensor_data.csv"
    model_path = "model/saved_models/random_forest_filter_model.joblib"
    
    # 1. Dataset check
    if not os.path.exists(csv_path):
        print("\n[Step 1] Dataset not found. Generating synthetic sensor dataset based on RO physics...")
        generate_water_sensor_dataset(n_samples=1200, output_path=csv_path)
    else:
        print(f"\n[Step 1] Sensor dataset located at: {csv_path}")

    # 2. Model training check
    if not os.path.exists(model_path):
        print("\n[Step 2] Training ML model (Random Forest vs Linear Baseline)...")
        train_and_evaluate(data_path=csv_path, model_output_dir="model/saved_models")
    else:
        print(f"\n[Step 2] Trained Random Forest model loaded from: {model_path}")

    predictor = FilterHealthPredictor(model_path=model_path)
    
    # 3. Demonstration Scenarios
    print("\n[Step 3] Running Live Demonstration on Different Filter Conditions:")
    print("-" * 70)
    
    scenarios = [
        {
            "scenario": "Scenario 1: Brand New RO Membrane (Clean Tap Water)",
            "tds_in": 420.0,
            "tds_out": 28.0,
            "temperature": 25.5,
            "flow_rate": 1.95,
            "expected": "Normal"
        },
        {
            "scenario": "Scenario 2: Normal Operation (High TDS Hard Water Input)",
            "tds_in": 780.0,
            "tds_out": 48.0,
            "temperature": 27.0,
            "flow_rate": 1.80,
            "expected": "Normal"
        },
        {
            "scenario": "Scenario 3: Moderate Membrane Scaling & Early Bio-fouling",
            "tds_in": 510.0,
            "tds_out": 115.0,
            "temperature": 26.8,
            "flow_rate": 1.25,
            "expected": "Monitor"
        },
        {
            "scenario": "Scenario 4: Severe Particulate Clogging (Flow Choked)",
            "tds_in": 480.0,
            "tds_out": 120.0,
            "temperature": 26.0,
            "flow_rate": 0.65,
            "expected": "Service Required"
        },
        {
            "scenario": "Scenario 5: Membrane Pore Damage / Salt Breakthrough",
            "tds_in": 550.0,
            "tds_out": 230.0,
            "temperature": 28.2,
            "flow_rate": 1.40,
            "expected": "Service Required"
        }
    ]
    
    for sc in scenarios:
        print(f"\n>>> {sc['scenario']}")
        print(f"    Raw Sensor Readings: Feed TDS={sc['tds_in']:.1f} ppm | Permeate TDS={sc['tds_out']:.1f} ppm | Temp={sc['temperature']:.1f}°C | Flow={sc['flow_rate']:.2f} L/min")
        
        result = predictor.predict_sample(
            tds_in=sc['tds_in'],
            tds_out=sc['tds_out'],
            temperature=sc['temperature'],
            flow_rate=sc['flow_rate']
        )
        
        phys = result['physics_indicators']
        print(f"    Physics Indicators  : Salt Rejection={phys['salt_rejection_pct']:.1f}% | Temp-Comp Rejection={phys['temp_compensated_rejection_pct']:.1f}% | Delta TDS={phys['delta_tds_ppm']:.1f} ppm")
        
        # Format status badge
        status = result['predicted_status']
        badge = f"[{status.upper()}]"
        
        probs = result['probabilities']
        prob_str = f"Normal: {probs.get('Normal', 0)*100:4.1f}% | Monitor: {probs.get('Monitor', 0)*100:4.1f}% | Service: {probs.get('Service Required', 0)*100:4.1f}%"
        
        print(f"    Model Prediction    : {badge} (Confidence: {result['confidence']*100:.1f}%)")
        print(f"    Class Probabilities : {prob_str}")
        if result['guardrail_triggered']:
            print(f"    Safety Guardrail    : [ACTIVATED] {result['guardrail_reason']}")
        print(f"    System Action       : {result['action_recommendation']}")
        print("-" * 70)

if __name__ == "__main__":
    run_demo()
