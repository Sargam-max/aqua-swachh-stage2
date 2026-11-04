"""
Aqua-Swachh: Synthetic Sensor Data Generator
============================================
Generates realistic water purification sensor logs based on filtration physics.
Simulates TDS before/after filtration, water temperature, flow rate, and degradation states.

Labels:
- Normal: High rejection (>88%), healthy flow rate (>1.4 L/min)
- Monitor: Early membrane fouling/scaling, moderate rejection (72%-88%), slightly reduced flow (1.0-1.4 L/min)
- Service Required: Severe fouling or membrane breakthrough, rejection (<72%) or severely restricted flow (<1.0 L/min)
"""

import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_water_sensor_dataset(
    n_samples: int = 1500,
    random_state: int = 42,
    output_path: str = "data/sensor_data.csv"
) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # Class distribution: 55% Normal, 25% Monitor, 20% Service Required
    n_normal = int(n_samples * 0.55)
    n_monitor = int(n_samples * 0.25)
    n_service = n_samples - n_normal - n_monitor
    
    start_time = datetime(2026, 10, 1, 8, 0, 0)
    timestamps = [start_time + timedelta(minutes=15 * i) for i in range(n_samples)]
    
    records = []
    
    # 1. Normal (Healthy Filter)
    for _ in range(n_normal):
        t_in = np.random.uniform(300, 850) # Feed water TDS (ppm)
        temp = np.random.normal(26.0, 3.5) # Temperature (°C)
        temp = np.clip(temp, 15.0, 40.0)
        
        # High rejection rate: 90% - 97%
        rejection = np.random.uniform(0.90, 0.97)
        # Temperature effect on permeability (higher temp slightly increases passage)
        temp_effect = 1.0 + 0.015 * (temp - 25.0)
        t_out = t_in * (1.0 - rejection) * temp_effect
        t_out += np.random.normal(0, 2.5) # Sensor noise
        t_out = max(10.0, t_out)
        
        # Healthy flow rate: 1.5 - 2.2 L/min
        flow = np.random.normal(1.85, 0.15)
        flow = np.clip(flow, 1.45, 2.40)
        
        records.append({
            'tds_in': round(float(t_in), 1),
            'tds_out': round(float(t_out), 1),
            'temperature': round(float(temp), 1),
            'flow_rate': round(float(flow), 2),
            'filter_status': 'Normal'
        })
        
    # 2. Monitor (Aging / Moderate Fouling)
    for _ in range(n_monitor):
        t_in = np.random.uniform(320, 900)
        temp = np.random.normal(26.5, 3.5)
        temp = np.clip(temp, 15.0, 40.0)
        
        # Degraded rejection rate: 74% - 87%
        rejection = np.random.uniform(0.74, 0.87)
        temp_effect = 1.0 + 0.015 * (temp - 25.0)
        t_out = t_in * (1.0 - rejection) * temp_effect
        t_out += np.random.normal(0, 3.5)
        t_out = max(40.0, t_out)
        
        # Slightly reduced flow: 1.05 - 1.45 L/min
        flow = np.random.normal(1.25, 0.12)
        flow = np.clip(flow, 1.00, 1.45)
        
        records.append({
            'tds_in': round(float(t_in), 1),
            'tds_out': round(float(t_out), 1),
            'temperature': round(float(temp), 1),
            'flow_rate': round(float(flow), 2),
            'filter_status': 'Monitor'
        })
        
    # 3. Service Required (Clogged Membrane or Pore Breakthrough)
    for _ in range(n_service):
        t_in = np.random.uniform(350, 950)
        temp = np.random.normal(26.0, 4.0)
        temp = np.clip(temp, 15.0, 40.0)
        
        failure_mode = np.random.choice(['pore_damage', 'severe_fouling', 'combined'])
        
        if failure_mode == 'pore_damage':
            # Severe rejection loss (rejection < 70%)
            rejection = np.random.uniform(0.40, 0.69)
            t_out = t_in * (1.0 - rejection)
            flow = np.random.normal(1.30, 0.20)
        elif failure_mode == 'severe_fouling':
            # Severe flow choke (flow < 0.95 L/min)
            rejection = np.random.uniform(0.68, 0.78)
            t_out = t_in * (1.0 - rejection)
            flow = np.random.uniform(0.35, 0.85)
        else: # combined
            rejection = np.random.uniform(0.45, 0.65)
            t_out = t_in * (1.0 - rejection)
            flow = np.random.uniform(0.40, 0.90)
            
        t_out += np.random.normal(0, 4.0)
        t_out = max(60.0, t_out)
        flow = max(0.20, float(flow))
        
        records.append({
            'tds_in': round(float(t_in), 1),
            'tds_out': round(float(t_out), 1),
            'temperature': round(float(temp), 1),
            'flow_rate': round(float(flow), 2),
            'filter_status': 'Service Required'
        })
        
    # Shuffle and attach timestamps
    df = pd.DataFrame(records)
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    df.insert(0, 'timestamp', [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps])
    
    # Ensure directory exists and save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully generated with {len(df)} samples: {output_path}")
    print(df['filter_status'].value_counts())
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic sensor dataset for Aqua-Swachh")
    parser.add_argument("--samples", type=int, default=1500, help="Number of records to generate")
    parser.add_argument("--output", type=str, default="data/sensor_data.csv", help="Output CSV path")
    args = parser.parse_args()
    
    generate_water_sensor_dataset(n_samples=args.samples, output_path=args.output)
