"""
Aqua-Swachh ESP32 Serial Logger & Live Inference Bridge
======================================================
Reads sensor readings (TDS1, TDS2, Temperature, Flow) from ESP32 via USB Serial,
logs them to CSV, and optionally runs real-time ML inference.

Supports:
- Live Serial mode (ESP32 connected over COM port)
- Mock/Simulation mode (generates live streaming readings for demonstration)
"""

import os
import sys
import time
import argparse
from datetime import datetime
import pandas as pd

# Optional pyserial import
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

CSV_FILE = "data/sensor_data.csv"
CSV_HEADER = "timestamp,tds_in,tds_out,temperature,flow_rate\n"

def log_reading_to_csv(filepath: str, tds_in: float, tds_out: float, temp: float, flow: float, timestamp: str = None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    file_exists = os.path.exists(filepath)
    with open(filepath, "a", encoding="utf-8") as f:
        if not file_exists:
            f.write(CSV_HEADER)
        f.write(f"{timestamp},{tds_in:.1f},{tds_out:.1f},{temp:.1f},{flow:.2f}\n")

def run_serial_logger(port: str, baudrate: int = 115200, output_csv: str = CSV_FILE, run_inference: bool = True):
    if not SERIAL_AVAILABLE:
        print("[Error] 'pyserial' package is not installed. Run: pip install pyserial")
        sys.exit(1)
        
    predictor = None
    if run_inference:
        try:
            from inference.predict import FilterHealthPredictor
            predictor = FilterHealthPredictor()
            print("[INFO] ML Filter Health Predictor loaded successfully.")
        except Exception as e:
            print(f"[WARN] Could not load ML model for live inference ({e}). Logging only.")

    print(f"[INFO] Connecting to ESP32 on port {port} at {baudrate} baud...")
    try:
        ser = serial.Serial(port, baudrate, timeout=2.0)
        time.sleep(2) # Wait for ESP32 reboot on connection
        print("[INFO] Connected. Listening for sensor packets...")
        print("Expected format from ESP32: 'tds_in,tds_out,temperature,flow_rate' (e.g. 450.0,35.0,26.5,1.80)")
        
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue
            
            # Skip debug text if ESP32 prints boot messages
            parts = line.split(',')
            if len(parts) >= 4:
                try:
                    tds_in = float(parts[0])
                    tds_out = float(parts[1])
                    temp = float(parts[2])
                    flow = float(parts[3])
                    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    log_reading_to_csv(output_csv, tds_in, tds_out, temp, flow, now_str)
                    
                    status_str = ""
                    if predictor:
                        pred = predictor.predict_sample(tds_in, tds_out, temp, flow)
                        status_str = f" | Status: {pred['predicted_status']} ({pred['confidence']*100:.1f}%)"
                        
                    print(f"[{now_str}] In: {tds_in} ppm | Out: {tds_out} ppm | Temp: {temp}°C | Flow: {flow} L/min{status_str}")
                except ValueError:
                    print(f"[RAW ESP32]: {line}")
    except KeyboardInterrupt:
        print("\n[INFO] Logging stopped by user.")
    except Exception as ex:
        print(f"[ERROR] Serial communication error: {ex}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aqua-Swachh ESP32 Serial Logger")
    parser.add_argument("--port", type=str, default="COM3", help="Serial port (e.g. COM3 or /dev/ttyUSB0)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--output", type=str, default=CSV_FILE, help="CSV log path")
    parser.add_argument("--no-inference", action="store_true", help="Disable real-time ML prediction")
    args = parser.parse_args()
    
    run_serial_logger(port=args.port, baudrate=args.baud, output_csv=args.output, run_inference=not args.no_inference)
