"""
Aqua-Swachh Real-Time Filter Condition Inference Engine
=======================================================
Provides high-level inference for single or streaming sensor readings.
Includes:
- Physics feature extraction
- Model prediction with probability confidence scores
- Deterministic safety guardrails (failsafe rules)
- User-friendly diagnostic messaging
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, List

from model.features import extract_features, compute_physics_features

DEFAULT_MODEL_PATH = "model/saved_models/random_forest_filter_model.joblib"

class FilterHealthPredictor:
    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                "Please run `python -m model.train` to train and save the model first."
            )
        self.model = joblib.load(model_path)
        self.classes_ = list(self.model.classes_)

    def predict_sample(
        self,
        tds_in: float,
        tds_out: float,
        temperature: float,
        flow_rate: float
    ) -> Dict[str, Any]:
        """
        Predicts filter health for a single sensor reading snapshot.
        """
        raw_df = pd.DataFrame([{
            'tds_in': float(tds_in),
            'tds_out': float(tds_out),
            'temperature': float(temperature),
            'flow_rate': float(flow_rate)
        }])
        
        # 1. Physics Features
        physics_df = compute_physics_features(raw_df)
        rejection_pct = float(physics_df['salt_rejection'].iloc[0]) * 100.0
        temp_comp_rejection_pct = float(physics_df['temp_comp_rejection'].iloc[0]) * 100.0
        delta_tds = float(physics_df['delta_tds'].iloc[0])
        
        # 2. Extract model features and predict
        X = extract_features(raw_df)
        predicted_status = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(self.classes_, probabilities)}
        
        # 3. Deterministic Safety Guardrails (Fail-Safe Checks)
        # Prevents any false-negative "Normal" if severe physical contamination or flow blockage is detected
        guardrail_triggered = False
        guardrail_reason = ""
        
        if tds_out >= 250.0:
            guardrail_triggered = True
            guardrail_reason = f"High permeate TDS detected ({tds_out:.1f} ppm >= 250 ppm safety ceiling)"
            predicted_status = "Service Required"
        elif rejection_pct < 55.0:
            guardrail_triggered = True
            guardrail_reason = f"Severe salt rejection loss ({rejection_pct:.1f}% < 55% threshold)"
            predicted_status = "Service Required"
        elif flow_rate < 0.40 and tds_in > 0:
            guardrail_triggered = True
            guardrail_reason = f"Severe flow choke / membrane clogging ({flow_rate:.2f} L/min < 0.40 L/min)"
            predicted_status = "Service Required"
            
        # 4. User Action & Diagnosis Recommendation
        action_recommendation = self._get_recommendation(predicted_status)
        
        return {
            'predicted_status': predicted_status,
            'confidence': float(max(probabilities)),
            'probabilities': prob_dict,
            'guardrail_triggered': guardrail_triggered,
            'guardrail_reason': guardrail_reason,
            'physics_indicators': {
                'salt_rejection_pct': round(rejection_pct, 2),
                'temp_compensated_rejection_pct': round(temp_comp_rejection_pct, 2),
                'delta_tds_ppm': round(delta_tds, 1),
                'water_temperature_c': round(temperature, 1),
                'flow_rate_lpm': round(flow_rate, 2)
            },
            'action_recommendation': action_recommendation
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs batch inference on a pandas DataFrame.
        """
        results = []
        for _, row in df.iterrows():
            res = self.predict_sample(
                tds_in=row['tds_in'],
                tds_out=row['tds_out'],
                temperature=row['temperature'],
                flow_rate=row['flow_rate']
            )
            results.append({
                'predicted_status': res['predicted_status'],
                'confidence': res['confidence'],
                'salt_rejection_pct': res['physics_indicators']['salt_rejection_pct'],
                'guardrail_triggered': res['guardrail_triggered']
            })
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    @staticmethod
    def _get_recommendation(status: str) -> str:
        if status == "Normal":
            return "Filter operating within nominal parameters. Water quality and flow are optimal."
        elif status == "Monitor":
            return "Early degradation or fouling detected. Purifier is functional; prepare for cartridge swap in coming cycles."
        elif status == "Service Required":
            return "Filter condition degraded beyond acceptable threshold. Cartridge service/replacement required."
        return "Unknown condition."

if __name__ == "__main__":
    predictor = FilterHealthPredictor()
    
    # Test cases representing three operational states
    test_samples = [
        {"name": "Fresh Filter", "tds_in": 450, "tds_out": 30, "temperature": 26.0, "flow_rate": 1.9},
        {"name": "Aging Filter", "tds_in": 480, "tds_out": 105, "temperature": 27.2, "flow_rate": 1.3},
        {"name": "Clogged / Damaged Filter", "tds_in": 520, "tds_out": 220, "temperature": 28.0, "flow_rate": 0.6}
    ]
    
    print("\n--- Model Inference Test Cases ---")
    for sample in test_samples:
        pred = predictor.predict_sample(
            tds_in=sample["tds_in"],
            tds_out=sample["tds_out"],
            temperature=sample["temperature"],
            flow_rate=sample["flow_rate"]
        )
        print(f"\nCase: {sample['name']}")
        print(f"Inputs: TDS In={sample['tds_in']}, TDS Out={sample['tds_out']}, Temp={sample['temperature']}°C, Flow={sample['flow_rate']} L/min")
        print(f"Prediction: {pred['predicted_status']} (Confidence: {pred['confidence']*100:.1f}%)")
        print(f"Rejection Efficiency: {pred['physics_indicators']['salt_rejection_pct']}%")
        print(f"Action: {pred['action_recommendation']}")
