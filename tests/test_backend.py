"""
Aqua-Swachh: Comprehensive Backend Verification Suite
======================================================
Tests:
1. Physics feature engineering (salt rejection, temp compensation, flow index, delta TDS)
2. Predictor model loading, single-sample inference, probability distribution
3. Deterministic safety guardrail overrides (high permeate TDS & low salt rejection)
4. Batch prediction logic & CSV parser
5. Live REST API endpoints via HTTP against running server
"""

import unittest
import json
import urllib.request
import urllib.parse
import pandas as pd
from model.features import (
    compute_physics_features,
    extract_features,
    FEATURE_COLUMNS
)
from inference.predict import FilterHealthPredictor

class TestPhysicsFeatures(unittest.TestCase):
    def test_salt_rejection_calculation(self):
        raw_df = pd.DataFrame([{
            'tds_in': 420.0,
            'tds_out': 28.0,
            'temperature': 25.0,
            'flow_rate': 1.95
        }])
        phys = compute_physics_features(raw_df)
        rejection = phys['salt_rejection'].iloc[0]
        self.assertAlmostEqual(rejection, 1.0 - (28.0 / 420.0), places=4)

    def test_salt_rejection_zero_feed(self):
        # Guard against division by zero
        raw_df = pd.DataFrame([{
            'tds_in': 0.0,
            'tds_out': 0.0,
            'temperature': 25.0,
            'flow_rate': 1.0
        }])
        phys = compute_physics_features(raw_df)
        self.assertFalse(pd.isna(phys['salt_rejection'].iloc[0]))

    def test_temperature_compensation(self):
        raw_df = pd.DataFrame([{
            'tds_in': 500.0,
            'tds_out': 50.0,
            'temperature': 25.0,
            'flow_rate': 2.0
        }])
        phys = compute_physics_features(raw_df)
        self.assertAlmostEqual(phys['tds_out_norm_25c'].iloc[0], 50.0, places=4)

    def test_delta_tds(self):
        raw_df = pd.DataFrame([{
            'tds_in': 500.0,
            'tds_out': 50.0,
            'temperature': 25.0,
            'flow_rate': 2.0
        }])
        phys = compute_physics_features(raw_df)
        self.assertEqual(phys['delta_tds'].iloc[0], 450.0)

    def test_feature_dataframe_structure(self):
        raw_df = pd.DataFrame([{
            'tds_in': 420.0,
            'tds_out': 28.0,
            'temperature': 25.5,
            'flow_rate': 1.95
        }])
        feats = extract_features(raw_df)
        self.assertEqual(list(feats.columns), FEATURE_COLUMNS)
        self.assertEqual(len(feats), 1)


class TestInferencePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = FilterHealthPredictor(model_path="model/saved_models/random_forest_filter_model.joblib")

    def test_nominal_sample(self):
        result = self.predictor.predict_sample(tds_in=420.0, tds_out=28.0, temperature=25.5, flow_rate=1.95)
        self.assertEqual(result["predicted_status"], "Normal")
        self.assertGreaterEqual(result["confidence"], 0.80)
        self.assertIn("Normal", result["probabilities"])
        self.assertIn("Monitor", result["probabilities"])
        self.assertIn("Service Required", result["probabilities"])
        self.assertFalse(result["guardrail_triggered"])

    def test_degraded_sample(self):
        result = self.predictor.predict_sample(tds_in=510.0, tds_out=115.0, temperature=26.8, flow_rate=1.25)
        self.assertEqual(result["predicted_status"], "Monitor")

    def test_clogged_flow_sample(self):
        result = self.predictor.predict_sample(tds_in=480.0, tds_out=120.0, temperature=26.0, flow_rate=0.65)
        self.assertEqual(result["predicted_status"], "Service Required")

    def test_safety_guardrail_high_permeate_tds(self):
        # Permeate TDS >= 250 ppm triggers failsafe rule
        result = self.predictor.predict_sample(tds_in=600.0, tds_out=260.0, temperature=25.0, flow_rate=1.8)
        self.assertEqual(result["predicted_status"], "Service Required")
        self.assertTrue(result["guardrail_triggered"])
        self.assertIn("permeate tds", result["guardrail_reason"].lower())

    def test_safety_guardrail_low_salt_rejection(self):
        # Salt rejection < 55% triggers failsafe rule
        result = self.predictor.predict_sample(tds_in=300.0, tds_out=150.0, temperature=25.0, flow_rate=1.5)
        self.assertEqual(result["predicted_status"], "Service Required")
        self.assertTrue(result["guardrail_triggered"])


class TestLiveApiEndpoints(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:8000"

    def test_predict_api(self):
        url = f"{self.BASE_URL}/api/predict"
        payload = json.dumps({
            "tds_in": 420.0,
            "tds_out": 28.0,
            "temperature": 25.5,
            "flow_rate": 1.95
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertEqual(data["predicted_status"], "Normal")
            self.assertIn("physics_indicators", data)
            self.assertIn("explanation", data)

    def test_scenarios_api(self):
        url = f"{self.BASE_URL}/api/scenarios"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            scenarios = json.loads(resp.read().decode("utf-8"))
            self.assertGreaterEqual(len(scenarios), 5)

    def test_metrics_api(self):
        url = f"{self.BASE_URL}/api/metrics"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            metrics = json.loads(resp.read().decode("utf-8"))
            self.assertIn("test_accuracy", metrics)
            self.assertIn("comparison", metrics)
            self.assertIn("Random Forest Classifier (Primary Model)", metrics["comparison"])


    def test_system_telemetry_api(self):
        url = f"{self.BASE_URL}/api/system-telemetry"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("hardware", data)
            self.assertIn("purifier_stats", data)

    def test_load_sample_csv_api(self):
        url = f"{self.BASE_URL}/api/load-sample-csv"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertGreaterEqual(data["total_records"], 100)
            self.assertIn("Normal", data["breakdown"])
            self.assertIn("Monitor", data["breakdown"])
            self.assertIn("Service Required", data["breakdown"])
            self.assertIn("rows_preview", data)

    def test_stream_sample_api(self):
        url = f"{self.BASE_URL}/api/stream-sample?state=Normal"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertIn("tds_in", data)
            self.assertIn("tds_out", data)
            self.assertIn("temperature", data)
            self.assertIn("flow_rate", data)

if __name__ == "__main__":
    unittest.main()
