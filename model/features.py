"""
Aqua-Swachh Feature Engineering Pipeline
========================================
Extracts domain-specific and physics-constrained features from water purifier sensors:
- Salt Rejection Efficiency (eta = 1 - TDS_out / TDS_in)
- Temperature-Compensated Rejection
- Absolute TDS Delta
- Flow Efficiency Indicator
- Rolling statistics (if sequential data is present)
"""

import numpy as np
import pandas as pd
from typing import List, Union, Dict

FEATURE_COLUMNS = [
    'tds_in',
    'tds_out',
    'temperature',
    'flow_rate',
    'salt_rejection',
    'delta_tds',
    'tds_out_norm_25c',
    'temp_comp_rejection',
    'flow_rejection_index'
]

def compute_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes physics-guided features from raw sensor readings.
    Works for both batch DataFrames and single-sample DataFrames.
    """
    data = df.copy()
    
    # 1. Salt Rejection Efficiency (eta = 1 - TDS_out / TDS_in)
    # Clip TDS_in to avoid division by zero
    safe_tds_in = np.maximum(data['tds_in'].values, 1.0)
    data['salt_rejection'] = 1.0 - (data['tds_out'].values / safe_tds_in)
    data['salt_rejection'] = np.clip(data['salt_rejection'], -1.0, 1.0)
    
    # 2. Absolute TDS Delta (amount of dissolved solids removed)
    data['delta_tds'] = data['tds_in'].values - data['tds_out'].values
    
    # 3. Temperature Normalization (Reference: 25°C, ~2% conductivity change per °C)
    temp_factor = 1.0 + 0.02 * (data['temperature'].values - 25.0)
    temp_factor = np.maximum(temp_factor, 0.5) # Guard against extreme low values
    data['tds_out_norm_25c'] = data['tds_out'].values / temp_factor
    data['tds_in_norm_25c'] = safe_tds_in / temp_factor
    data['temp_comp_rejection'] = 1.0 - (data['tds_out_norm_25c'] / data['tds_in_norm_25c'])
    data['temp_comp_rejection'] = np.clip(data['temp_comp_rejection'], -1.0, 1.0)
    
    # 4. Flow Rejection Composite Index
    # When filters clog, flow rate plummets even before salt rejection crashes.
    # When membranes tear, rejection plummets while flow may stay high.
    data['flow_rejection_index'] = data['flow_rate'].values * np.maximum(data['salt_rejection'].values, 0.0)
    
    return data

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw sensor dataframe and returns only model feature columns.
    """
    featured_df = compute_physics_features(df)
    return featured_df[FEATURE_COLUMNS]
