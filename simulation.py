"""
Water Distribution Network Simulation Module (Upgraded)
Simulates real-time sensor data for 3 regions with realistic physical dependencies.
Engineering-grade simulation with parameter interactions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import random


# Realistic sensor value ranges for drinking water (expanded for extreme scenarios)
SENSOR_RANGES = {
    'temperature': {'min': 10, 'max': 30, 'unit': '°C'},
    'flow': {'min': 0.5, 'max': 5.0, 'unit': 'm³/h'},
    'pressure': {'min': 2.0, 'max': 6.0, 'unit': 'bar'},
    'chlorine': {'min': 0.1, 'max': 2.5, 'unit': 'mg/L'},
    'pH': {'min': 6.0, 'max': 9.0, 'unit': ''},
    'turbidity': {'min': 0.1, 'max': 8.0, 'unit': 'NTU'},  # Expanded for severe contamination
    'conductivity': {'min': 150, 'max': 1000, 'unit': 'µS/cm'}  # Expanded range
}

# Ideal values for normalized deviation risk calculation
IDEAL_VALUES = {
    'chlorine': {'ideal': 1.0, 'safe_min': 0.5, 'safe_max': 2.0},
    'pH': {'ideal': 7.2, 'safe_min': 6.5, 'safe_max': 8.5},
    'turbidity': {'ideal': 0.3, 'safe_min': 0.0, 'safe_max': 1.0},
    'pressure': {'ideal': 4.0, 'safe_min': 3.0, 'safe_max': 6.0},
    'flow': {'ideal': 2.0, 'safe_min': 1.0, 'safe_max': 5.0},
    'conductivity': {'ideal': 400, 'safe_min': 200, 'safe_max': 600}  # µS/cm
}

# Risk weights for each parameter (total = 100)
RISK_WEIGHTS = {
    'chlorine': 25,       # Most critical for disinfection
    'turbidity': 20,      # Indicates contamination
    'pH': 12,             # Chemical balance
    'pressure': 13,       # Leak/intrusion indicator
    'flow': 12,           # Stagnation indicator
    'conductivity': 8,    # Dissolved solids indicator
    'temporal': 10        # Time-based risk patterns
}

# Region-specific baseline characteristics
REGION_PROFILES = {
    'A': {'temp_offset': 0, 'pressure_factor': 1.0, 'flow_factor': 1.0, 'conductivity_base': 450, 'name': 'Industrial Zone'},
    'B': {'temp_offset': 2, 'pressure_factor': 0.9, 'flow_factor': 0.85, 'conductivity_base': 380, 'name': 'Residential Area'},
    'C': {'temp_offset': -1, 'pressure_factor': 1.1, 'flow_factor': 1.1, 'conductivity_base': 420, 'name': 'Commercial District'}
}

# Temporal risk patterns (hour of day)
# Higher risk during low-usage hours (stagnation) and peak hours (pressure drops)
TEMPORAL_RISK_FACTORS = {
    'night': {'hours': range(0, 6), 'risk_modifier': 1.15, 'reason': 'Low usage - stagnation risk'},
    'morning_peak': {'hours': range(6, 9), 'risk_modifier': 1.1, 'reason': 'Morning peak - pressure stress'},
    'day': {'hours': range(9, 17), 'risk_modifier': 1.0, 'reason': 'Normal operation'},
    'evening_peak': {'hours': range(17, 21), 'risk_modifier': 1.08, 'reason': 'Evening peak - high demand'},
    'late_evening': {'hours': range(21, 24), 'risk_modifier': 1.05, 'reason': 'Declining usage'}
}


def generate_base_temperature(region: str) -> float:
    """Generate base temperature with regional offset and time-based variation."""
    base = 18 + REGION_PROFILES[region]['temp_offset']
    variation = np.random.normal(0, 2)
    return np.clip(base + variation, SENSOR_RANGES['temperature']['min'], 
                   SENSOR_RANGES['temperature']['max'])


def generate_flow(region: str) -> float:
    """Generate flow rate based on region profile."""
    base_flow = 2.5 * REGION_PROFILES[region]['flow_factor']
    variation = np.random.normal(0, 0.5)
    return np.clip(base_flow + variation, SENSOR_RANGES['flow']['min'], 
                   SENSOR_RANGES['flow']['max'])


def generate_pressure(region: str) -> float:
    """Generate pressure based on region profile."""
    base_pressure = 4.0 * REGION_PROFILES[region]['pressure_factor']
    variation = np.random.normal(0, 0.3)
    return np.clip(base_pressure + variation, SENSOR_RANGES['pressure']['min'], 
                   SENSOR_RANGES['pressure']['max'])


def generate_pH() -> float:
    """Generate pH value within safe drinking water range."""
    base_pH = 7.2
    variation = np.random.normal(0, 0.3)
    return np.clip(base_pH + variation, SENSOR_RANGES['pH']['min'], 
                   SENSOR_RANGES['pH']['max'])


def generate_conductivity(region: str, temperature: float) -> float:
    """
    Generate electrical conductivity based on region and temperature.
    Conductivity increases with temperature (~2% per °C).
    High conductivity indicates high dissolved solids (potential contamination).
    """
    base_cond = REGION_PROFILES[region]['conductivity_base']
    # Temperature coefficient: conductivity increases ~2% per degree above 20°C
    temp_factor = 1 + 0.02 * (temperature - 20)
    conductivity = base_cond * temp_factor
    variation = np.random.normal(0, 30)
    return np.clip(conductivity + variation, SENSOR_RANGES['conductivity']['min'], 
                   SENSOR_RANGES['conductivity']['max'])


def get_temporal_features(timestamp: datetime = None) -> dict:
    """
    Extract temporal features from timestamp for time-based risk patterns.
    
    Returns:
        dict with hour, day_of_week, is_weekend, time_period, risk_modifier
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    hour = timestamp.hour
    day_of_week = timestamp.weekday()  # 0=Monday, 6=Sunday
    is_weekend = day_of_week >= 5
    
    # Determine time period and risk modifier
    time_period = 'day'
    risk_modifier = 1.0
    period_reason = 'Normal operation'
    
    for period_name, period_info in TEMPORAL_RISK_FACTORS.items():
        if hour in period_info['hours']:
            time_period = period_name
            risk_modifier = period_info['risk_modifier']
            period_reason = period_info['reason']
            break
    
    # Weekends have different usage patterns
    if is_weekend:
        if time_period == 'morning_peak':
            risk_modifier *= 0.9  # Less morning rush on weekends
        elif time_period == 'night':
            risk_modifier *= 1.05  # Potentially longer stagnation
    
    return {
        'hour': hour,
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'time_period': time_period,
        'temporal_risk_modifier': round(risk_modifier, 3),
        'period_reason': period_reason
    }


def apply_physical_dependencies(temperature: float, pressure: float, flow: float, 
                                 base_chlorine: float = 1.2, base_turbidity: float = 0.4) -> tuple:
    """
    Apply realistic physical dependencies between parameters.
    
    Physical relationships modeled:
    - Pressure drop increases turbidity (sediment disturbance)
    - High temperature increases chlorine decay
    - Low flow increases chlorine decay (stagnation effect)
    - Combined effects for realistic behavior
    """
    chlorine = base_chlorine
    turbidity = base_turbidity
    
    # Pressure effect on turbidity: low pressure stirs up sediments
    if pressure < 3.0:
        turbidity += (3.0 - pressure) * 0.8
    
    # Temperature effect on chlorine: heat accelerates decay
    if temperature > 25:
        chlorine -= (temperature - 25) * 0.03
    
    # Flow effect on chlorine: stagnation reduces chlorine faster
    if flow < 1.0:
        chlorine -= 0.25
    elif flow < 1.5:
        chlorine -= 0.1
    
    # Combined effect: low flow + high temp = worse chlorine decay
    if flow < 1.5 and temperature > 22:
        chlorine -= 0.05
    
    # Pressure drop can also affect chlorine (intrusion risk)
    if pressure < 2.5:
        chlorine -= 0.1
        turbidity += 0.3
    
    # Add natural variation
    chlorine += np.random.normal(0, 0.05)
    turbidity += np.random.uniform(0, 0.15)
    
    # Clip to valid ranges
    chlorine = np.clip(chlorine, SENSOR_RANGES['chlorine']['min'], 
                       SENSOR_RANGES['chlorine']['max'])
    turbidity = np.clip(turbidity, SENSOR_RANGES['turbidity']['min'], 
                        SENSOR_RANGES['turbidity']['max'])
    
    return chlorine, turbidity


def simulate_sensor_data(region: str = 'A', contamination_event: bool = False, 
                         timestamp: datetime = None) -> dict:
    """
    Generate a single row of simulated sensor data with physical dependencies.
    
    Args:
        region: Region identifier ('A', 'B', or 'C')
        contamination_event: If True, simulate abnormal sensor behavior
        timestamp: Optional timestamp for temporal features (uses now if None)
    
    Returns:
        Dictionary containing sensor readings with temporal features
    """
    if region not in REGION_PROFILES:
        raise ValueError(f"Invalid region: {region}. Must be 'A', 'B', or 'C'")
    
    if timestamp is None:
        timestamp = datetime.now()
    
    # Generate independent base values
    temperature = generate_base_temperature(region)
    flow = generate_flow(region)
    pressure = generate_pressure(region)
    pH = generate_pH()
    
    # Generate conductivity (depends on temperature)
    conductivity = generate_conductivity(region, temperature)
    
    # Get temporal features
    temporal = get_temporal_features(timestamp)
    
    # Apply physical dependencies for chlorine and turbidity
    chlorine, turbidity = apply_physical_dependencies(
        temperature, pressure, flow,
        base_chlorine=1.2, base_turbidity=0.4
    )
    
    # Apply contamination event effects
    if contamination_event:
        event_type = random.choice(['chemical', 'physical', 'biological', 'leak'])
        
        if event_type == 'chemical':
            pH = np.clip(pH + random.choice([-1.5, 1.5]), 5.5, 9.5)
            chlorine = max(0.1, chlorine * 0.3)
            conductivity = min(900, conductivity * 1.5)  # Chemical contamination increases conductivity
        elif event_type == 'physical':
            pressure = max(1.8, pressure * 0.5)
            turbidity = min(8, turbidity * 5)
            conductivity = min(850, conductivity * 1.3)
        elif event_type == 'biological':
            temperature = min(35, temperature + 5)
            chlorine = max(0.1, chlorine * 0.35)
            turbidity = min(6, turbidity * 3)
        else:  # leak
            pressure = max(1.5, pressure * 0.4)
            turbidity = min(7, turbidity * 4)
            chlorine = max(0.15, chlorine * 0.5)
            conductivity = min(800, conductivity * 1.2)
    
    return {
        'timestamp': timestamp,
        'region': region,
        'temperature': round(temperature, 2),
        'flow': round(flow, 2),
        'pressure': round(pressure, 2),
        'chlorine': round(chlorine, 2),
        'pH': round(pH, 2),
        'turbidity': round(turbidity, 2),
        'conductivity': round(conductivity, 1),
        'hour': temporal['hour'],
        'day_of_week': temporal['day_of_week'],
        'is_weekend': temporal['is_weekend'],
        'time_period': temporal['time_period'],
        'temporal_risk_modifier': temporal['temporal_risk_modifier']
    }


def generate_training_dataset(n_samples: int = 1000, include_anomalies: bool = True) -> pd.DataFrame:
    """
    Generate a comprehensive dataset for model training with diverse scenarios.
    Includes normal operations, various anomaly types, edge cases, and interaction scenarios.
    """
    data = []
    regions = ['A', 'B', 'C']
    
    # Generate varied timestamps for temporal diversity
    base_time = datetime.now()
    
    # Scenario distribution (percentages)
    # 60% normal, 40% various anomalies/edge cases
    scenario_distribution = {
        'normal': 0.60,
        'contamination': 0.10,        # Standard contamination events
        'pipe_leak': 0.05,            # Low pressure + high turbidity
        'stagnation': 0.05,           # Low flow + low chlorine
        'chemical_spill': 0.03,       # pH extreme + high conductivity
        'disinfectant_failure': 0.04, # Very low chlorine
        'pressure_surge': 0.03,       # Sudden pressure changes
        'gradual_decay': 0.04,        # Slow chlorine decay
        'night_stagnation': 0.03,     # Night + low flow scenarios
        'peak_demand': 0.03           # High demand periods
    }
    
    for i in range(n_samples):
        region = random.choice(regions)
        
        # Vary the timestamp
        hour_offset = random.randint(0, 23)
        day_offset = random.randint(0, 6)
        simulated_time = base_time.replace(
            hour=hour_offset,
            minute=random.randint(0, 59)
        )
        from datetime import timedelta
        simulated_time = simulated_time - timedelta(days=day_offset)
        
        # Select scenario based on distribution
        scenario = random.choices(
            list(scenario_distribution.keys()),
            weights=list(scenario_distribution.values())
        )[0]
        
        is_anomaly = scenario != 'normal'
        
        if scenario == 'normal':
            row = simulate_sensor_data(region=region, contamination_event=False, timestamp=simulated_time)
        
        elif scenario == 'contamination':
            row = simulate_sensor_data(region=region, contamination_event=True, timestamp=simulated_time)
        
        elif scenario == 'pipe_leak':
            # Specific pipe leak scenario: low pressure + high turbidity
            row = simulate_sensor_data(region=region, contamination_event=False, timestamp=simulated_time)
            row['pressure'] = round(random.uniform(1.5, 2.4), 2)
            row['turbidity'] = round(random.uniform(2.5, 6.0), 2)
            row['flow'] = round(max(0.5, row['flow'] * 0.7), 2)
        
        elif scenario == 'stagnation':
            # Dead-end pipe stagnation: low flow + chlorine decay
            row = simulate_sensor_data(region=region, contamination_event=False, timestamp=simulated_time)
            row['flow'] = round(random.uniform(0.3, 0.9), 2)
            row['chlorine'] = round(random.uniform(0.15, 0.45), 2)
            row['turbidity'] = round(random.uniform(0.8, 2.0), 2)
        
        elif scenario == 'chemical_spill':
            # Industrial chemical contamination
            row = simulate_sensor_data(region=region, contamination_event=False, timestamp=simulated_time)
            row['pH'] = round(random.choice([random.uniform(5.5, 6.3), random.uniform(8.6, 9.2)]), 2)
            row['conductivity'] = round(random.uniform(700, 950), 1)
            row['chlorine'] = round(random.uniform(0.2, 0.5), 2)
        
        elif scenario == 'disinfectant_failure':
            # Chlorine dosing system failure
            row = simulate_sensor_data(region=region, contamination_event=False, timestamp=simulated_time)
            row['chlorine'] = round(random.uniform(0.05, 0.25), 2)
            # Everything else normal - this is the dangerous subtle failure
        
        elif scenario == 'pressure_surge':
            # Main break or valve issue causing pressure problems
            row = simulate_sensor_data(region=region, contamination_event=False, timestamp=simulated_time)
            row['pressure'] = round(random.choice([random.uniform(1.8, 2.3), random.uniform(5.5, 6.5)]), 2)
            row['turbidity'] = round(min(5.0, row['turbidity'] * 2.5), 2)
        
        elif scenario == 'gradual_decay':
            # Slow degradation over distance from treatment plant
            row = simulate_sensor_data(region=region, contamination_event=False, timestamp=simulated_time)
            decay_factor = random.uniform(0.4, 0.7)
            row['chlorine'] = round(row['chlorine'] * decay_factor, 2)
            row['turbidity'] = round(min(3.0, row['turbidity'] * 1.5), 2)
        
        elif scenario == 'night_stagnation':
            # Night-time low usage causing stagnation
            simulated_time = simulated_time.replace(hour=random.randint(1, 5))
            row = simulate_sensor_data(region=region, contamination_event=False, timestamp=simulated_time)
            row['flow'] = round(random.uniform(0.4, 0.8), 2)
            row['chlorine'] = round(row['chlorine'] * 0.6, 2)
        
        elif scenario == 'peak_demand':
            # Peak usage causing pressure drops
            simulated_time = simulated_time.replace(hour=random.choice([7, 8, 18, 19]))
            row = simulate_sensor_data(region=region, contamination_event=False, timestamp=simulated_time)
            row['flow'] = round(random.uniform(3.5, 4.8), 2)
            row['pressure'] = round(max(2.5, row['pressure'] * 0.8), 2)
        
        row['is_anomaly'] = is_anomaly
        row['scenario'] = scenario
        data.append(row)
    
    return pd.DataFrame(data)


def get_live_data(region: str = 'A', contamination_event: bool = False) -> pd.DataFrame:
    """Get a single row of live data as a DataFrame."""
    data = simulate_sensor_data(region=region, contamination_event=contamination_event)
    return pd.DataFrame([data])


# ============================================================
# ADVANCED RISK INDEX (Normalized Deviation Model)
# ============================================================

def calculate_deviation_risk(value: float, ideal: float, safe_min: float, safe_max: float) -> float:
    """
    Calculate normalized deviation from ideal value.
    Returns a value between 0 and ~2 representing severity of deviation.
    """
    if value < ideal:
        acceptable_range = ideal - safe_min
        if acceptable_range == 0:
            return 0
        deviation = (ideal - value) / acceptable_range
    else:
        acceptable_range = safe_max - ideal
        if acceptable_range == 0:
            return 0
        deviation = (value - ideal) / acceptable_range
    
    return min(max(deviation, 0), 2.0)


def calculate_risk_index(data: dict) -> tuple:
    """
    Calculate Water Quality Risk Index (0-100) using normalized deviation model.
    
    Formula:
    - For each parameter: deviation = abs(value - ideal) / acceptable_range
    - risk_component = deviation * weight
    - Plus interaction risks for dangerous combinations
    - Plus temporal risk modifier based on time patterns
    
    Returns:
        Tuple of (risk_score, risk_breakdown)
    """
    risk_score = 0
    risk_breakdown = {}
    
    chlorine = data.get('chlorine', 1.0)
    turbidity = data.get('turbidity', 0.5)
    pH = data.get('pH', 7.2)
    pressure = data.get('pressure', 4.0)
    flow = data.get('flow', 2.0)
    conductivity = data.get('conductivity', 400)
    temporal_modifier = data.get('temporal_risk_modifier', 1.0)
    
    # Chlorine risk (weight: 25)
    chlorine_dev = calculate_deviation_risk(
        chlorine, IDEAL_VALUES['chlorine']['ideal'],
        IDEAL_VALUES['chlorine']['safe_min'], IDEAL_VALUES['chlorine']['safe_max']
    )
    chlorine_risk = chlorine_dev * RISK_WEIGHTS['chlorine']
    risk_breakdown['chlorine_risk'] = round(chlorine_risk, 1)
    risk_score += chlorine_risk
    
    # Turbidity risk (weight: 20) - lower is better
    if turbidity <= IDEAL_VALUES['turbidity']['ideal']:
        turbidity_dev = 0
    else:
        acceptable_range = IDEAL_VALUES['turbidity']['safe_max'] - IDEAL_VALUES['turbidity']['ideal']
        turbidity_dev = (turbidity - IDEAL_VALUES['turbidity']['ideal']) / acceptable_range
        turbidity_dev = min(turbidity_dev, 2.0)
    turbidity_risk = turbidity_dev * RISK_WEIGHTS['turbidity']
    risk_breakdown['turbidity_risk'] = round(turbidity_risk, 1)
    risk_score += turbidity_risk
    
    # pH risk (weight: 12)
    pH_dev = calculate_deviation_risk(
        pH, IDEAL_VALUES['pH']['ideal'],
        IDEAL_VALUES['pH']['safe_min'], IDEAL_VALUES['pH']['safe_max']
    )
    pH_risk = pH_dev * RISK_WEIGHTS['pH']
    risk_breakdown['pH_risk'] = round(pH_risk, 1)
    risk_score += pH_risk
    
    # Pressure risk (weight: 13) - low pressure is bad
    if pressure >= IDEAL_VALUES['pressure']['ideal']:
        pressure_dev = 0
    else:
        acceptable_range = IDEAL_VALUES['pressure']['ideal'] - IDEAL_VALUES['pressure']['safe_min']
        pressure_dev = (IDEAL_VALUES['pressure']['ideal'] - pressure) / acceptable_range
        pressure_dev = min(pressure_dev, 2.0)
    pressure_risk = pressure_dev * RISK_WEIGHTS['pressure']
    risk_breakdown['pressure_risk'] = round(pressure_risk, 1)
    risk_score += pressure_risk
    
    # Flow risk (weight: 12) - low flow is bad
    if flow >= IDEAL_VALUES['flow']['ideal']:
        flow_dev = 0
    else:
        acceptable_range = IDEAL_VALUES['flow']['ideal'] - IDEAL_VALUES['flow']['safe_min']
        flow_dev = (IDEAL_VALUES['flow']['ideal'] - flow) / acceptable_range
        flow_dev = min(flow_dev, 2.0)
    flow_risk = flow_dev * RISK_WEIGHTS['flow']
    risk_breakdown['flow_risk'] = round(flow_risk, 1)
    risk_score += flow_risk
    
    # Conductivity risk (weight: 8) - high conductivity indicates dissolved contaminants
    if conductivity <= IDEAL_VALUES['conductivity']['ideal']:
        conductivity_dev = 0
    else:
        acceptable_range = IDEAL_VALUES['conductivity']['safe_max'] - IDEAL_VALUES['conductivity']['ideal']
        conductivity_dev = (conductivity - IDEAL_VALUES['conductivity']['ideal']) / acceptable_range
        conductivity_dev = min(conductivity_dev, 2.0)
    conductivity_risk = conductivity_dev * RISK_WEIGHTS['conductivity']
    risk_breakdown['conductivity_risk'] = round(conductivity_risk, 1)
    risk_score += conductivity_risk
    
    # Temporal risk (weight: 10) - time-based patterns
    # Temporal modifier ranges from 1.0 (normal) to ~1.15 (high risk periods)
    temporal_base_risk = (temporal_modifier - 1.0) * 100  # Convert modifier to risk points
    temporal_risk = min(temporal_base_risk * RISK_WEIGHTS['temporal'] / 10, RISK_WEIGHTS['temporal'])
    risk_breakdown['temporal_risk'] = round(temporal_risk, 1)
    risk_breakdown['temporal_modifier'] = temporal_modifier
    risk_score += temporal_risk
    
    # ============================================================
    # INTERACTION RISKS (Dangerous Combinations)
    # ============================================================
    interaction_risk = 0
    interaction_reasons = []
    
    # Low chlorine + high turbidity → severe contamination risk (+18)
    if chlorine < 0.4 and turbidity > 2.0:
        interaction_risk += 18
        interaction_reasons.append("Low chlorine + high turbidity")
    
    # Low pressure + high turbidity → pipe intrusion/leak (+14)
    if pressure < 2.5 and turbidity > 2.0:
        interaction_risk += 14
        interaction_reasons.append("Low pressure + high turbidity")
    
    # Low flow + low chlorine → stagnation with depleted disinfectant (+10)
    if flow < 1.0 and chlorine < 0.5:
        interaction_risk += 10
        interaction_reasons.append("Low flow + low chlorine")
    
    # Additional dangerous combinations
    if pH < 6.5 and chlorine < 0.5:
        interaction_risk += 8
        interaction_reasons.append("Acidic pH + low chlorine")
    
    if pressure < 2.0 and flow < 1.0:
        interaction_risk += 10
        interaction_reasons.append("Very low pressure + low flow")
    
    # High conductivity + high turbidity → likely contamination
    if conductivity > 600 and turbidity > 1.5:
        interaction_risk += 12
        interaction_reasons.append("High conductivity + turbidity")
    
    risk_breakdown['interaction_risk'] = round(interaction_risk, 1)
    risk_breakdown['interaction_reasons'] = interaction_reasons
    risk_score += interaction_risk
    
    # Cap at 100
    risk_score = min(100, risk_score)
    
    return round(risk_score, 1), risk_breakdown


def get_pipe_status(data: dict) -> tuple:
    """
    Determine pipe status based on sensor readings.
    Returns: (status_name, color_hex)
    """
    pressure = data.get('pressure', 4.0)
    turbidity = data.get('turbidity', 0.5)
    chlorine = data.get('chlorine', 1.0)
    flow = data.get('flow', 2.0)
    
    if pressure < 2.5:
        return 'leak', '#F44336'
    elif turbidity > 2.0:
        return 'contamination', '#FF9800'
    elif chlorine < 0.4:
        return 'decay', '#9C27B0'
    elif flow < 1.0:
        return 'stagnation', '#FFC107'
    else:
        return 'normal', '#2196F3'


def detect_root_cause(data: dict, risk_score: float) -> str:
    """
    Advanced root cause detection using multiple parameter pattern analysis.
    """
    chlorine = data.get('chlorine', 1.0)
    turbidity = data.get('turbidity', 0.5)
    pH = data.get('pH', 7.0)
    pressure = data.get('pressure', 4.0)
    flow = data.get('flow', 2.5)
    temperature = data.get('temperature', 20)
    
    causes = {
        'Disinfectant Decay': 0,
        'Pipe Leak / Intrusion': 0,
        'Stagnation Risk': 0,
        'External Contamination': 0,
        'Normal Operation': 0
    }
    
    # Disinfectant Decay
    if chlorine < 0.5:
        causes['Disinfectant Decay'] += 35
    if chlorine < 0.3:
        causes['Disinfectant Decay'] += 25
    if temperature > 25 and chlorine < 0.6:
        causes['Disinfectant Decay'] += 20
    if turbidity < 1.0 and pressure > 3.0 and chlorine < 0.5:
        causes['Disinfectant Decay'] += 15
    
    # Pipe Leak / Intrusion
    if pressure < 2.5:
        causes['Pipe Leak / Intrusion'] += 40
    if pressure < 3.0 and turbidity > 1.5:
        causes['Pipe Leak / Intrusion'] += 30
    if pressure < 2.0:
        causes['Pipe Leak / Intrusion'] += 20
    
    # Stagnation Risk
    if flow < 1.0:
        causes['Stagnation Risk'] += 45
    if flow < 1.5 and chlorine < 0.6:
        causes['Stagnation Risk'] += 25
    if flow < 1.2 and turbidity > 0.8:
        causes['Stagnation Risk'] += 15
    
    # External Contamination
    abnormal_count = sum([
        chlorine < 0.3,
        turbidity > 2.5,
        pH < 6.3 or pH > 8.7,
        pressure < 2.0
    ])
    
    if abnormal_count >= 3:
        causes['External Contamination'] += 70
    elif abnormal_count == 2 and turbidity > 2.0:
        causes['External Contamination'] += 45
    elif turbidity > 3.0 and chlorine < 0.4:
        causes['External Contamination'] += 35
    
    # Normal Operation
    if risk_score < 15:
        causes['Normal Operation'] += 90
    elif risk_score < 25:
        causes['Normal Operation'] += 60
    elif risk_score < 35:
        causes['Normal Operation'] += 30
    
    most_probable = max(causes, key=causes.get)
    
    if max(causes.values()) < 15:
        return 'Normal Operation'
    
    return most_probable


def generate_ai_interpretation(data: dict, risk_score: float, root_cause: str, risk_breakdown: dict) -> str:
    """
    Generate a human-readable AI interpretation of current water quality status.
    Returns structured interpretation with | delimiter for formatting.
    Groups issues with their causes.
    """
    chlorine = data.get('chlorine', 1.0)
    turbidity = data.get('turbidity', 0.5)
    pressure = data.get('pressure', 4.0)
    flow = data.get('flow', 2.5)
    conductivity = data.get('conductivity', 400)
    
    # Main assessment
    if risk_score < 20:
        main_status = "✅ Water quality is within optimal parameters"
    elif risk_score < 40:
        main_status = "⚠️ Minor deviations detected from ideal conditions"
    elif risk_score < 60:
        main_status = "⚠️ Moderate risk detected requiring attention"
    else:
        main_status = "🚨 Critical risk level - immediate investigation recommended"
    
    # Build structured findings with causes inline
    findings = []
    
    # Chlorine issues with cause
    if chlorine < 0.4:
        cause = "→ Cause: Disinfectant system failure or high demand"
        findings.append(f"🧪 Chlorine critically low: {chlorine:.2f} mg/L {cause}")
    elif chlorine < 0.6:
        cause = "→ Cause: Gradual decay or distance from treatment"
        findings.append(f"🧪 Chlorine below optimal: {chlorine:.2f} mg/L {cause}")
    
    # Pressure issues with cause
    if pressure < 2.5:
        cause = "→ Cause: Pipe leak, main break, or valve issue"
        findings.append(f"📉 Low pressure: {pressure:.2f} bar {cause}")
    
    # Turbidity issues with cause
    if turbidity > 2.0:
        cause = "→ Cause: Contamination, sediment disturbance, or intrusion"
        findings.append(f"🌫️ High turbidity: {turbidity:.2f} NTU {cause}")
    elif turbidity > 1.0:
        cause = "→ Cause: Minor sediment or pipe scaling"
        findings.append(f"🌫️ Elevated turbidity: {turbidity:.2f} NTU {cause}")
    
    # Flow issues with cause
    if flow < 1.0:
        cause = "→ Cause: Dead-end pipe, low demand, or blockage"
        findings.append(f"🌊 Low flow: {flow:.2f} m³/h {cause}")
    
    # Conductivity issues with cause
    if conductivity > 700:
        cause = "→ Cause: Chemical contamination or mineral intrusion"
        findings.append(f"⚡ High conductivity: {conductivity:.0f} µS/cm {cause}")
    
    # Interaction warnings with consequences
    interaction_reasons = risk_breakdown.get('interaction_reasons', [])
    if interaction_reasons:
        # Map combinations to their consequences
        consequences = {
            'Low chlorine + high turbidity': '⚠️ Dangerous combination: Low chlorine + high turbidity → Consequence: Pathogens can multiply rapidly without disinfection',
            'Low pressure + high turbidity': '⚠️ Dangerous combination: Low pressure + high turbidity → Consequence: Contaminants may be drawn into pipes through cracks',
            'Low flow + low chlorine': '⚠️ Dangerous combination: Low flow + low chlorine → Consequence: Bacterial biofilm growth in stagnant water',
            'Acidic pH + low chlorine': '⚠️ Dangerous combination: Acidic pH + low chlorine → Consequence: Pipe corrosion releasing metals into water',
            'Very low pressure + low flow': '⚠️ Dangerous combination: Very low pressure + low flow → Consequence: System failure risk, backflow contamination possible',
            'High conductivity + turbidity': '⚠️ Dangerous combination: High conductivity + turbidity → Consequence: Chemical spill or industrial contamination likely'
        }
        for reason in interaction_reasons:
            if reason in consequences:
                findings.append(consequences[reason])
            else:
                findings.append(f"⚠️ Dangerous combination: {reason} → Consequence: Elevated health risk")
    
    # Overall suspected root cause (if different from specific causes)
    if root_cause != 'Normal Operation' and not findings:
        findings.append(f"🔍 System assessment: {root_cause}")
    
    # Build final output
    parts = [main_status]
    parts.extend(findings)
    
    return " | ".join(parts)


if __name__ == "__main__":
    print("Testing upgraded water simulation module...\n")
    
    for region in ['A', 'B', 'C']:
        data = simulate_sensor_data(region=region)
        risk, breakdown = calculate_risk_index(data)
        cause = detect_root_cause(data, risk)
        pipe_status, pipe_color = get_pipe_status(data)
        interpretation = generate_ai_interpretation(data, risk, cause, breakdown)
        
        print(f"Region {region} ({REGION_PROFILES[region]['name']}):")
        print(f"  Sensors: Cl={data['chlorine']}, Turb={data['turbidity']}, pH={data['pH']}, P={data['pressure']}, F={data['flow']}")
        print(f"  Risk Index: {risk}, Pipe: {pipe_status}")
        print(f"  Root Cause: {cause}")
        print(f"  AI: {interpretation}\n")
    
    print("=" * 60)
    print("Contamination Event Test:")
    data = simulate_sensor_data(region='A', contamination_event=True)
    risk, breakdown = calculate_risk_index(data)
    cause = detect_root_cause(data, risk)
    print(f"  Risk: {risk}, Interaction: {breakdown.get('interaction_risk', 0)}")
    print(f"  Cause: {cause}")
