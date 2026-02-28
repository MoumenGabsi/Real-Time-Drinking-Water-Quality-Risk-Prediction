# AquaGuard AI - Urban Water Quality Prediction System

## Overview

Engineering-grade AI-powered real-time water quality monitoring dashboard for urban drinking water distribution networks. Features advanced risk calculation with normalized deviation model, dangerous combination detection, **conductivity sensing**, **temporal risk patterns**, and AI-powered interpretation.

**Developed for WEDTECT Challenge 2026** - Predicting Urban Drinking Water Quality Degradation Using AI

## Key Features

### Multi-Sensor Data Integration

- **7 Physical Sensors**: pH, Turbidity, Conductivity, Chlorine, Temperature, Pressure, Flow
- **Temporal Features**: Hour of day, day of week, weekend detection
- **Physical Dependencies**: Realistic sensor correlations

### Advanced Risk Calculation

- **Normalized Deviation Model**: Risk calculated based on deviation from ideal values
- **Weighted Parameters**: Chlorine (25), Turbidity (20), pH (12), Pressure (13), Flow (12), Conductivity (8), Temporal (10)
- **Interaction Risks**: Bonus risk for dangerous parameter combinations

### Temporal Risk Factors

| Time Period  | Hours | Risk Modifier |
| ------------ | ----- | ------------- |
| Night        | 0-5   | 1.15x         |
| Morning Peak | 6-9   | 1.10x         |
| Day          | 10-16 | 1.00x         |
| Evening Peak | 17-20 | 1.08x         |
| Late Evening | 21-23 | 1.05x         |

### Interaction Risk Rules

| Condition                                   | Risk Bonus |
| ------------------------------------------- | ---------- |
| Low chlorine (<0.4) + High turbidity (>2.0) | +18        |
| Low pressure (<2.5) + High turbidity (>2.0) | +14        |
| High conductivity (>600) + Turbidity (>1.5) | +12        |
| Low flow (<1.0) + Low chlorine (<0.5)       | +10        |
| Very low pressure (<2.0) + Low flow (<1.0)  | +10        |
| Acidic pH (<6.5) + Low chlorine (<0.5)      | +8         |

### Physical Dependencies (Simulation)

- Pressure drop → Turbidity increase (sediment disturbance)
- High temperature → Chlorine decay acceleration
- Low flow → Chlorine decay (stagnation effect)
- Temperature → Conductivity coefficient adjustments

### Dashboard Features

- **Live Monitoring**: Real-time simulated sensor data
- **Network Control**: Manual adjustment of all 7 parameters
- **Pipe Visualization**: Color-coded network topology
- **AI Interpretation**: Human-readable analysis
- **Root Cause Detection**: Multi-parameter pattern analysis
- **Temporal Awareness**: Time-based risk adjustments

## Project Structure

```
water_ai_project/
├── simulation.py      # Sensor simulation + risk calculation + temporal features
├── model.py           # ML model training with 9 features
├── app.py             # Streamlit dashboard with modern UI
├── requirements.txt   # Dependencies
└── README.md          # Documentation
```

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Step 1: Train the AI Model

```bash
python model.py
```

Expected output:

```
MODEL PERFORMANCE
  R² Score:           ~0.99
  Mean Absolute Error: ~1.4
  Root Mean Sq Error:  ~2.1
```

### Step 2: Run the Dashboard

```bash
streamlit run app.py
```

Access at **http://localhost:8501**

## Technical Methodology

### Dataset Generation

The system generates synthetic training data simulating realistic urban water distribution:

- **5,000 samples** across 3 distinct region profiles (Industrial, Residential, Commercial)
- **~16% anomaly rate** with contamination events, pipe leaks, stagnation
- **Temporal diversity**: Data generated across all hours and days of week
- **Physical correlations**: Sensor readings follow realistic dependencies

### Risk Index Formula

For each parameter:

```
deviation = abs(value - ideal) / acceptable_range
risk_component = deviation * weight
```

Plus interaction risks for dangerous combinations.
Plus temporal risk based on time of day patterns.
Total capped at 100.

### Sensor Parameters & Ranges

| Parameter    | Ideal  | Safe Range | Weight | Unit  |
| ------------ | ------ | ---------- | ------ | ----- |
| Chlorine     | 1.0    | 0.5 - 2.0  | 25     | mg/L  |
| Turbidity    | 0.3    | < 1.0      | 20     | NTU   |
| pH           | 7.2    | 6.5 - 8.5  | 12     | -     |
| Pressure     | 4.0    | > 3.0      | 13     | bar   |
| Flow         | 2.0    | > 1.0      | 12     | m³/h  |
| Conductivity | 400    | 200 - 600  | 8      | µS/cm |
| Temporal     | varies | time-based | 10     | -     |

### ML Model Features

```python
MODEL_FEATURES = [
    'temperature',   # Environmental factor
    'flow',          # Flow rate sensor
    'pressure',      # Pressure sensor
    'chlorine',      # Disinfectant level
    'pH',            # Acidity measure
    'turbidity',     # Particle content
    'conductivity',  # Dissolved solids
    'hour',          # Hour of day (0-23)
    'is_weekend'     # Weekend flag (0/1)
]
```

### Model Performance

- **Algorithm**: Random Forest Regressor (100 trees, max_depth=15)
- **R² Score**: 0.993
- **MAE**: 1.38
- **RMSE**: 2.07

### Feature Importance (from training)

| Feature      | Importance |
| ------------ | ---------- |
| Turbidity    | 73.4%      |
| Chlorine     | 16.8%      |
| Pressure     | 4.7%       |
| Hour         | 2.6%       |
| pH           | 1.4%       |
| Flow         | 0.6%       |
| Is_weekend   | 0.3%       |
| Conductivity | 0.2%       |
| Temperature  | 0.1%       |

### Pipe Status Detection

| Status             | Condition           | Color  |
| ------------------ | ------------------- | ------ |
| Leak               | Pressure < 2.5 bar  | Red    |
| Contamination      | Turbidity > 2.0 NTU | Orange |
| Disinfectant Decay | Chlorine < 0.4 mg/L | Purple |
| Stagnation         | Flow < 1.0 m³/h     | Yellow |
| Normal             | All parameters OK   | Blue   |

### Root Cause Categories

- Disinfectant Decay
- Pipe Leak / Intrusion
- Stagnation Risk
- External Contamination
- Normal Operation

## Risk Level Thresholds

| Score  | Level    | Action           |
| ------ | -------- | ---------------- |
| 0-30   | SAFE     | Normal operation |
| 30-60  | WARNING  | Monitor closely  |
| 60-100 | CRITICAL | Immediate action |

## Dashboard Views

### Live Monitoring

- Real-time sensor readings (7 parameters)
- AI risk prediction
- AI interpretation text
- Root cause analysis
- Dynamic alerts
- Temporal risk indicator

### Network Control

- Manual slider controls for all 7 sensor parameters
- Pipe network visualization with status
- All regions risk summary
- Detailed analysis per region
- Conductivity monitoring (µS/cm)

## API Reference

### simulation.py

```python
# Generate sensor data with temporal features
data = simulate_sensor_data(region='A', contamination_event=False)

# Calculate risk (normalized deviation + interactions + temporal)
risk_score, breakdown = calculate_risk_index(data)

# Get temporal risk features
temporal = get_temporal_features(datetime.now())
# Returns: {'hour', 'day_of_week', 'is_weekend', 'time_period', 'risk_modifier'}

# Get pipe status
status, color = get_pipe_status(data)

# Detect root cause
cause = detect_root_cause(data, risk_score)

# Generate AI interpretation
text = generate_ai_interpretation(data, risk_score, cause, breakdown)
```

### model.py

```python
from model import predict_risk, load_model, MODEL_FEATURES

model = load_model()
risk = predict_risk(sensor_data, model)
# sensor_data must include all MODEL_FEATURES
```

## Challenge Requirements Compliance

| Requirement                 | Implementation                           |
| --------------------------- | ---------------------------------------- |
| Multi-sensor data           | 7 sensors + temporal features            |
| Temporal/environmental info | Hour, day of week, weekend, time period  |
| Real-time risk index        | 0-100 scale with weighted parameters     |
| Cause identification        | Root cause detection + AI interpretation |
| Early warnings              | Visual alerts, status pills, risk labels |
| Dashboard                   | Modern Streamlit UI with network viz     |

## Troubleshooting

### Model Not Found

```bash
python model.py  # Train the model first
```

### Import Errors

Ensure you're in the `water_ai_project` directory.

### Port In Use

```bash
streamlit run app.py --server.port 8502
```

## License

MIT License - Built for Urban Water Quality Prediction Hackathon 2026

---

**Built with ❤️ for smarter water infrastructure | WEDTECT Challenge 2026**
