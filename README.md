# 💧 AquaGuard AI

### Real-Time Drinking Water Quality Risk Prediction System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-1.0+-red.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-ML-orange.svg" alt="sklearn">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</p>

<p align="center">
  <a href="https://aquaguardai.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-AquaGuard_AI-FF4B4B?style=for-the-badge" alt="Live Demo">
  </a>
</p>

<p align="center">
  <b>👉 <a href="https://aquaguardai.streamlit.app/">Try the Live Demo</a> 👈</b>
</p>

---

## 🎯 Project Overview

AquaGuard AI is an intelligent water quality monitoring system that uses **machine learning** and **predictive analytics** to detect water quality degradation **before it becomes critical**. The system combines real-time multi-sensor data with temporal patterns to provide early warnings and actionable insights.

**🏆 Developed for WEDTECT Challenge 2026**

### Key Features

| Feature                      | Description                                                         |
| ---------------------------- | ------------------------------------------------------------------- |
| 🤖 **AI Risk Prediction**    | Random Forest model predicting water quality risk index (R² = 0.99) |
| 🔮 **Predictive Analytics**  | Time-series trend analysis to forecast future sensor values         |
| ⚠️ **Early Warning System**  | Alerts hours before thresholds are breached                         |
| 🌐 **Interactive Dashboard** | Modern glassmorphism UI with animated water flow visualization      |
| 🎛️ **Manual Controls**       | Adjust sensor values to simulate scenarios                          |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AquaGuard AI Dashboard                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│   │  Region A   │    │  Region B   │    │  Region C   │    │
│   │ Industrial  │    │ Residential │    │ Commercial  │    │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    │
│          │                  │                  │            │
│          ▼                  ▼                  ▼            │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              Multi-Sensor Data Fusion                │  │
│   │  pH | Turbidity | Chlorine | Pressure | Flow | Temp  │  │
│   └─────────────────────────┬───────────────────────────┘  │
│                             │                              │
│          ┌──────────────────┼──────────────────┐          │
│          ▼                  ▼                  ▼          │
│   ┌────────────┐    ┌─────────────┐    ┌────────────┐    │
│   │ Risk Index │    │  Trend      │    │  Root      │    │
│   │ Prediction │    │  Analysis   │    │  Cause     │    │
│   │ (ML Model) │    │ (Time-Series)│    │ Detection  │    │
│   └────────────┘    └─────────────┘    └────────────┘    │
│                             │                              │
│                             ▼                              │
│   ┌─────────────────────────────────────────────────────┐  │
│   │              Early Warning Generation                │  │
│   │     "Chlorine will reach danger in ~3.5 hours"      │  │
│   └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Sensor Parameters

| Sensor           | Range      | Unit  | Purpose                    |
| ---------------- | ---------- | ----- | -------------------------- |
| **Chlorine**     | 0.1 - 2.5  | mg/L  | Disinfection effectiveness |
| **pH**           | 6.0 - 9.0  | -     | Chemical balance           |
| **Turbidity**    | 0.1 - 8.0  | NTU   | Contamination indicator    |
| **Pressure**     | 2.0 - 6.0  | bar   | Leak/intrusion detection   |
| **Flow**         | 0.5 - 5.0  | m³/h  | Stagnation detection       |
| **Conductivity** | 150 - 1000 | µS/cm | Dissolved solids           |
| **Temperature**  | 10 - 30    | °C    | Decay rate factor          |

---

## 🔮 How Prediction Works

### 1️⃣ Data Collection

```python
# Sensor readings stored with timestamps
history.add_reading(region, sensor_data, timestamp)
```

### 2️⃣ Trend Detection

```python
# Linear regression on recent values
rate_per_hour = calculate_trend(values)
# Example: chlorine dropping at -0.1 mg/L per hour
```

### 3️⃣ Future Prediction

```
Predicted Value = Current + (Rate × Hours)

Example:
  Current chlorine: 0.8 mg/L
  Rate: -0.1 mg/L/hour
  In 6 hours: 0.8 + (-0.1 × 6) = 0.2 mg/L (DANGER!)
```

### 4️⃣ Early Warning

```
⚠️ Time to Threshold = (Threshold - Current) / Rate

🚨 CRITICAL: < 6 hours to danger zone
⚠️ WARNING:  < 12 hours to warning zone
```

---

## 🚀 Quick Start

### Prerequisites

```bash
pip install streamlit pandas numpy scikit-learn joblib
```

### Run the Application

```bash
# Train the model (first time only)
python model.py

# Launch the dashboard
streamlit run app.py
```

### Access

Open your browser to: **http://localhost:8501**

---

## 📁 Project Structure

```
water_ai_project/
├── app.py              # Streamlit dashboard (UI)
├── simulation.py       # Sensor simulation & analytics
├── model.py            # ML model training
├── water_risk_model.pkl # Trained model file
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## 🧠 Machine Learning Model

| Metric               | Value                   |
| -------------------- | ----------------------- |
| **Algorithm**        | Random Forest Regressor |
| **Trees**            | 100                     |
| **Max Depth**        | 15                      |
| **R² Score**         | 0.9925                  |
| **MAE**              | 1.68                    |
| **Training Samples** | 10,000                  |

### Features Used

- Temperature, Flow, Pressure
- Chlorine, pH, Turbidity
- Conductivity
- Hour of day, Weekend flag

### Training Scenarios

1. Normal operation
2. Contamination events
3. Pipe leaks
4. Water stagnation
5. Chemical spills
6. Disinfectant failure
7. Pressure surges
8. Gradual decay
9. Night stagnation
10. Peak demand stress

---

## 🎨 Dashboard Features

### Network Visualization

- Animated water flow through pipes
- Color-coded risk status (🟢 Safe / 🟡 Warning / 🔴 Critical)
- Interactive region cards with real-time metrics

### Predictive Analytics Panel

- Trend indicators (↑ ↓ →) for each sensor
- Rate of change per hour
- Early warnings with time-to-threshold

### AI Analysis

- Automatic root cause detection
- Dangerous combination warnings
- Consequence explanations

---

## 📈 Risk Calculation

```
Risk Index = Σ (Deviation × Weight) + Interaction Bonus + Temporal Factor

Where:
- Chlorine:     25% weight (most critical)
- Turbidity:    20% weight
- Pressure:     13% weight
- pH:           12% weight
- Flow:         12% weight
- Conductivity:  8% weight
- Temporal:     10% weight
```

### Interaction Penalties

| Combination                   | Penalty | Consequence            |
| ----------------------------- | ------- | ---------------------- |
| Low chlorine + High turbidity | +15     | Pathogen growth risk   |
| Low pressure + High turbidity | +12     | Backflow contamination |
| Low flow + Low chlorine       | +10     | Biofilm formation      |
| Acidic pH + Low chlorine      | +8      | Pipe corrosion         |

---

## 🏆 Challenge Compliance

This project was built for the **WEDTECT Challenge** and meets all requirements:

✅ Multi-sensor data integration (pH, turbidity, conductivity, chlorine, temperature, pressure, flow)  
✅ Temporal pattern analysis  
✅ Real-time risk index estimation  
✅ Contamination anticipation before critical  
✅ Root cause identification  
✅ Interactive dashboard  
✅ Early warning system

---

## 👨‍💻 Author

**Moumen Gabsi**

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  <b>🌊 Protecting water quality, one prediction at a time 🌊</b>
</p>
