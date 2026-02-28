"""
AI Smart Water Quality Monitoring Dashboard (Upgraded)
Engineering-grade real-time water quality risk prediction system.
Features: Normalized deviation risk, interaction risks, pipe visualization, AI interpretation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import (
    simulate_sensor_data,
    calculate_risk_index,
    detect_root_cause,
    get_pipe_status,
    generate_ai_interpretation,
    SENSOR_RANGES,
    REGION_PROFILES,
    IDEAL_VALUES
)
from model import predict_risk, load_model, MODEL_PATH

# Page configuration
st.set_page_config(
    page_title="AI Water Quality Monitor",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-box {
        text-align: center;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.5rem 0;
    }
    .ai-interpretation {
        background: linear-gradient(135deg, #E8EAF6 0%, #C5CAE9 100%);
        border-left: 5px solid #3F51B5;
        padding: 15px 20px;
        border-radius: 0 10px 10px 0;
        margin: 15px 0;
    }
    
    /* ===== CONNECTED PIPE NETWORK SYSTEM ===== */
    .pipe-network-container {
        background: linear-gradient(180deg, #eceff1 0%, #cfd8dc 100%);
        border-radius: 20px;
        padding: 25px;
        margin: 15px 0;
        border: 2px solid #b0bec5;
    }
    
    .water-source-box {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        color: white;
        padding: 12px 25px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 1rem;
        box-shadow: 0 4px 15px rgba(13, 71, 161, 0.4);
        display: inline-block;
    }
    
    .main-pipe-horizontal {
        background: linear-gradient(180deg, #42A5F5 0%, #1E88E5 50%, #42A5F5 100%);
        height: 22px;
        border-radius: 11px;
        position: relative;
        box-shadow: inset 0 3px 6px rgba(0,0,0,0.2), 0 3px 10px rgba(66, 165, 245, 0.3);
    }
    
    .main-pipe-horizontal::before {
        content: '';
        position: absolute;
        top: 4px;
        left: 5%;
        right: 5%;
        height: 5px;
        background: rgba(255,255,255,0.35);
        border-radius: 3px;
    }
    
    .vertical-pipe-segment {
        width: 18px;
        background: linear-gradient(90deg, #42A5F5 0%, #64B5F6 50%, #42A5F5 100%);
        border-radius: 9px;
        margin: 0 auto;
        position: relative;
        box-shadow: inset 3px 0 6px rgba(0,0,0,0.15);
    }
    
    .vertical-pipe-segment::before {
        content: '';
        position: absolute;
        left: 4px;
        top: 0;
        bottom: 0;
        width: 4px;
        background: rgba(255,255,255,0.3);
        border-radius: 2px;
    }
    
    .splitter-hub {
        background: linear-gradient(135deg, #455A64 0%, #263238 100%);
        color: white;
        width: 55px;
        height: 55px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin: 0 auto;
        box-shadow: 0 4px 15px rgba(38, 50, 56, 0.5);
        font-size: 1.2rem;
    }
    
    .connector-node {
        width: 14px;
        height: 14px;
        background: linear-gradient(135deg, #0D47A1 0%, #1565C0 100%);
        border-radius: 50%;
        margin: 0 auto;
        box-shadow: 0 0 10px rgba(13, 71, 161, 0.5);
    }
    
    .region-endpoint {
        padding: 15px;
        border-radius: 15px;
        text-align: center;
        transition: all 0.3s ease;
        border: 3px solid;
    }
    
    .region-endpoint:hover {
        transform: scale(1.02);
    }
    
    .status-badge {
        padding: 8px 16px;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: bold;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: white;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    .status-NORMAL { background: linear-gradient(135deg, #4CAF50 0%, #388E3C 100%); }
    .status-LEAK { background: linear-gradient(135deg, #F44336 0%, #C62828 100%); }
    .status-CONTAMINATION { background: linear-gradient(135deg, #FF9800 0%, #EF6C00 100%); }
    .status-DECAY { background: linear-gradient(135deg, #9C27B0 0%, #6A1B9A 100%); }
    .status-STAGNATION { background: linear-gradient(135deg, #FFC107 0%, #F57C00 100%); color: #333; }
    
    .pipe-legend-bar {
        display: flex;
        justify-content: center;
        gap: 15px;
        flex-wrap: wrap;
        background: rgba(255,255,255,0.7);
        padding: 12px 20px;
        border-radius: 30px;
        margin-top: 20px;
    }
    
    .legend-chip {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 5px 12px;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 500;
        background: white;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
    }
    
    .legend-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    .slider-panel {
        background: white;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
        margin-bottom: 10px;
    }
    
    .slider-panel-header {
        font-weight: bold;
        font-size: 0.95rem;
        padding-bottom: 10px;
        margin-bottom: 10px;
        border-bottom: 2px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


def get_risk_color(risk: float) -> str:
    """Return color based on risk level."""
    if risk < 30:
        return "#4CAF50"  # Green
    elif risk < 60:
        return "#FF9800"  # Orange
    else:
        return "#F44336"  # Red


def get_risk_label(risk: float) -> str:
    """Return risk level label."""
    if risk < 30:
        return "SAFE"
    elif risk < 60:
        return "WARNING"
    else:
        return "CRITICAL"


def display_sensor_metric(col, name: str, value: float, unit: str, ideal: float = None):
    """Display a sensor metric with ideal comparison."""
    if name in IDEAL_VALUES:
        ideal_val = IDEAL_VALUES[name]['ideal']
        deviation = abs(value - ideal_val)
        if deviation < 0.2:
            delta_color = "normal"
        elif deviation < 0.5:
            delta_color = "off"
        else:
            delta_color = "inverse"
        delta_text = f"Ideal: {ideal_val}"
    else:
        delta_color = "off"
        delta_text = f"Range: {SENSOR_RANGES[name]['min']}-{SENSOR_RANGES[name]['max']}"
    
    col.metric(
        label=f"🔹 {name.capitalize()}",
        value=f"{value:.2f} {unit}",
        delta=delta_text,
        delta_color=delta_color
    )


def render_pipe_visualization(regions_data: dict, model=None):
    """Render the connected pipe network visualization with real-time status."""
    
    # Calculate risks for all regions
    region_status = {}
    for region, data in regions_data.items():
        status, color = get_pipe_status(data)
        risk, _ = calculate_risk_index(data)
        if model:
            try:
                ml_risk = predict_risk(data, model)
            except:
                ml_risk = risk
        else:
            ml_risk = risk
        
        region_status[region] = {
            'status': status,
            'color': color,
            'risk': ml_risk,
            'name': REGION_PROFILES[region]['name']
        }
    
    st.markdown("### 🏭 Water Distribution Network")
    
    # === WATER PROVIDER SOURCE ===
    st.markdown("""
    <div style="text-align: center; margin-bottom: 5px;">
        <div class="water-source-box">
            💧 WATER TREATMENT PLANT
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === VERTICAL PIPE FROM SOURCE ===
    st.markdown("""
    <div style="display: flex; justify-content: center;">
        <div class="vertical-pipe-segment" style="height: 40px;"></div>
    </div>
    <div style="text-align: center;"><div class="connector-node"></div></div>
    """, unsafe_allow_html=True)
    
    # === MAIN HORIZONTAL PIPELINE ===
    st.markdown("""
    <div style="margin: 8px 10%;">
        <div class="main-pipe-horizontal">
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); 
                        color: white; font-weight: bold; font-size: 0.75rem; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                MAIN PIPELINE
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # === SPLITTER NODE ===
    st.markdown("""
    <div style="text-align: center;"><div class="connector-node"></div></div>
    <div style="display: flex; justify-content: center; margin: 5px 0;">
        <div class="vertical-pipe-segment" style="height: 25px;"></div>
    </div>
    <div style="text-align: center; margin-bottom: 8px;">
        <div class="splitter-hub">⚡</div>
        <div style="font-size: 0.8rem; color: #455A64; font-weight: bold; margin-top: 3px;">SPLITTER</div>
    </div>
    """, unsafe_allow_html=True)
    
    # === HORIZONTAL DISTRIBUTION BAR ===
    st.markdown("""
    <div style="margin: 5px 5%;">
        <div class="main-pipe-horizontal" style="height: 16px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # === THREE VERTICAL PIPES TO REGIONS ===
    pipe_cols = st.columns(3)
    
    for idx, region in enumerate(['A', 'B', 'C']):
        info = region_status[region]
        status_upper = info['status'].upper()
        
        # Map status to display text
        status_display = {
            'NORMAL': '✓ NORMAL',
            'LEAK': '⚠ LEAK',
            'CONTAMINATION': '☣ CONTAMINATION', 
            'DECAY': '🧪 DECAY',
            'STAGNATION': '🌊 STAGNATION'
        }
        
        icon_map = {'A': '🏭', 'B': '🏠', 'C': '🏪'}
        
        with pipe_cols[idx]:
            # Connection point
            st.markdown('<div style="text-align: center;"><div class="connector-node"></div></div>', unsafe_allow_html=True)
            
            # Vertical pipe colored by status
            st.markdown(f'''
            <div style="display: flex; justify-content: center;">
                <div style="width: 16px; height: 60px; background: linear-gradient(90deg, {info['color']}CC 0%, {info['color']} 50%, {info['color']}CC 100%); 
                            border-radius: 8px; box-shadow: 0 2px 8px {info['color']}60;"></div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Region endpoint box
            st.markdown(f'''
            <div class="region-endpoint" style="background: {info['color']}15; border-color: {info['color']};">
                <div style="font-size: 2rem; margin-bottom: 5px;">{icon_map[region]}</div>
                <div style="font-weight: bold; font-size: 1.1rem; color: #333;">Region {region}</div>
                <div style="font-size: 0.85rem; color: #666; margin-bottom: 8px;">{info['name']}</div>
                <div class="status-badge status-{status_upper}">{status_display.get(status_upper, status_upper)}</div>
                <div style="margin-top: 10px; font-size: 1.5rem; font-weight: bold; color: {info['color']};">
                    {info['risk']:.1f}
                </div>
                <div style="font-size: 0.75rem; color: #888;">Risk Index</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # === LEGEND ===
    st.markdown("""
    <div class="pipe-legend-bar">
        <div class="legend-chip"><div class="legend-dot" style="background: #4CAF50;"></div>Normal</div>
        <div class="legend-chip"><div class="legend-dot" style="background: #F44336;"></div>Leak</div>
        <div class="legend-chip"><div class="legend-dot" style="background: #FF9800;"></div>Contamination</div>
        <div class="legend-chip"><div class="legend-dot" style="background: #9C27B0;"></div>Decay</div>
        <div class="legend-chip"><div class="legend-dot" style="background: #FFC107;"></div>Stagnation</div>
    </div>
    """, unsafe_allow_html=True)


def render_risk_display(risk_score: float, model_available: bool):
    """Render the main risk index display."""
    risk_color = get_risk_color(risk_score)
    risk_label = get_risk_label(risk_score)
    
    # Progress bar
    st.progress(int(min(risk_score, 100)) / 100)
    
    # Big risk display
    st.markdown(f"""
    <div class="risk-box" style="background-color: {risk_color}15; border: 3px solid {risk_color};">
        <h1 style="color: {risk_color}; margin: 0; font-size: 3.5rem;">{risk_score:.1f}</h1>
        <p style="color: {risk_color}; font-weight: bold; font-size: 1.3rem; margin: 5px 0;">{risk_label}</p>
        <p style="color: #666; font-size: 0.9rem; margin: 0;">
            {'🤖 AI Model Prediction' if model_available else '📐 Formula Calculation'}
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_ai_interpretation(interpretation: str):
    """Render the AI interpretation section."""
    st.markdown("### 🧠 AI Interpretation")
    st.markdown(f"""
    <div class="ai-interpretation">
        <p style="margin: 0; color: #333; line-height: 1.6;">{interpretation}</p>
    </div>
    """, unsafe_allow_html=True)


def render_risk_breakdown(risk_breakdown: dict):
    """Render the risk breakdown section."""
    st.markdown("### 📊 Risk Breakdown")
    
    # Main risk components
    components = ['chlorine_risk', 'turbidity_risk', 'pH_risk', 'pressure_risk', 'flow_risk']
    weights = {'chlorine_risk': 30, 'turbidity_risk': 25, 'pH_risk': 15, 'pressure_risk': 15, 'flow_risk': 15}
    
    for comp in components:
        value = risk_breakdown.get(comp, 0)
        max_weight = weights.get(comp, 30)
        factor_name = comp.replace('_risk', '').replace('_', ' ').upper()
        pct = min(value / max_weight, 1.0) if max_weight > 0 else 0
        
        st.markdown(f"**{factor_name}**: {value:.1f} / {max_weight}")
        st.progress(pct)
    
    # Interaction risk
    interaction = risk_breakdown.get('interaction_risk', 0)
    if interaction > 0:
        st.markdown(f"**⚠️ INTERACTION RISK**: +{interaction:.1f}")
        reasons = risk_breakdown.get('interaction_reasons', [])
        if reasons:
            st.caption(f"Triggers: {', '.join(reasons)}")


def render_root_cause(root_cause: str):
    """Render root cause analysis."""
    cause_config = {
        'Disinfectant Decay': {'icon': '🧪', 'color': '#9C27B0'},
        'Pipe Leak / Intrusion': {'icon': '🔧', 'color': '#F44336'},
        'Stagnation Risk': {'icon': '🌊', 'color': '#2196F3'},
        'External Contamination': {'icon': '☣️', 'color': '#FF5722'},
        'Normal Operation': {'icon': '✅', 'color': '#4CAF50'}
    }
    
    config = cause_config.get(root_cause, {'icon': '❓', 'color': '#666'})
    
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; background-color: {config['color']}15; 
                border-radius: 15px; border: 2px solid {config['color']};">
        <div style="font-size: 3rem;">{config['icon']}</div>
        <h3 style="color: {config['color']}; margin: 10px 0 0 0;">{root_cause}</h3>
    </div>
    """, unsafe_allow_html=True)


def render_alert(risk_score: float, root_cause: str, region: str):
    """Render dynamic alert based on risk level."""
    if risk_score >= 60:
        st.error(f"""
        ## 🚨 CRITICAL ALERT - Region {region}
        
        **Risk Level:** {risk_score:.1f}/100 (CRITICAL)
        
        **Detected Issue:** {root_cause}
        
        **Immediate Actions Required:**
        - Isolate affected water supply section
        - Dispatch maintenance team immediately
        - Notify regional water authority
        - Consider issuing public advisory
        """)
    elif risk_score >= 30:
        st.warning(f"""
        ### ⚡ Warning - Region {region}
        
        **Risk Level:** {risk_score:.1f}/100 (ELEVATED)
        
        **Potential Issue:** {root_cause}
        
        Monitor closely and prepare for intervention if conditions worsen.
        """)
    else:
        st.success(f"""
        ### ✅ All Clear - Region {region}
        
        **Risk Level:** {risk_score:.1f}/100 (SAFE)
        
        Water quality parameters within acceptable ranges. System operating normally.
        """)


def main():
    # Header
    st.markdown('<p class="main-header">💧 AI Smart Water Quality Monitoring System</p>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Engineering-grade real-time water quality risk prediction with AI-powered analysis</p>', 
                unsafe_allow_html=True)
    
    st.divider()
    
    # Initialize session state
    if 'manual_values' not in st.session_state:
        st.session_state.manual_values = {
            'A': {'temperature': 20.0, 'flow': 2.5, 'pressure': 4.0, 'chlorine': 1.2, 'pH': 7.2, 'turbidity': 0.5},
            'B': {'temperature': 22.0, 'flow': 2.1, 'pressure': 3.6, 'chlorine': 1.0, 'pH': 7.3, 'turbidity': 0.6},
            'C': {'temperature': 19.0, 'flow': 2.7, 'pressure': 4.4, 'chlorine': 1.3, 'pH': 7.1, 'turbidity': 0.4}
        }
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Control Panel")
        
        view_mode = st.radio(
            "📺 View Mode",
            options=['Live Monitoring', 'Network Control'],
            help="Switch between live simulation and manual control"
        )
        
        st.divider()
        
        if view_mode == 'Live Monitoring':
            region = st.selectbox("📍 Select Region", options=['A', 'B', 'C'])
            
            st.divider()
            st.subheader("🧪 Simulation")
            
            if st.button("⚠️ Simulate Contamination", help="Force abnormal readings"):
                st.session_state['contamination'] = True
            
            if st.button("🔄 Reset to Normal"):
                st.session_state['contamination'] = False
            
            if st.session_state.get('contamination', False):
                st.warning("⚠️ Contamination active!")
            
            st.divider()
            if st.button("🔄 Refresh Data", type="primary"):
                st.rerun()
        else:
            st.info("🔧 Adjust sensor values manually in the Network Control view.")
        
        st.divider()
        st.subheader("🤖 Model Status")
        if os.path.exists(MODEL_PATH):
            st.success("✅ AI Model Ready")
        else:
            st.error("❌ Model not trained")
            st.info("Run: `python model.py`")
    
    # Main content
    if view_mode == 'Live Monitoring':
        render_live_view(region, st.session_state.get('contamination', False))
    else:
        render_network_control_view()


def render_live_view(region: str, is_contamination: bool):
    """Render live monitoring view."""
    # Generate data
    sensor_data = simulate_sensor_data(region=region, contamination_event=is_contamination)
    formula_risk, risk_breakdown = calculate_risk_index(sensor_data)
    
    # ML prediction
    try:
        model = load_model()
        ml_risk = predict_risk(sensor_data, model)
        model_available = True
    except FileNotFoundError:
        ml_risk = formula_risk
        model_available = False
    
    root_cause = detect_root_cause(sensor_data, ml_risk)
    interpretation = generate_ai_interpretation(sensor_data, ml_risk, root_cause, risk_breakdown)
    
    # Layout
    col_sensors, col_risk = st.columns([2, 1])
    
    with col_sensors:
        st.subheader(f"📊 Sensor Readings - Region {region} ({REGION_PROFILES[region]['name']})")
        
        row1 = st.columns(3)
        row2 = st.columns(3)
        
        sensors_r1 = ['temperature', 'chlorine', 'pH']
        sensors_r2 = ['turbidity', 'pressure', 'flow']
        
        for i, s in enumerate(sensors_r1):
            display_sensor_metric(row1[i], s, sensor_data[s], SENSOR_RANGES[s]['unit'])
        for i, s in enumerate(sensors_r2):
            display_sensor_metric(row2[i], s, sensor_data[s], SENSOR_RANGES[s]['unit'])
    
    with col_risk:
        st.subheader("🎯 Risk Index")
        render_risk_display(ml_risk, model_available)
    
    st.divider()
    
    # AI Interpretation
    render_ai_interpretation(interpretation)
    
    st.divider()
    
    # Risk Assessment
    st.subheader("🔬 Detailed Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Root Cause")
        render_root_cause(root_cause)
    
    with col2:
        render_risk_breakdown(risk_breakdown)
    
    st.divider()
    
    # Alert
    render_alert(ml_risk, root_cause, region)
    
    # Raw data
    with st.expander("📋 Raw Sensor Data"):
        display_data = {
            'Parameter': ['Region', 'Temperature', 'Flow', 'Pressure', 'Chlorine', 'pH', 'Turbidity'],
            'Value': [region, f"{sensor_data['temperature']:.2f} °C", f"{sensor_data['flow']:.2f} m³/h",
                     f"{sensor_data['pressure']:.2f} bar", f"{sensor_data['chlorine']:.2f} mg/L",
                     f"{sensor_data['pH']:.2f}", f"{sensor_data['turbidity']:.2f} NTU"],
            'Ideal': ['—', '—', '2.0 m³/h', '4.0 bar', '1.0 mg/L', '7.2', '< 0.3 NTU']
        }
        st.dataframe(pd.DataFrame(display_data), hide_index=True)


def render_network_control_view():
    """Render integrated network control view - pipe visualization + sliders on same page."""
    
    # Load model
    try:
        model = load_model()
        model_available = True
    except FileNotFoundError:
        model = None
        model_available = False
    
    # === PIPE NETWORK VISUALIZATION (TOP) ===
    # Collect current values first to show real-time status
    regions_data = {}
    for region in ['A', 'B', 'C']:
        vals = st.session_state.manual_values[region]
        regions_data[region] = {
            'temperature': vals['temperature'], 
            'flow': vals['flow'], 
            'pressure': vals['pressure'],
            'chlorine': vals['chlorine'], 
            'pH': vals['pH'], 
            'turbidity': vals['turbidity']
        }
    
    # Render pipe network with current data
    render_pipe_visualization(regions_data, model)
    
    st.divider()
    
    # === SENSOR CONTROLS (BELOW NETWORK) ===
    st.markdown("### 🎛️ Sensor Control Panel")
    st.caption("Adjust values below to see real-time changes in the pipe network above")
    
    # Three column layout for region controls
    ctrl_cols = st.columns(3)
    
    for idx, region in enumerate(['A', 'B', 'C']):
        with ctrl_cols[idx]:
            vals = st.session_state.manual_values[region]
            status, color = get_pipe_status(regions_data[region])
            
            # Header with status indicator
            st.markdown(f'''
            <div class="slider-panel">
                <div class="slider-panel-header" style="color: {color};">
                    {'🏭' if region == 'A' else '🏠' if region == 'B' else '🏪'} Region {region} - {REGION_PROFILES[region]['name']}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Compact sliders
            temp = st.slider(f"🌡️ Temp (°C)", 10.0, 30.0, float(vals['temperature']), 0.5, key=f"t_{region}")
            chlorine = st.slider(f"🧪 Chlorine", 0.1, 2.0, float(vals['chlorine']), 0.05, key=f"c_{region}")
            ph = st.slider(f"⚗️ pH", 6.5, 8.5, float(vals['pH']), 0.1, key=f"p_{region}")
            turbidity = st.slider(f"💨 Turbidity", 0.1, 5.0, float(vals['turbidity']), 0.1, key=f"tb_{region}")
            pressure = st.slider(f"📊 Pressure", 2.0, 6.0, float(vals['pressure']), 0.1, key=f"pr_{region}")
            flow = st.slider(f"🌊 Flow", 0.5, 5.0, float(vals['flow']), 0.1, key=f"f_{region}")
            
            # Update session state
            st.session_state.manual_values[region] = {
                'temperature': temp, 'chlorine': chlorine, 'pH': ph,
                'turbidity': turbidity, 'pressure': pressure, 'flow': flow
            }
            
            # Update regions_data with new values for analysis
            regions_data[region] = {
                'temperature': temp, 'flow': flow, 'pressure': pressure,
                'chlorine': chlorine, 'pH': ph, 'turbidity': turbidity
            }
    
    st.divider()
    
    # === QUICK STATUS SUMMARY ===
    st.markdown("### ⚡ Quick Status Overview")
    
    status_cols = st.columns(3)
    for idx, region in enumerate(['A', 'B', 'C']):
        with status_cols[idx]:
            data = regions_data[region]
            risk, breakdown = calculate_risk_index(data)
            
            if model_available:
                ml_risk = predict_risk(data, model)
            else:
                ml_risk = risk
            
            root_cause = detect_root_cause(data, ml_risk)
            risk_color = get_risk_color(ml_risk)
            risk_label = get_risk_label(ml_risk)
            
            # Compact status card
            st.markdown(f"""
            <div style="text-align: center; padding: 15px; background: {risk_color}10; 
                        border-radius: 12px; border-left: 5px solid {risk_color};">
                <div style="font-weight: bold; color: #333;">Region {region}</div>
                <div style="font-size: 1.8rem; font-weight: bold; color: {risk_color}; margin: 5px 0;">{ml_risk:.1f}</div>
                <div style="font-size: 0.85rem; color: {risk_color}; font-weight: bold;">{risk_label}</div>
                <div style="font-size: 0.8rem; color: #666; margin-top: 5px;">🔍 {root_cause}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # === DETAILED ANALYSIS (EXPANDABLE) ===
    with st.expander("📋 Detailed Region Analysis", expanded=False):
        selected = st.selectbox("Select region for detailed view:", ['A', 'B', 'C'], key="detail_select")
        
        if selected in regions_data:
            data = regions_data[selected]
            data['region'] = selected
            data['timestamp'] = datetime.now()
            
            risk, breakdown = calculate_risk_index(data)
            ml_risk = predict_risk(data, model) if model_available else risk
            root_cause = detect_root_cause(data, ml_risk)
            interpretation = generate_ai_interpretation(data, ml_risk, root_cause, breakdown)
            
            render_ai_interpretation(interpretation)
            
            col1, col2 = st.columns(2)
            with col1:
                render_root_cause(root_cause)
            with col2:
                render_risk_breakdown(breakdown)
            
            render_alert(ml_risk, root_cause, selected)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #888; font-size: 0.75rem; margin-top: 30px;">
        💧 AI Smart Water Quality Monitoring System | Real-time Network Visualization<br>
        Adjust sliders to see immediate status changes in the pipe network
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
