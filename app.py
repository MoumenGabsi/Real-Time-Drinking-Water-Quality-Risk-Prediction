"""
AI Smart Water Quality Monitoring Dashboard
Modern, compact design with real-time pipe visualization and PREDICTIVE ANALYTICS.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulation import (
    simulate_sensor_data,
    calculate_risk_index,
    detect_root_cause,
    get_pipe_status,
    generate_ai_interpretation,
    get_temporal_features,
    SENSOR_RANGES,
    REGION_PROFILES,
    IDEAL_VALUES,
    TEMPORAL_RISK_FACTORS,
    CRITICAL_THRESHOLDS,
    SensorHistory,
    calculate_trend,
    generate_predictions,
    generate_early_warnings,
    get_trend_summary,
    simulate_history_for_demo
)
from model import predict_risk, load_model, MODEL_PATH

# Page configuration
st.set_page_config(
    page_title="AquaGuard AI",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS with glassmorphism and enhanced visuals
st.markdown("""
<style>
    /* ===== GLOBAL STYLES ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #162544 50%, #0d1f35 100%);
    }
    
    /* Global button enhancements */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        color: #fff !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
        height: auto !important;
        min-height: 42px !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.4) 0%, rgba(139, 92, 246, 0.4) 100%) !important;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3) !important;
        transform: translateY(-2px) !important;
        border-color: rgba(0, 212, 255, 0.6) !important;
    }
    
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #00ff88, #00d4ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.5px;
        animation: shimmer 3s linear infinite;
    }
    
    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }
    
    .sub-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #8899aa;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    /* ===== WATER FLOW ANIMATIONS ===== */
    @keyframes flowRight {
        0% { background-position: 0% 50%; }
        100% { background-position: 200% 50%; }
    }
    
    @keyframes flowDown {
        0% { background-position: 50% 0%; }
        100% { background-position: 50% 200%; }
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); box-shadow: 0 0 20px rgba(0, 212, 255, 0.4); }
        50% { transform: scale(1.05); box-shadow: 0 0 35px rgba(0, 212, 255, 0.7); }
    }
    
    @keyframes ripple {
        0% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0.4); }
        100% { box-shadow: 0 0 0 15px rgba(0, 212, 255, 0); }
    }
    
    @keyframes waterGlow {
        0%, 100% { opacity: 0.6; }
        50% { opacity: 1; }
    }
    
    /* ===== GLASS CARD ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
    }
    
    /* ===== PIPE NETWORK ===== */
    .network-container {
        background: linear-gradient(180deg, rgba(0, 150, 255, 0.08) 0%, rgba(0, 200, 150, 0.05) 100%);
        border: 1px solid rgba(0, 212, 255, 0.25);
        border-radius: 24px;
        padding: 30px 20px;
        margin: 15px 0;
        position: relative;
        overflow: hidden;
    }
    
    .network-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: radial-gradient(ellipse at 50% 0%, rgba(0, 212, 255, 0.1) 0%, transparent 70%);
        pointer-events: none;
    }
    
    .source-node {
        background: linear-gradient(135deg, #00d4ff 0%, #00a5cc 50%, #0088aa 100%);
        color: #001a2c;
        padding: 14px 32px;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.95rem;
        display: inline-block;
        box-shadow: 0 0 40px rgba(0, 212, 255, 0.5), inset 0 2px 10px rgba(255,255,255,0.3);
        letter-spacing: 0.5px;
        animation: pulse 2s ease-in-out infinite;
        position: relative;
    }
    
    .source-node::after {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 100%;
        height: 100%;
        border-radius: 30px;
        transform: translate(-50%, -50%);
        animation: ripple 2s ease-out infinite;
    }
    
    .main-pipe {
        background: linear-gradient(90deg, 
            rgba(0, 180, 255, 0.2),
            rgba(0, 212, 255, 0.5),
            rgba(0, 255, 200, 0.5),
            rgba(0, 212, 255, 0.5),
            rgba(0, 180, 255, 0.2));
        background-size: 200% 100%;
        height: 10px;
        border-radius: 5px;
        margin: 0 8%;
        position: relative;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4), inset 0 2px 4px rgba(255,255,255,0.2);
        animation: flowRight 2s linear infinite;
    }
    
    .main-pipe::before {
        content: '';
        position: absolute;
        top: 2px;
        left: 10%;
        right: 10%;
        height: 3px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
        border-radius: 2px;
        animation: waterGlow 1.5s ease-in-out infinite;
    }
    
    .v-pipe {
        width: 10px;
        border-radius: 5px;
        margin: 0 auto;
        position: relative;
        box-shadow: 0 0 15px currentColor;
    }
    
    .v-pipe-animated {
        background: linear-gradient(180deg,
            rgba(0, 180, 255, 0.3),
            rgba(0, 212, 255, 0.6),
            rgba(0, 255, 200, 0.6),
            rgba(0, 212, 255, 0.6),
            rgba(0, 180, 255, 0.3)) !important;
        background-size: 100% 200% !important;
        animation: flowDown 1.5s linear infinite !important;
    }
    
    .hub-node {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f2744 100%);
        color: #00d4ff;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto;
        font-size: 1.3rem;
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.4), inset 0 0 20px rgba(0, 212, 255, 0.1);
        border: 2px solid rgba(0, 212, 255, 0.4);
        animation: pulse 3s ease-in-out infinite;
    }
    
    .region-card {
        background: linear-gradient(180deg, rgba(15, 30, 50, 0.9) 0%, rgba(10, 20, 40, 0.95) 100%);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 18px;
        text-align: center;
        border: 2px solid;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .region-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 60%;
        background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, transparent 100%);
        pointer-events: none;
    }
    
    .region-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4);
    }
    
    .region-icon {
        font-size: 2.2rem;
        margin-bottom: 8px;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));
    }
    
    .region-name {
        color: #fff;
        font-weight: 700;
        font-size: 1.05rem;
        margin: 5px 0;
        text-shadow: 0 1px 3px rgba(0,0,0,0.3);
    }
    
    .region-subname {
        color: #8899aa;
        font-size: 0.78rem;
        margin-bottom: 12px;
    }
    
    .risk-value {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 8px 0;
        text-shadow: 0 2px 10px currentColor;
    }
    
    .risk-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        padding: 4px 12px;
        border-radius: 12px;
        display: inline-block;
        margin: 8px 0;
    }
    
    .status-pill {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.5px;
        padding: 6px 14px;
        border-radius: 20px;
        display: inline-block;
        color: white;
        text-transform: uppercase;
    }
    
    /* Status Colors */
    .status-normal { background: linear-gradient(135deg, #10b981 0%, #059669 100%); box-shadow: 0 0 15px rgba(16, 185, 129, 0.4); }
    .status-leak { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); box-shadow: 0 0 15px rgba(239, 68, 68, 0.4); }
    .status-contamination { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); box-shadow: 0 0 15px rgba(245, 158, 11, 0.4); }
    .status-decay { background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%); box-shadow: 0 0 15px rgba(139, 92, 246, 0.4); }
    .status-stagnation { background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); color: #1a1a1a; box-shadow: 0 0 15px rgba(251, 191, 36, 0.4); }
    
    /* Risk Colors */
    .risk-safe { color: #10b981; }
    .risk-warning { color: #f59e0b; }
    .risk-critical { color: #ef4444; }
    
    .label-safe { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    .label-warning { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
    .label-critical { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
    
    /* ===== TEMPORAL INFO BAR ===== */
    .temporal-bar {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.15) 0%, rgba(59, 130, 246, 0.15) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 16px;
        padding: 12px 20px;
        margin: 15px auto;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 30px;
        flex-wrap: wrap;
        max-width: 800px;
    }
    
    .temporal-item {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #c4d4e4;
        font-size: 0.85rem;
    }
    
    .temporal-icon {
        font-size: 1.1rem;
    }
    
    .temporal-value {
        font-weight: 600;
        color: #fff;
    }
    
    .temporal-modifier {
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .modifier-low { background: rgba(16, 185, 129, 0.3); color: #10b981; }
    .modifier-medium { background: rgba(245, 158, 11, 0.3); color: #f59e0b; }
    .modifier-high { background: rgba(239, 68, 68, 0.3); color: #ef4444; }
    
    /* ===== LEGEND ===== */
    .legend-bar {
        display: flex;
        justify-content: center;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 15px;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.7rem;
        color: #8899aa;
        background: rgba(255, 255, 255, 0.05);
        padding: 6px 12px;
        border-radius: 15px;
    }
    
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
    }
    
    /* ===== CONTROL BUTTONS ===== */
    .control-btn {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: #00d4ff;
        padding: 8px 16px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .control-btn:hover {
        background: rgba(0, 212, 255, 0.3);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* ===== AI PANEL ===== */
    .ai-panel {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(139, 92, 246, 0.2);
        border-radius: 16px;
        padding: 15px 20px;
        margin: 15px 0;
    }
    
    .ai-title {
        color: #a78bfa;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    .ai-content {
        display: flex;
        flex-direction: column;
        gap: 6px;
    }
    
    .ai-item {
        color: #c4b5fd;
        font-size: 0.85rem;
        line-height: 1.5;
        padding: 6px 12px;
        background: rgba(139, 92, 246, 0.08);
        border-radius: 8px;
        border-left: 3px solid rgba(139, 92, 246, 0.4);
    }
    
    .ai-item:first-child {
        font-weight: 600;
        font-size: 0.9rem;
        background: rgba(139, 92, 246, 0.15);
        border-left-color: #a78bfa;
    }
    
    .ai-text {
        color: #c4b5fd;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    
    /* ===== HIDE STREAMLIT DEFAULTS ===== */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stPopover {
        background: rgba(15, 23, 42, 0.95) !important;
        border: 1px solid rgba(0, 212, 255, 0.2) !important;
        border-radius: 16px !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.2) 0%, rgba(0, 212, 255, 0.1) 100%);
        border: 1px solid rgba(0, 212, 255, 0.3);
        color: #00d4ff;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: rgba(0, 212, 255, 0.3);
        border-color: rgba(0, 212, 255, 0.5);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Enhanced Slider styling */
    .stSlider > div > div {
        background: rgba(0, 212, 255, 0.15) !important;
        border-radius: 10px !important;
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00d4ff, #00ff88) !important;
        border-radius: 10px !important;
    }
    
    .stSlider > div > div > div > div {
        background: #fff !important;
        border: 3px solid #00d4ff !important;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.6) !important;
    }
    
    .stSlider label {
        color: #a0d4e8 !important;
        font-weight: 500 !important;
    }
    
    /* Slider min/max values - make them brighter */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #ffffff !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }
    
    /* Alternative selector for slider range text */
    .stSlider > div > div:last-child {
        color: #ffffff !important;
    }
    
    .stSlider small {
        color: #ffffff !important;
        font-weight: 500 !important;
    }
    }
    
    /* ===== CONTROL PANEL ===== */
    .control-panel {
        background: linear-gradient(145deg, rgba(0, 30, 60, 0.9) 0%, rgba(0, 20, 40, 0.95) 100%);
        border: 2px solid rgba(0, 212, 255, 0.4);
        border-radius: 20px;
        padding: 20px 25px;
        margin-top: 49px;
        box-shadow: 
            0 0 30px rgba(0, 212, 255, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(20px);
    }
    
    .control-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    .control-icon {
        width: 45px;
        height: 45px;
        background: linear-gradient(135deg, #00d4ff 0%, #0099cc 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
    }
    
    .control-title {
        color: #fff;
        font-size: 1.1rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .control-subtitle {
        color: #00d4ff;
        font-size: 0.8rem;
        font-weight: 500;
        opacity: 0.9;
    }
    
    .slider-group {
        background: rgba(0, 212, 255, 0.05);
        border-radius: 12px;
        padding: 12px 15px;
        margin: 8px 0;
        border-left: 3px solid rgba(0, 212, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .slider-group:hover {
        background: rgba(0, 212, 255, 0.1);
        border-left-color: #00d4ff;
    }
    
    .done-btn-container {
        margin-top: 20px;
        padding-top: 15px;
        border-top: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    /* ===== ENHANCED BUTTON STYLES ===== */
    .prediction-panel .stButton > button {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.3) 0%, rgba(139, 92, 246, 0.3) 100%) !important;
        border: 1px solid rgba(236, 72, 153, 0.4) !important;
        color: #fff !important;
        font-weight: 600 !important;
        padding: 10px 20px !important;
        border-radius: 12px !important;
        transition: all 0.3s ease !important;
    }
    
    .prediction-panel .stButton > button:hover {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.5) 0%, rgba(139, 92, 246, 0.5) 100%) !important;
        box-shadow: 0 0 20px rgba(236, 72, 153, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* ===== PREDICTIVE ANALYTICS PANEL ===== */
    .prediction-panel {
        background: linear-gradient(135deg, rgba(236, 72, 153, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(236, 72, 153, 0.25);
        border-radius: 20px;
        padding: 20px;
        margin: 20px 0;
    }
    
    .prediction-title {
        color: #ec4899;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .trend-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
        gap: 12px;
        margin: 15px 0;
    }
    
    .trend-item {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 12px;
        padding: 12px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .trend-sensor {
        color: #a0aec0;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    
    .trend-value {
        font-size: 1.2rem;
        font-weight: 700;
        margin: 5px 0;
    }
    
    .trend-rate {
        font-size: 0.7rem;
        padding: 3px 8px;
        border-radius: 10px;
        display: inline-block;
    }
    
    .trend-up { color: #f59e0b; }
    .trend-down { color: #3b82f6; }
    .trend-stable { color: #10b981; }
    .trend-unknown { color: #6b7280; }
    
    .rate-up { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
    .rate-down { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
    .rate-stable { background: rgba(16, 185, 129, 0.2); color: #10b981; }
    
    /* Early Warning Styles */
    .warning-container {
        margin-top: 15px;
    }
    
    .early-warning {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 15px;
        border-radius: 12px;
        margin: 8px 0;
        animation: warningPulse 2s ease-in-out infinite;
    }
    
    .warning-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(239, 68, 68, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    
    .warning-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(245, 158, 11, 0.1) 100%);
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    
    .warning-icon {
        font-size: 1.5rem;
    }
    
    .warning-content {
        flex: 1;
    }
    
    .warning-title {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 3px;
    }
    
    .warning-danger .warning-title { color: #ef4444; }
    .warning-warning .warning-title { color: #f59e0b; }
    
    .warning-detail {
        font-size: 0.75rem;
        color: #a0aec0;
    }
    
    .warning-time {
        background: rgba(0, 0, 0, 0.3);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .warning-danger .warning-time { color: #ef4444; }
    .warning-warning .warning-time { color: #f59e0b; }
    
    @keyframes warningPulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.85; }
    }
    
    .no-warnings {
        text-align: center;
        padding: 20px;
        color: #10b981;
        font-size: 0.9rem;
    }
    
    .prediction-forecast {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 12px;
        padding: 15px;
        margin-top: 15px;
    }
    
    .forecast-title {
        color: #a78bfa;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    
    .forecast-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        font-size: 0.8rem;
    }
    
    .forecast-sensor {
        color: #a0aec0;
    }
    
    .forecast-values {
        display: flex;
        gap: 15px;
    }
    
    .forecast-time {
        color: #6b7280;
        font-size: 0.7rem;
    }
</style>
""", unsafe_allow_html=True)


def get_risk_color(risk: float) -> str:
    if risk < 30:
        return "#10b981"
    elif risk < 60:
        return "#f59e0b"
    return "#ef4444"


def get_risk_class(risk: float) -> str:
    if risk < 30:
        return "safe"
    elif risk < 60:
        return "warning"
    return "critical"


def get_risk_label(risk: float) -> str:
    if risk < 30:
        return "SAFE"
    elif risk < 60:
        return "WARNING"
    return "CRITICAL"


def main():
    # Initialize session state with defaults
    if 'manual_values' not in st.session_state:
        st.session_state.manual_values = {
            'A': {'temperature': 20.0, 'flow': 2.5, 'pressure': 4.0, 'chlorine': 1.2, 'pH': 7.2, 'turbidity': 0.5, 'conductivity': 380},
            'B': {'temperature': 22.0, 'flow': 2.1, 'pressure': 3.6, 'chlorine': 1.0, 'pH': 7.3, 'turbidity': 0.6, 'conductivity': 420},
            'C': {'temperature': 19.0, 'flow': 2.7, 'pressure': 4.4, 'chlorine': 1.3, 'pH': 7.1, 'turbidity': 0.4, 'conductivity': 360}
        }
    
    # Track which region is being edited (None = no editing)
    if 'editing_region' not in st.session_state:
        st.session_state.editing_region = None
    
    # Initialize sensor history for predictive analytics
    if 'sensor_history' not in st.session_state:
        st.session_state.sensor_history = SensorHistory(max_history=24)
    
    # Initialize demo scenario
    if 'demo_scenario' not in st.session_state:
        st.session_state.demo_scenario = 'normal'
    
    # Load model
    try:
        model = load_model()
        model_available = True
    except:
        model = None
        model_available = False
    
    # ===== MAIN CONTENT =====
    # Header
    st.markdown('<h1 class="main-title">💧 AquaGuard AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Real-time Water Quality Intelligence System</p>', unsafe_allow_html=True)
    
    # Video Demo Button
    st.markdown('''
    <div style="text-align: center; margin: 15px 0;">
        <a href="https://drive.google.com/file/d/1DTbMbCOfG-h2KtBr2mweSFL8sDKBzo_s/view" target="_blank" style="text-decoration: none;">
            <div style="display: inline-flex; align-items: center; gap: 10px; background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(236, 72, 153, 0.2) 100%); border: 1px solid rgba(239, 68, 68, 0.4); padding: 12px 24px; border-radius: 30px; cursor: pointer; transition: all 0.3s ease;">
                <span style="font-size: 1.3rem;">🎬</span>
                <span style="color: #fff; font-weight: 600; font-size: 0.95rem;">Watch Demo Video</span>
            </div>
        </a>
    </div>
    ''', unsafe_allow_html=True)
    
    # Function to get region info from data
    def compute_region_info(region, data):
        status, color = get_pipe_status(data)
        risk, breakdown = calculate_risk_index(data)
        if model_available:
            ml_risk = predict_risk(data, model)
        else:
            ml_risk = risk
        root_cause = detect_root_cause(data, ml_risk)
        return {
            'status': status,
            'color': color,
            'risk': ml_risk,
            'breakdown': breakdown,
            'root_cause': root_cause,
            'name': REGION_PROFILES[region]['name']
        }
    
    # Get data for all regions from stored values
    def get_region_data(region):
        vals = st.session_state.manual_values[region]
        temporal = get_temporal_features(datetime.now())
        return {
            'temperature': vals['temperature'],
            'flow': vals['flow'],
            'pressure': vals['pressure'],
            'chlorine': vals['chlorine'],
            'pH': vals['pH'],
            'turbidity': vals['turbidity'],
            'conductivity': vals.get('conductivity', 400),
            'hour': temporal['hour'],
            'is_weekend': temporal['is_weekend'],
            'temporal_risk_modifier': temporal['temporal_risk_modifier'],
            'time_period': temporal['time_period']
        }
    
    # ===== MAIN NETWORK VISUALIZATION =====
    st.markdown('<div class="network-container">', unsafe_allow_html=True)
    
    # Source
    st.markdown('''
    <div style="text-align: center; margin-bottom: 10px;">
        <span class="source-node">💧 WATER SOURCE</span>
    </div>
    <div style="display: flex; justify-content: center; margin: 8px 0;">
        <div class="v-pipe v-pipe-animated" style="height: 30px;"></div>
    </div>
    <div class="main-pipe"></div>
    <div style="display: flex; justify-content: center; margin: 8px 0;">
        <div class="v-pipe v-pipe-animated" style="height: 20px;"></div>
    </div>
    <div style="text-align: center; margin-bottom: 10px;">
        <div class="hub-node">⚡</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Distribution bar
    st.markdown('<div class="main-pipe" style="height: 6px; margin: 5px 5%;"></div>', unsafe_allow_html=True)
    
    # Helper to render a region card with given data/info
    def render_card(region, data, info):
        risk_class = get_risk_class(info['risk'])
        status_class = f"status-{info['status']}"
        icon = '🏭' if region == 'A' else '🏠' if region == 'B' else '🏪'
        
        # Pipe color based on status - use gradient for flow effect
        if info['risk'] < 30:
            pipe_gradient = "linear-gradient(180deg, rgba(16, 185, 129, 0.3), rgba(0, 212, 255, 0.6), rgba(16, 185, 129, 0.3))"
        elif info['risk'] < 60:
            pipe_gradient = "linear-gradient(180deg, rgba(245, 158, 11, 0.3), rgba(255, 200, 50, 0.6), rgba(245, 158, 11, 0.3))"
        else:
            pipe_gradient = "linear-gradient(180deg, rgba(239, 68, 68, 0.3), rgba(255, 100, 100, 0.6), rgba(239, 68, 68, 0.3))"
        
        # Vertical pipe with animation
        st.markdown(f'''
        <div style="display: flex; justify-content: center; margin: 5px 0;">
            <div class="v-pipe" style="height: 40px; background: {pipe_gradient}; background-size: 100% 200%; animation: flowDown 1.5s linear infinite;"></div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Region card
        st.markdown(f'''
        <div class="region-card" style="border-color: {info['color']};">
            <div class="region-icon">{icon}</div>
            <div class="region-name">Region {region}</div>
            <div class="region-subname">{info['name']}</div>
            <div class="risk-value risk-{risk_class}">{info['risk']:.1f}</div>
            <div class="risk-label label-{risk_class}">{get_risk_label(info['risk'])}</div>
            <div><span class="{status_class} status-pill">{info['status'].upper()}</span></div>
        </div>
        ''', unsafe_allow_html=True)
    
    # Helper to render controls and return LIVE data
    def render_controls(region):
        current = st.session_state.manual_values[region]
        region_name = REGION_PROFILES[region]['name']
        icon = '🏭' if region == 'A' else '🏠' if region == 'B' else '🏪'
        
        # Enhanced control panel header
        st.markdown(f'''
        <div class="control-panel">
            <div class="control-header">
                <div class="control-icon">{icon}</div>
                <div>
                    <div class="control-title">Sensor Controls</div>
                    <div class="control-subtitle">Region {region} • {region_name}</div>
                </div>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Sliders with enhanced grouping
        st.markdown('<div class="slider-group">', unsafe_allow_html=True)
        new_temp = st.slider("🌡️ Temperature (°C)", 10.0, 30.0, float(current['temperature']), 0.5, key=f"t_{region}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="slider-group">', unsafe_allow_html=True)
        new_chlorine = st.slider("🧪 Chlorine (mg/L)", 0.1, 2.0, float(current['chlorine']), 0.05, key=f"c_{region}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="slider-group">', unsafe_allow_html=True)
        new_ph = st.slider("⚗️ pH Level", 6.5, 8.5, float(current['pH']), 0.1, key=f"ph_{region}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="slider-group">', unsafe_allow_html=True)
        new_turb = st.slider("💨 Turbidity (NTU)", 0.1, 5.0, float(current['turbidity']), 0.1, key=f"tb_{region}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="slider-group">', unsafe_allow_html=True)
        new_press = st.slider("📊 Pressure (bar)", 2.0, 6.0, float(current['pressure']), 0.1, key=f"pr_{region}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="slider-group">', unsafe_allow_html=True)
        new_flow = st.slider("🌊 Flow (m³/h)", 0.5, 5.0, float(current['flow']), 0.1, key=f"fl_{region}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="slider-group">', unsafe_allow_html=True)
        new_cond = st.slider("⚡ Conductivity (µS/cm)", 200, 800, int(current.get('conductivity', 400)), 10, key=f"cd_{region}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Build live data from slider values (with temporal features)
        temporal = get_temporal_features(datetime.now())
        live_data = {
            'temperature': new_temp,
            'chlorine': new_chlorine,
            'pH': new_ph,
            'turbidity': new_turb,
            'pressure': new_press,
            'flow': new_flow,
            'conductivity': new_cond,
            'hour': temporal['hour'],
            'is_weekend': temporal['is_weekend'],
            'temporal_risk_modifier': temporal['temporal_risk_modifier'],
            'time_period': temporal['time_period']
        }
        
        # Save to session state so it persists (without temporal - those are calculated fresh)
        st.session_state.manual_values[region] = {
            'temperature': new_temp,
            'chlorine': new_chlorine,
            'pH': new_ph,
            'turbidity': new_turb,
            'pressure': new_press,
            'flow': new_flow,
            'conductivity': new_cond
        }
        
        # Enhanced done button
        st.markdown('<div class="done-btn-container">', unsafe_allow_html=True)
        if st.button("✅ Done - Apply Changes", key=f"done_{region}", use_container_width=True):
            st.session_state.editing_region = None
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
        return live_data
    
    editing = st.session_state.editing_region
    
    # Store computed data/info for all regions
    all_data = {}
    all_info = {}
    
    # Layout depends on whether we're editing
    if editing is None:
        # Normal view - 3 columns with adjust buttons
        cols = st.columns(3)
        for idx, region in enumerate(['A', 'B', 'C']):
            data = get_region_data(region)
            info = compute_region_info(region, data)
            all_data[region] = data
            all_info[region] = info
            
            with cols[idx]:
                render_card(region, data, info)
                if st.button(f"⚙️ Adjust Region {region}", key=f"edit_{region}", use_container_width=True):
                    st.session_state.editing_region = region
                    st.rerun()
    else:
        # Editing mode - render controls FIRST to get live values, then card
        if editing == 'A':
            # Region A: [Controls | Card A] (controls first so we get live values)
            top_cols = st.columns([1, 1])
            with top_cols[0]:
                live_data = render_controls('A')
            
            # Compute LIVE info from slider values
            live_info = compute_region_info('A', live_data)
            all_data['A'] = live_data
            all_info['A'] = live_info
            
            with top_cols[1]:
                render_card('A', live_data, live_info)
            
            # Show B and C below
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            other_cols = st.columns([1, 1, 2])
            for idx, region in enumerate(['B', 'C']):
                data = get_region_data(region)
                info = compute_region_info(region, data)
                all_data[region] = data
                all_info[region] = info
                with other_cols[idx]:
                    render_card(region, data, info)
                
        elif editing == 'B':
            # Region B in the middle with controls to the right
            top_cols = st.columns([0.6, 1, 1, 0.6])
            
            # A on left
            data_a = get_region_data('A')
            info_a = compute_region_info('A', data_a)
            all_data['A'] = data_a
            all_info['A'] = info_a
            with top_cols[0]:
                render_card('A', data_a, info_a)
            
            # Controls for B (get live values)
            with top_cols[1]:
                live_data = render_controls('B')
            
            live_info = compute_region_info('B', live_data)
            all_data['B'] = live_data
            all_info['B'] = live_info
            
            # Card B with live values
            with top_cols[2]:
                render_card('B', live_data, live_info)
            
            # C on right
            data_c = get_region_data('C')
            info_c = compute_region_info('C', data_c)
            all_data['C'] = data_c
            all_info['C'] = info_c
            with top_cols[3]:
                render_card('C', data_c, info_c)
                
        else:  # editing == 'C'
            # Region C: [Controls | Card C]
            top_cols = st.columns([1, 1])
            with top_cols[0]:
                live_data = render_controls('C')
            
            live_info = compute_region_info('C', live_data)
            all_data['C'] = live_data
            all_info['C'] = live_info
            
            with top_cols[1]:
                render_card('C', live_data, live_info)
            
            # Show A and B below
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            other_cols = st.columns([2, 1, 1])
            for idx, region in enumerate(['A', 'B']):
                data = get_region_data(region)
                info = compute_region_info(region, data)
                all_data[region] = data
                all_info[region] = info
                with other_cols[idx + 1]:
                    render_card(region, data, info)
    
    # Legend
    st.markdown('''
    <div class="legend-bar">
        <div class="legend-item"><div class="legend-dot" style="background: #10b981;"></div>Normal</div>
        <div class="legend-item"><div class="legend-dot" style="background: #ef4444;"></div>Leak</div>
        <div class="legend-item"><div class="legend-dot" style="background: #f59e0b;"></div>Contamination</div>
        <div class="legend-item"><div class="legend-dot" style="background: #8b5cf6;"></div>Decay</div>
        <div class="legend-item"><div class="legend-dot" style="background: #fbbf24;"></div>Stagnation</div>
    </div>
    ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== AI INTERPRETATION PANEL =====
    # Find the region with highest risk
    worst_region = max(all_info.keys(), key=lambda r: all_info[r]['risk'])
    worst_info = all_info[worst_region]
    
    interpretation = generate_ai_interpretation(
        all_data[worst_region],
        worst_info['risk'],
        worst_info['root_cause'],
        worst_info['breakdown']
    )
    
    # Format interpretation as HTML list
    interpretation_items = interpretation.split(' | ')
    interpretation_html = ''.join([f'<div class="ai-item">{item}</div>' for item in interpretation_items])
    
    st.markdown(f'''
    <div class="ai-panel">
        <div class="ai-title">🤖 AI Analysis - Highest Risk: Region {worst_region}</div>
        <div class="ai-content">{interpretation_html}</div>
    </div>
    ''', unsafe_allow_html=True)
    
    # ===== PREDICTIVE ANALYTICS PANEL =====
    st.markdown('<div class="prediction-panel">', unsafe_allow_html=True)
    st.markdown('<div class="prediction-title">🔮 Predictive Analytics - Early Warning System</div>', unsafe_allow_html=True)
    
    # Demo scenario selector and record button
    pred_cols = st.columns([2, 1, 1])
    with pred_cols[0]:
        demo_scenario = st.selectbox(
            "📊 Simulation Scenario",
            ['normal', 'degrading_chlorine', 'rising_turbidity', 'pressure_drop'],
            index=['normal', 'degrading_chlorine', 'rising_turbidity', 'pressure_drop'].index(st.session_state.demo_scenario),
            key='scenario_select',
            help="Select a scenario to see how the predictive system detects degradation patterns"
        )
        st.session_state.demo_scenario = demo_scenario
    
    with pred_cols[1]:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)  # Spacer to align with selectbox
        if st.button("📝 Record Current", key="record_btn", help="Add current sensor values to history", use_container_width=True):
            for region in ['A', 'B', 'C']:
                st.session_state.sensor_history.add_reading(region, all_data[region])
            st.success("Recorded!")
    
    with pred_cols[2]:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)  # Spacer to align with selectbox
        if st.button("🎬 Load Demo History", key="demo_btn", help="Load 12h of simulated history", use_container_width=True):
            for region in ['A', 'B', 'C']:
                demo_history = simulate_history_for_demo(region, hours=12, scenario=demo_scenario)
                st.session_state.sensor_history = demo_history
            st.success(f"Loaded {demo_scenario} scenario!")
            st.rerun()
    
    # Show predictions for each region
    for region in ['A', 'B', 'C']:
        region_name = REGION_PROFILES[region]['name']
        history_count = len(st.session_state.sensor_history.get_values(region, 'chlorine'))
        
        if history_count >= 3:
            predictions = generate_predictions(st.session_state.sensor_history, region)
            warnings = generate_early_warnings(predictions, region)
            trend_summary = get_trend_summary(predictions)
            
            # Region header
            st.markdown(f"<h4 style='color: #ec4899; margin-top: 15px;'>Region {region} - {region_name}</h4>", unsafe_allow_html=True)
            
            # Trend grid - render as columns instead of HTML grid
            trend_cols = st.columns(6)
            for idx, sensor in enumerate(['chlorine', 'pH', 'turbidity', 'pressure', 'flow', 'conductivity']):
                if sensor in trend_summary:
                    ts = trend_summary[sensor]
                    current_val = predictions[sensor]['trend'].get('current_value', 'N/A')
                    if isinstance(current_val, float):
                        current_val = f"{current_val:.2f}"
                    
                    # Color based on direction
                    if ts['direction'] == 'increasing':
                        color = "#f59e0b"
                        symbol = "↑"
                    elif ts['direction'] == 'decreasing':
                        color = "#3b82f6"
                        symbol = "↓"
                    else:
                        color = "#10b981"
                        symbol = "→"
                    
                    with trend_cols[idx]:
                        st.markdown(f"""
                        <div style="background: rgba(0,0,0,0.3); border-radius: 12px; padding: 12px; text-align: center; border: 1px solid rgba(255,255,255,0.1);">
                            <div style="color: #a0aec0; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px;">{sensor}</div>
                            <div style="font-size: 1.2rem; font-weight: 700; color: {color}; margin: 5px 0;">{symbol} {current_val}</div>
                            <div style="font-size: 0.7rem; padding: 3px 8px; border-radius: 10px; background: rgba(255,255,255,0.1); color: {color}; display: inline-block;">{ts['rate_str']}</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Early warnings
            if warnings:
                st.markdown('<div class="warning-container">', unsafe_allow_html=True)
                for w in warnings[:3]:  # Show top 3 warnings
                    warning_class = f"warning-{w['type']}"
                    icon = "🚨" if w['type'] == 'danger' else "⚠️"
                    st.markdown(f'''
                    <div class="early-warning {warning_class}">
                        <div class="warning-icon">{icon}</div>
                        <div class="warning-content">
                            <div class="warning-title">{w['sensor'].title()} Alert</div>
                            <div class="warning-detail">{w['message']}</div>
                        </div>
                        <div class="warning-time">~{w['hours_until']:.1f}h</div>
                    </div>
                    ''', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="no-warnings">✅ No degradation trends detected - System stable</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="color: #6b7280; padding: 10px; text-align: center;">
                Region {region}: Need more data points ({history_count}/3 minimum). 
                Click "Record Current" or "Load Demo History" to add data.
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # ===== QUICK STATS BAR =====
    stat_cols = st.columns(4)
    
    with stat_cols[0]:
        avg_risk = np.mean([r['risk'] for r in all_info.values()])
        st.metric("📊 Avg Risk", f"{avg_risk:.1f}", delta=None)
    
    with stat_cols[1]:
        critical_count = sum(1 for r in all_info.values() if r['risk'] >= 60)
        st.metric("🚨 Critical", f"{critical_count}", delta=None)
    
    with stat_cols[2]:
        warning_count = sum(1 for r in all_info.values() if 30 <= r['risk'] < 60)
        st.metric("⚠️ Warning", f"{warning_count}", delta=None)
    
    with stat_cols[3]:
        safe_count = sum(1 for r in all_info.values() if r['risk'] < 30)
        st.metric("✅ Safe", f"{safe_count}", delta=None)
    
    # ===== DETAILED ANALYSIS EXPANDER =====
    with st.expander("📋 Detailed Analysis & Raw Data"):
        detail_cols = st.columns(3)
        
        for idx, region in enumerate(['A', 'B', 'C']):
            with detail_cols[idx]:
                info = all_info[region]
                data = all_data[region]
                
                st.markdown(f"**Region {region} - {info['name']}**")
                st.markdown(f"Risk: **{info['risk']:.1f}** | Status: **{info['status'].upper()}**")
                st.markdown(f"Root Cause: {info['root_cause']}")
                
                st.caption("Sensor Values:")
                st.text(f"🌡️ Temp: {data['temperature']:.1f}°C")
                st.text(f"🧪 Chlorine: {data['chlorine']:.2f} mg/L")
                st.text(f"⚗️ pH: {data['pH']:.1f}")
                st.text(f"💨 Turbidity: {data['turbidity']:.2f} NTU")
                st.text(f"📊 Pressure: {data['pressure']:.1f} bar")
                st.text(f"🌊 Flow: {data['flow']:.1f} m³/h")
                
                # Risk breakdown
                st.caption("Risk Breakdown:")
                breakdown = info['breakdown']
                for key in ['chlorine_risk', 'turbidity_risk', 'pH_risk', 'pressure_risk', 'flow_risk']:
                    if key in breakdown:
                        name = key.replace('_risk', '').upper()
                        st.progress(min(breakdown[key] / 30, 1.0), text=f"{name}: {breakdown[key]:.1f}")
                
                if breakdown.get('interaction_risk', 0) > 0:
                    st.warning(f"⚡ Interaction Risk: +{breakdown['interaction_risk']:.0f}")
    
    # Footer
    st.markdown('''
    <div style="text-align: center; color: #4a5568; font-size: 0.75rem; margin-top: 30px; padding: 15px;">
        💧 AquaGuard AI | Powered by Machine Learning | Real-time Water Quality Monitoring<br>
        <span style="color: #00d4ff;">●</span> System Online | Model: {'✅ Ready' if model_available else '❌ Not Loaded'}
    </div>
    '''.replace("{'✅ Ready' if model_available else '❌ Not Loaded'}", '✅ Ready' if model_available else '❌ Not Loaded'), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
