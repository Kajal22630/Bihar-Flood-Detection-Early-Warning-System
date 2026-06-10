# 🌊 Bihar Flood Detection & Early Warning System

A **Multi-Model Ensemble Framework** for real-time flood prediction in Bihar using **BiLSTM Neural Networks**, **Sentinel-1 SAR Satellite Data**, and **Hydrological Telemetry**.

## 🎯 Overview

This system provides:
- **Real-time flood risk prediction** at village/district level
- **BiLSTM neural network** for temporal pattern recognition  
- **Google Earth Engine integration** for SAR satellite imagery
- **Interactive web dashboard** with GIS visualization
- **Emergency contact system** with SDRF/NDRF protocols

## 🏗️ Architecture
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Hydrological │────▶│ BiLSTM │────▶│ Ensemble │
│ Telemetry │ │ Neural Engine │ │ Prediction │
└─────────────────┘ └─────────────────┘ └─────────────────┘
│ │ │
▼ ▼ ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Sentinel-1 │ │ Statistical │ │ Risk Level │
│ SAR Data │ │ Baseline │ │ (GREEN/AMBER/ │
│ (GEE) │ │ │ │ RED/CRITICAL) │
└─────────────────┘ └─────────────────┘ └─────────────────┘


## 📊 Model Performance

| Metric | Value |
|--------|-------|
| Sensitivity (Recall) | 96.2% |
| Mean Absolute Error | 0.024 |
| F1-Score | 0.94 |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Google Earth Engine account (for SAR data)

### Installation

```bash
# Clone repository
git clone https://github.com/Kajal22630/Bihar-Flood-Detection-Early-Warning-System.git
cd Bihar-Flood-Detection-Early-Warning-System

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
