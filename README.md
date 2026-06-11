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

## Technologies
- Python
- TensorFlow
- Flask
- Google Earth Engine
- Pandas
- NumPy

## Features
- Flood Risk Prediction
- Real-Time Monitoring
- Early Warning Alerts
- Interactive Dashboard

## Dataset
Weather, Rainfall, River Level, and SAR Imagery Data

## Authors
Kajal Kumari
B.Tech CSE (AI)
2022-2026
