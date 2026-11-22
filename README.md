# Z24 Bridge – Phase 1 SHM Digital Twin

Structural Health Monitoring (SHM) Digital Twin for the Swiss Z24 Bridge using a 1D CNN-LSTM Autoencoder for unsupervised anomaly detection and localization on 27-channel acceleration frequency response spectra (0–25 Hz).

## Overview

- **Core Model:** 1D CNN-LSTM Autoencoder (`z24_cnn_lstm_ae.pt`, 65,803 parameters) trained exclusively on undamaged reference data (Scenario 0) to learn normal structural dynamics.
- **Sensor Setup:** 27 acceleration sensor channels across 5 structural zones:
  - **Left Abutment:** Sensors S0–S2
  - **Left Span (30 m):** Sensors S3–S11
  - **Center Pier:** Sensors S12–S15
  - **Right Span (30 m):** Sensors S16–S23
  - **Right Abutment:** Sensors S24–S26
- **Feature Space:** 150 frequency bins (0 to 25 Hz) across 27 channels (`shape: (150, 27)`).
- **Threshold:** `0.493652` MSE (derived from `z24_scaler.npz`).

## Features

1. **3-Tier Structural Health Assessment:**
   - 🟢 **HEALTHY BASELINE:** `max_sensor_mse < threshold`
   - 🟡 **EARLY LOCALIZED WARNING:** `max_sensor_mse >= threshold` (detects early damage at 8%–15% severity before zone-wide collapse)
   - 🔴 **CRITICAL STRUCTURAL ANOMALY:** `worst_zone_avg_mse >= threshold` or `overall_mse >= threshold`
2. **Interactive Multi-Zone Fault Sandbox:**
   - Simultaneously apply independent structural faults across multiple zones:
     - 🏗️ Pier Settlement (stiffness loss & low-frequency modal downshift)
     - ⚡ Concrete Cracking / Spalling (high-frequency distortion & acoustic emission noise)
     - ⛓️ Tendon Rupture (loss of prestress & resonance shifts)
     - ⛰️ Abutment / Foundation Shift (edge support strain & ground drift)
   - Active Faults registry with individual fault removal and 0% healthy reset.
3. **Interactive 2D Bridge Digital Twin (SVG):**
   - Clean architectural profile of the 60m two-span box-girder bridge.
   - Dynamic node coloring (Green / Warning Yellow / Pulsing Red) strictly following model threshold.
   - Click-to-target zone selection and sensor pin-point inspection.
4. **Blind AI Diagnostic Testing:**
   - Secretly feeds raw mystery vibration data to the AI without labels.
   - AI predicts the damaged zone, confidence score, and top stressed sensors.
   - "Reveal Ground Truth" mode allows verifying predictions against actual hidden scenarios.
5. **Operational Noise & Traffic Load Immunity:**
   - Built-in simulation of heavy vehicle traffic and ambient temperature drift to verify the autoencoder doesn't produce false alarms under routine environmental variations.
6. **Z24 Dataset Replay:**
   - Historical replay of recordings from the 1998 Swiss Z24 benchmark.

## Repository Structure

```
├── backend/
│   ├── main.py              # FastAPI server with CNN-LSTM model & SHM endpoints
│   └── requirements.txt     # Python dependencies
├── frontend/
│   └── index.html           # Single-page Digital Twin application
├── z24_cnn_lstm_ae.pt       # Trained PyTorch model weights
├── z24_scaler.npz           # Feature scaler parameters & anomaly threshold
├── z24_demo_samples.npz     # Z24 benchmark scenario samples (85 samples x 150 x 27)
├── start_shm.bat            # One-click startup script for Windows
├── .gitignore
└── README.md
```

## Quick Start

### 1. Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Node.js / npm (optional, for `npx serve`)

### 2. Run with One-Click Script (Windows)
Double-click `start_shm.bat` or run:
```bat
start_shm.bat
```

### 3. Manual Start

**Backend:**
```bash
pip install -r backend/requirements.txt
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npx serve -l 3000 .
```
Open [http://localhost:3000](http://localhost:3000) in your browser. API docs available at [http://localhost:8000/docs](http://localhost:8000/docs).
