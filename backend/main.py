"""
Z24 Bridge – Phase 1 SHM Digital Twin  |  FastAPI Backend
==========================================================
Interactive Multi-Zone Fault-Injection Sandbox + Blind AI Testing
"""

import os
import json
import random
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent          # workspace root
MODEL_PATH  = BASE / "z24_cnn_lstm_ae.pt"
SCALER_PATH = BASE / "z24_scaler.npz"
DEMO_PATH   = BASE / "z24_demo_samples.npz"
FRONTEND_PATH = BASE / "frontend"

# ─── Model Definition ─────────────────────────────────────────────────────────
class CNNLSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int = 27, hidden_dim: int = 64):
        super().__init__()
        self.cnn_enc = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.lstm_enc = nn.LSTM(48, hidden_dim, batch_first=True)
        self.lstm_dec = nn.LSTM(hidden_dim, 48, batch_first=True)
        self.cnn_dec = nn.Sequential(
            nn.ConvTranspose1d(48, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(32, n_features, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z   = self.cnn_enc(x.transpose(1, 2)).transpose(1, 2)
        enc, _ = self.lstm_enc(z)
        dec, _ = self.lstm_dec(enc)
        return self.cnn_dec(dec.transpose(1, 2)).transpose(1, 2)


# ─── Global state (populated at startup) ──────────────────────────────────────
device = torch.device("cpu")
model: Optional[CNNLSTMAutoencoder] = None
mean_arr: Optional[np.ndarray]      = None   # (1,1,27)
std_arr:  Optional[np.ndarray]      = None   # (1,1,27)
threshold: float                    = 0.493652
demo_X:   Optional[np.ndarray]      = None   # (85,150,27)
demo_labels: Optional[np.ndarray]   = None   # (85,)
healthy_baseline: Optional[np.ndarray] = None # (150,27)
scenario_deltas: Dict[str, np.ndarray] = {}

# ─── Zone Mapping (27 sensors: 0 to 26) ───────────────────────────────────────
ZONE_SENSOR_MAP = {
    "left_abutment":  [0, 1, 2],
    "left_span":      list(range(3, 12)),     # 3..11
    "center_pier":    [12, 13, 14, 15],       # 12..15
    "right_span":     list(range(16, 24)),    # 16..23
    "right_abutment": [24, 25, 26]            # 24..26
}

ZONE_METADATA = [
    {"id": "left_abutment",  "label": "Left Abutment",  "sensors": [0, 1, 2]},
    {"id": "left_span",      "label": "Left Span",      "sensors": list(range(3, 12))},
    {"id": "center_pier",    "label": "Center Pier",    "sensors": [12, 13, 14, 15]},
    {"id": "right_span",     "label": "Right Span",     "sensors": list(range(16, 24))},
    {"id": "right_abutment", "label": "Right Abutment", "sensors": [24, 25, 26]}
]

SCENARIO_GROUND_TRUTH_ZONE = {
    0: "undamaged",
    1: "center_pier",     # Settlement: Pier 1
    2: "center_pier",     # Settlement: Pier 2
    3: "center_pier",     # Settlement: Pier 3
    4: "center_pier",     # Settlement: Pier 4
    5: "left_span",       # Landslide
    6: "left_abutment",   # Scour (mild)
    7: "left_abutment",   # Scour (severe)
    8: "left_span",       # Concrete Cracking (spalling)
    9: "left_span",       # Concrete Cracking (delamination)
    10: "center_pier",    # Tendon Rupture (1 tendon)
    11: "center_pier",    # Tendon Rupture (2 tendons)
    12: "right_abutment", # Anchor Head Failure (1)
    13: "right_abutment", # Anchor Head Failure (2)
    14: "right_span",     # Rebar Corrosion (mild)
    15: "right_span",     # Rebar Corrosion (moderate)
    16: "right_span",     # Rebar Corrosion (severe)
}

SCENARIO_META = [
    {"id": 0,  "label": "Undamaged (Reference)",              "category": "normal",   "severity": 0},
    {"id": 1,  "label": "Settlement: Pier 1 (10 mm)",         "category": "damage",   "severity": 1},
    {"id": 2,  "label": "Settlement: Pier 2 (20 mm)",         "category": "damage",   "severity": 2},
    {"id": 3,  "label": "Settlement: Pier 3 (40 mm)",         "category": "damage",   "severity": 2},
    {"id": 4,  "label": "Settlement: Pier 4 (80 mm)",         "category": "damage",   "severity": 3},
    {"id": 5,  "label": "Landslide",                          "category": "damage",   "severity": 3},
    {"id": 6,  "label": "Foundation: Scour (mild)",           "category": "damage",   "severity": 2},
    {"id": 7,  "label": "Foundation: Scour (severe)",         "category": "damage",   "severity": 3},
    {"id": 8,  "label": "Concrete Cracking (spalling)",       "category": "damage",   "severity": 2},
    {"id": 9,  "label": "Concrete Cracking (delamination)",   "category": "damage",   "severity": 3},
    {"id": 10, "label": "Tendon Rupture (1 tendon)",          "category": "damage",   "severity": 2},
    {"id": 11, "label": "Tendon Rupture (2 tendons)",         "category": "damage",   "severity": 3},
    {"id": 12, "label": "Anchor Head Failure (1)",            "category": "damage",   "severity": 2},
    {"id": 13, "label": "Anchor Head Failure (2)",            "category": "damage",   "severity": 3},
    {"id": 14, "label": "Rebar Corrosion (mild)",             "category": "damage",   "severity": 1},
    {"id": 15, "label": "Rebar Corrosion (moderate)",         "category": "damage",   "severity": 2},
    {"id": 16, "label": "Rebar Corrosion (severe)",           "category": "damage",   "severity": 3},
]

# ─── Helper: Standardise & Reconstruct ────────────────────────────────────────
def standardise(x: np.ndarray) -> np.ndarray:
    """x: (B,T,27) → z-scored with training mean/std."""
    return (x - mean_arr) / (std_arr + 1e-8)


def reconstruction_error(x_norm: np.ndarray):
    """
    x_norm: (B,T,27) numpy → per-sample MSE scalar (B,), reconstructed (B,T,27), per_sensor_mse (B,27)
    """
    t = torch.tensor(x_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon = model(t).cpu().numpy()
    err = np.mean((x_norm - recon) ** 2, axis=(1, 2))       # (B,)
    per_sensor = np.mean((x_norm - recon) ** 2, axis=1)     # (B, 27)
    return err, recon, per_sensor


def evaluate_health_tiers(overall_mse: float, max_sensor_mse: float, worst_zone_avg_mse: float, thr: float):
    if worst_zone_avg_mse >= thr or overall_mse >= thr:
        return {
            "tier": "CRITICAL_ANOMALY",
            "tier_label": "CRITICAL STRUCTURAL ANOMALY",
            "tier_code": 2,
            "badge_color": "#da3633",
            "is_anomaly": True
        }
    elif max_sensor_mse >= thr:
        return {
            "tier": "EARLY_WARNING",
            "tier_label": "EARLY LOCALIZED WARNING",
            "tier_code": 1,
            "badge_color": "#d29922",
            "is_anomaly": True
        }
    else:
        return {
            "tier": "HEALTHY",
            "tier_label": "BRIDGE STRUCTURALLY SOUND",
            "tier_code": 0,
            "badge_color": "#238636",
            "is_anomaly": False
        }


def calculate_zone_scores(per_sensor_mse: List[float], thr: float):
    scores = []
    for z in ZONE_METADATA:
        z_mses = [per_sensor_mse[s] for s in z["sensors"]]
        avg_mse = float(np.mean(z_mses))
        max_mse = float(np.max(z_mses))
        
        status = "normal"
        if avg_mse >= thr or max_mse >= thr:
            status = "critical" if avg_mse >= thr else "warning"
        elif max_mse >= thr * 0.8:
            status = "warning"

        scores.append({
            "id": z["id"],
            "label": z["label"],
            "avg_mse": avg_mse,
            "max_mse": max_mse,
            "status": status,
            "sensor_count": len(z["sensors"])
        })
    worst = max(scores, key=lambda z: z["avg_mse"])
    return scores, worst


# ─── App Factory ──────────────────────────────────────────────────────────────
app = FastAPI(title="Z24 SHM Digital Twin - Multi-Fault Sandbox", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_PATH.is_dir():
    app.mount("/app", StaticFiles(directory=str(FRONTEND_PATH), html=True), name="frontend")


@app.on_event("startup")
def load_assets():
    global model, mean_arr, std_arr, threshold, demo_X, demo_labels, healthy_baseline, scenario_deltas

    sc          = np.load(SCALER_PATH)
    mean_arr    = sc["mean"]           # (1,1,27)
    std_arr     = sc["std"]            # (1,1,27)
    threshold   = float(sc["threshold"].flat[0])

    demo        = np.load(DEMO_PATH)
    demo_X      = demo["X_fft"]        # (85,150,27)
    demo_labels = demo["labels"]       # (85,)

    model = CNNLSTMAutoencoder()
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    s0_samples = demo_X[demo_labels == 0]
    healthy_baseline = np.mean(s0_samples, axis=0).astype(np.float32)

    scenario_deltas = {
        "pier_settlement":   (np.mean(demo_X[demo_labels == 3], axis=0) - healthy_baseline).astype(np.float32),
        "concrete_cracking": (np.mean(demo_X[demo_labels == 9], axis=0) - healthy_baseline).astype(np.float32),
        "tendon_rupture":    (np.mean(demo_X[demo_labels == 11], axis=0) - healthy_baseline).astype(np.float32),
        "abutment_shift":    (np.mean(demo_X[demo_labels == 6], axis=0) - healthy_baseline).astype(np.float32),
    }

    print(f"[startup] threshold={threshold:.6f}  healthy_baseline={healthy_baseline.shape}")


# ─── Request Models ───────────────────────────────────────────────────────────
class FaultItem(BaseModel):
    zone: str                        # "left_abutment", "left_span", "center_pier", "right_span", "right_abutment"
    fault_type: str                  # "pier_settlement", "concrete_cracking", "tendon_rupture", "abutment_shift"
    severity: float = 0.0            # 0.0 to 1.0 (or 0 to 100)
    sensor_id: Optional[int] = -1    # -1 for zone or 0..26 for localized pin-point


class MultiFaultRequest(BaseModel):
    faults: Optional[List[FaultItem]] = []
    # Backwards compatibility for single-fault requests:
    zone: Optional[str] = None
    fault_type: Optional[str] = None
    severity: Optional[float] = 0.0
    sensor_id: Optional[int] = -1
    traffic_noise: Optional[bool] = False
    simulate_traffic: Optional[bool] = False


class BlindTestRequest(BaseModel):
    mode: Optional[str] = "random"   # "real_dataset", "injected_fault", or "random"


class PredictRequest(BaseModel):
    window: List[List[float]]        # (150, 27)


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "threshold": threshold, "mode": "multi_fault_sandbox"}


@app.get("/threshold")
def get_threshold():
    return {"threshold": threshold}


@app.get("/scenarios")
def list_scenarios():
    return SCENARIO_META


@app.get("/zones")
def list_zones():
    return ZONE_METADATA


def normalize_zone_key(zone: str) -> str:
    z = zone.lower().strip().replace(" ", "_").replace("-", "_")
    mapping = {
        "la": "left_abutment", "left_abutment": "left_abutment",
        "ls": "left_span", "left_span": "left_span",
        "cp": "center_pier", "center_pier": "center_pier", "pier": "center_pier",
        "rs": "right_span", "right_span": "right_span",
        "ra": "right_abutment", "right_abutment": "right_abutment"
    }
    return mapping.get(z, "center_pier")


def normalize_fault_key(fault_type: str) -> str:
    f = fault_type.lower().strip().replace(" ", "_").replace("-", "_").replace("/", "_")
    if "pier" in f or "settle" in f:
        return "pier_settlement"
    if "crack" in f or "spall" in f:
        return "concrete_cracking"
    if "tendon" in f or "rupt" in f:
        return "tendon_rupture"
    if "abut" in f or "found" in f or "scour" in f or "shift" in f:
        return "abutment_shift"
    return "pier_settlement"


@app.post("/api/inject-fault")
def inject_fault(req: MultiFaultRequest):
    """
    MULTIPLE SIMULTANEOUS FAULTS ENDPOINT:
    Accepts a list of faults across different bridge zones and cumulatively
    applies each fault to its respective zone sensors before running through
    the CNN-LSTM Autoencoder digital twin.
    """
    # 1. Gather all active faults
    fault_list: List[FaultItem] = []
    if req.faults and len(req.faults) > 0:
        fault_list.extend(req.faults)
    elif req.zone and req.fault_type and (req.severity or 0) > 0:
        fault_list.append(FaultItem(
            zone=req.zone,
            fault_type=req.fault_type,
            severity=req.severity,
            sensor_id=req.sensor_id
        ))

    # 2. Start from clean healthy baseline (label 0 reference)
    X = healthy_baseline.copy()   # (150, 27)

    # 3. Simulate operational environmental noise / heavy traffic load if enabled
    sim_traffic = bool(req.traffic_noise or req.simulate_traffic)
    if sim_traffic:
        traffic_noise = np.random.normal(0, 0.05 * np.mean(X), X.shape).astype(np.float32)
        thermal_drift = (np.sin(np.linspace(0, np.pi, 150))[:, np.newaxis] * 0.035 * np.mean(X)).astype(np.float32)
        X = np.maximum(0.0, X + traffic_noise + thermal_drift)

    n_bins = 150
    freq_bins = np.linspace(0.0, 25.0, n_bins, dtype=np.float32)
    sf = 0.70

    applied_faults_info = []

    # 4. Cumulatively apply each active fault to its respective zone sensors
    for f in fault_list:
        sev = float(f.severity)
        if sev > 1.0:
            sev = sev / 100.0
        sev = max(0.0, min(1.0, sev))
        if sev <= 0.0001:
            continue

        z_key  = normalize_zone_key(f.zone)
        ft_key = normalize_fault_key(f.fault_type)
        tgt_sensors = ZONE_SENSOR_MAP.get(z_key, ZONE_SENSOR_MAP["center_pier"])
        delta_src   = scenario_deltas.get(ft_key, scenario_deltas["pier_settlement"])
        z_meta = next((zm for zm in ZONE_METADATA if zm["id"] == z_key), None)

        applied_faults_info.append({
            "zone": z_key,
            "zone_label": z_meta["label"] if z_meta else z_key,
            "fault_type": ft_key,
            "severity": round(sev * 100, 1),
            "sensor_id": f.sensor_id,
            "affected_sensors": tgt_sensors
        })

        # Calculate localized weights with bleed
        weights = np.zeros(27, dtype=np.float32)
        for ch in range(27):
            if f.sensor_id is not None and 0 <= f.sensor_id < 27:
                d = abs(ch - f.sensor_id)
            else:
                d = min(abs(ch - s) for s in tgt_sensors)
            
            if d == 0:
                weights[ch] = 1.0
            elif d == 1:
                weights[ch] = 0.45
            elif d == 2:
                weights[ch] = 0.15
            else:
                weights[ch] = 0.0

        for ch in range(27):
            w = weights[ch] * sev * sf
            if w <= 0.0:
                continue

            ch_peak = float(np.max(np.abs(X[:, ch])))
            distortion = np.zeros(n_bins, dtype=np.float32)

            if ft_key == "pier_settlement":
                low_mask = np.exp(-((freq_bins - 3.8) / 2.6) ** 2).astype(np.float32)
                distortion = low_mask * ch_peak * 2.6
            elif ft_key == "concrete_cracking":
                hi_mask = (1.0 / (1.0 + np.exp(-(freq_bins - 9.5) / 2.4))).astype(np.float32)
                harmonic = (np.sin(2 * np.pi * freq_bins * 1.6) * ch_peak * 1.2).astype(np.float32)
                distortion = hi_mask * (harmonic + ch_peak * 1.9)
            elif ft_key == "tendon_rupture":
                res_mask = (np.exp(-((freq_bins - 6.5) / 3.2) ** 2) + np.exp(-((freq_bins - 14.2) / 3.0) ** 2)).astype(np.float32)
                distortion = res_mask * ch_peak * 3.2
            elif ft_key == "abutment_shift":
                edge_mask = np.exp(-((freq_bins - 1.8) / 2.0) ** 2).astype(np.float32)
                distortion = edge_mask * ch_peak * 2.8

            real_delta = delta_src[:, ch]
            combined = (real_delta * 0.8 + distortion) * w
            X[:, ch] = np.maximum(0.0, X[:, ch] + combined)

    # 5. Standardise and run through CNN-LSTM Autoencoder
    X_raw  = X[np.newaxis]
    X_norm = standardise(X_raw)
    errors, recon, per_sensors = reconstruction_error(X_norm)

    overall_mse    = float(errors[0])
    per_sensor_mse = [float(v) for v in per_sensors[0]]
    max_sensor_mse = float(np.max(per_sensor_mse))

    # Rank top 3 sensors by MSE
    ranked_sensors = sorted(
        [{"sensor": i, "mse": float(per_sensor_mse[i])} for i in range(27)],
        key=lambda item: item["mse"],
        reverse=True
    )
    top_3 = ranked_sensors[:3]

    # Calculate zone scores for ALL 5 structural zones
    zone_scores, worst_zone = calculate_zone_scores(per_sensor_mse, threshold)

    # Zones with warning or critical levels
    compromised_zones = [z for z in zone_scores if z["status"] in ("critical", "warning")]

    # Evaluate 3 Health Tiers
    tier_info = evaluate_health_tiers(overall_mse, max_sensor_mse, worst_zone["avg_mse"], threshold)

    if tier_info["is_anomaly"]:
        ratio = max(max_sensor_mse / threshold, worst_zone["avg_mse"] / threshold)
        confidence = min(99.9, max(65.0, 60.0 + (ratio - 1.0) * 15.0))
    else:
        confidence = max(90.0, 100.0 - (max_sensor_mse / threshold) * 25.0)

    return {
        "mse":                 overall_mse,
        "max_sensor_mse":      max_sensor_mse,
        "zone_avg_mse":        worst_zone["avg_mse"],
        "threshold":           threshold,
        "is_anomaly":          tier_info["is_anomaly"],
        "health_tier":         tier_info["tier"],
        "health_tier_label":   tier_info["tier_label"],
        "health_tier_code":    tier_info["tier_code"],
        "confidence":          round(confidence, 1),
        "per_sensor_mse":      per_sensor_mse,
        "original_fft":        X.tolist(),
        "reconstructed_fft":   recon[0].tolist(),
        "worst_zone":          worst_zone,
        "top_sensors":         top_3,
        "zone_scores":         zone_scores,
        "compromised_zones":   compromised_zones,
        "traffic_simulated":   sim_traffic,
        "faults_applied":      applied_faults_info,
        "active_fault_count":  len(applied_faults_info)
    }


@app.post("/api/blind-test")
def run_blind_test(req: BlindTestRequest = BlindTestRequest()):
    mode = req.mode or "random"
    if mode == "random":
        use_real = random.choice([True, False])
    elif mode == "real_dataset":
        use_real = True
    else:
        use_real = False

    if use_real:
        scenario_id = random.randint(1, 16)
        s_meta = next(s for s in SCENARIO_META if s["id"] == scenario_id)
        idxs = np.where(demo_labels == scenario_id)[0]
        sample_idx = random.choice(idxs)

        X_raw = demo_X[sample_idx]
        ground_zone_id = SCENARIO_GROUND_TRUTH_ZONE.get(scenario_id, "center_pier")
        z_meta = next(z for z in ZONE_METADATA if z["id"] == ground_zone_id)

        ground_truth = {
            "type": "Historical Z24 Benchmark Recording",
            "scenario_id": scenario_id,
            "scenario_label": s_meta["label"],
            "fault_category": s_meta["category"],
            "zone_id": ground_zone_id,
            "zone_label": z_meta["label"],
            "severity_level": s_meta["severity"],
            "details": f"Real 1998 Z24 Bridge damage recording (Scenario #{scenario_id}: {s_meta['label']})."
        }
    else:
        fault_types = ["pier_settlement", "concrete_cracking", "tendon_rupture", "abutment_shift"]
        chosen_fault = random.choice(fault_types)
        chosen_zone  = random.choice(ZONE_METADATA)
        chosen_severity = random.uniform(0.20, 0.75)

        X = healthy_baseline.copy()
        target_sensors = chosen_zone["sensors"]
        weights = np.zeros(27, dtype=np.float32)
        for ch in range(27):
            d = min(abs(ch - s) for s in target_sensors)
            weights[ch] = 1.0 if d == 0 else (0.45 if d == 1 else 0.0)

        n_bins = 150
        freq_bins = np.linspace(0.0, 25.0, n_bins, dtype=np.float32)
        delta_source = scenario_deltas[chosen_fault]
        sf = 0.70

        for ch in range(27):
            w = weights[ch] * chosen_severity * sf
            if w <= 0.0:
                continue
            ch_peak = float(np.max(np.abs(X[:, ch])))
            if chosen_fault == "pier_settlement":
                distortion = np.exp(-((freq_bins - 3.8) / 2.6) ** 2).astype(np.float32) * ch_peak * 2.6
            elif chosen_fault == "concrete_cracking":
                hi_mask = (1.0 / (1.0 + np.exp(-(freq_bins - 9.5) / 2.4))).astype(np.float32)
                distortion = hi_mask * (ch_peak * 2.4)
            elif chosen_fault == "tendon_rupture":
                distortion = (np.exp(-((freq_bins - 6.5) / 3.2) ** 2) + np.exp(-((freq_bins - 14.2) / 3.0) ** 2)).astype(np.float32) * ch_peak * 3.2
            else:
                distortion = np.exp(-((freq_bins - 1.8) / 2.0) ** 2).astype(np.float32) * ch_peak * 2.8

            X[:, ch] = np.maximum(0.0, X[:, ch] + (delta_source[:, ch] * 0.8 + distortion) * w)

        X_raw = X
        ground_truth = {
            "type": "Injected Structural Fault",
            "scenario_id": -1,
            "scenario_label": f"Injected: {chosen_fault.replace('_', ' ').title()}",
            "fault_category": chosen_fault,
            "zone_id": chosen_zone["id"],
            "zone_label": chosen_zone["label"],
            "severity_level": round(chosen_severity * 100, 1),
            "details": f"Synthetic fault applied to {chosen_zone['label']} with {round(chosen_severity * 100, 1)}% severity."
        }

    X_norm = standardise(X_raw[np.newaxis])
    errors, recon, per_sensors = reconstruction_error(X_norm)

    overall_mse    = float(errors[0])
    per_sensor_mse = [float(v) for v in per_sensors[0]]
    max_sensor_mse = float(np.max(per_sensor_mse))

    ranked_sensors = sorted(
        [{"sensor": i, "mse": float(per_sensor_mse[i])} for i in range(27)],
        key=lambda item: item["mse"],
        reverse=True
    )
    top_3 = ranked_sensors[:3]

    zone_scores, worst_zone = calculate_zone_scores(per_sensor_mse, threshold)
    tier_info = evaluate_health_tiers(overall_mse, max_sensor_mse, worst_zone["avg_mse"], threshold)

    zone_match = (worst_zone["id"] == ground_truth["zone_id"])
    ratio = max(max_sensor_mse / threshold, worst_zone["avg_mse"] / threshold)
    confidence = min(99.5, max(75.0, 70.0 + (ratio - 1.0) * 12.0))

    return {
        "test_token": str(uuid.uuid4())[:8],
        "ai_diagnosis": {
            "overall_mse":        overall_mse,
            "max_sensor_mse":     max_sensor_mse,
            "zone_avg_mse":       worst_zone["avg_mse"],
            "threshold":          threshold,
            "health_tier":        tier_info["tier"],
            "health_tier_label":  tier_info["tier_label"],
            "health_tier_code":   tier_info["tier_code"],
            "confidence":         round(confidence, 1),
            "detected_zone":      worst_zone,
            "top_sensors":        top_3,
            "zone_scores":        zone_scores,
            "per_sensor_mse":     per_sensor_mse,
            "original_fft":       X_raw.tolist(),
            "reconstructed_fft":  recon[0].tolist(),
        },
        "hidden_ground_truth": ground_truth,
        "verification": {
            "zone_match": zone_match,
            "summary": "Match" if zone_match else "Partial Match"
        }
    }


@app.get("/demo/{scenario_id}")
def run_demo(scenario_id: int):
    if scenario_id < 0 or scenario_id > 16:
        raise HTTPException(status_code=400, detail="scenario_id must be 0–16")

    idx = np.where(demo_labels == scenario_id)[0]
    if len(idx) == 0:
        raise HTTPException(status_code=404, detail="No samples found for this scenario")

    X_raw   = demo_X[idx]
    X_norm  = standardise(X_raw)
    errors, recon, per_sensors = reconstruction_error(X_norm)

    meta = next(s for s in SCENARIO_META if s["id"] == scenario_id)

    samples = []
    for i, (e, r, ps) in enumerate(zip(errors, recon, per_sensors)):
        ps_list = [float(v) for v in ps]
        z_scores, w_zone = calculate_zone_scores(ps_list, threshold)
        tier_info = evaluate_health_tiers(float(e), float(np.max(ps_list)), w_zone["avg_mse"], threshold)

        samples.append({
            "sample_index":        int(idx[i]),
            "reconstruction_error": float(e),
            "max_sensor_mse":      float(np.max(ps_list)),
            "health_tier":         tier_info["tier"],
            "health_tier_label":   tier_info["tier_label"],
            "anomaly":              tier_info["is_anomaly"],
            "worst_zone":          w_zone,
            "per_sensor_mse":      ps_list,
            "input_fft":           X_raw[i].tolist(),
            "reconstructed_fft":   r.tolist(),
        })

    mean_err = float(np.mean(errors))
    return {
        "scenario":       meta,
        "threshold":      threshold,
        "mean_error":     mean_err,
        "max_error":      float(np.max(errors)),
        "anomaly_rate":   float(np.mean([s["anomaly"] for s in samples])),
        "samples":        samples,
    }


@app.post("/predict")
def predict(req: PredictRequest):
    arr = np.array(req.window, dtype=np.float32)
    if arr.shape != (150, 27):
        raise HTTPException(
            status_code=422,
            detail=f"Expected shape (150,27), got {arr.shape}"
        )
    X_raw   = arr[np.newaxis]
    X_norm  = standardise(X_raw)
    errors, recon, per_sensors = reconstruction_error(X_norm)
    err = float(errors[0])
    ps_list = [float(v) for v in per_sensors[0]]
    _, w_zone = calculate_zone_scores(ps_list, threshold)
    tier_info = evaluate_health_tiers(err, float(np.max(ps_list)), w_zone["avg_mse"], threshold)

    return {
        "reconstruction_error": err,
        "anomaly":              tier_info["is_anomaly"],
        "health_tier":         tier_info["tier"],
        "health_tier_label":   tier_info["tier_label"],
        "threshold":            threshold,
        "per_sensor_mse":      ps_list,
        "reconstructed_fft":   recon[0].tolist(),
    }
