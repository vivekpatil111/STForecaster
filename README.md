
# STForecaster 🌫️
### Physics-Informed Spatiotemporal Deep Learning for Extreme PM2.5 Forecasting over India

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

</div>

---

> **Given 10 hours of past atmospheric data across India's 140×124 spatial grid → predicts PM2.5 concentrations for the next 16 hours.**
> Achieves **>3000× speedup** over WRF-Chem numerical solvers with less than 20% SMAPE degradation.

---

## 📌 About

Air pollution is one of India's most critical public health challenges. PM2.5 — fine particulate matter — causes millions of premature deaths annually. Accurate short-term forecasting is essential for issuing health advisories and informing policy decisions.

Traditional numerical solvers like WRF-Chem take **45–90 minutes per forecast case**. STForecaster does it in **~0.8 seconds** — a >3000× speedup — while maintaining strong forecasting accuracy.

This is a personal research project exploring the intersection of **atmospheric physics** and **deep learning** for real-world air quality forecasting at national scale.

---

## ❌ Why Standard Deep Learning Fails Here

| Physics Problem | Standard DL Failure | STForecaster's Fix |
|---|---|---|
| Spatiotemporal Transport | CNNs are spatially memoryless, fail to capture advection | 3-Layer ConvLSTM tracking 10-hour advection history |
| Extreme Events | Plain MSE ignores PM2.5 spikes (episodes = only 5.6% of data) | Episode-Aware Loss via STL Decomposition |
| Emission Scaling | Log-normal distributions span >10 orders of magnitude | Hard physics constraints: log1p + 1e9 scaling |

---

## 🧠 Model Architecture — STForecaster (10.01M Parameters)

```
Input  (B × 21 × 10 × 140 × 124)
           │
           ▼
  ┌─────────────────────┐
  │  Temporal Positional │  Learnable 5D embeddings
  │  Embedding           │  encode hour-of-sequence position
  └─────────────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  Batched CNN Encoder │  B*T frames in one GPU call (no loop)
  │  1/2 & 1/4 skips    │  ~10× faster than sequential
  └─────────────────────┘
           │
           ▼
  ┌──────────────────────────┐
  │  3-Layer ConvLSTM         │  192 hidden dims
  │  Temporal Backbone        │  Captures pollutant advection
  └──────────────────────────┘
           │
           ▼
  ┌──────────────────────────┐
  │  Dual Attention           │
  │  ├─ Temporal-Spatial     │  1×1 conv scorer over 10 hidden states
  │  └─ Temporal-Channel     │  SE-style squeeze-excite
  └──────────────────────────┘
           │
           ▼
  ┌─────────────────────┐
  │  U-Net Decoder       │  Upsamples back to 140×124
  │  Last-frame skip     │  Uses last input frame (NOT mean)
  └─────────────────────┘
           │
           ▼
Output (B × 16 × 140 × 124) — 16-hour PM2.5 forecast
```

### Key Architecture Decisions

| Decision | Why |
|---|---|
| Batched Encoder (B*T) | All timesteps in one GPU call — ~10× faster than loop |
| Last-frame skip connections | Preserves most recent spatial state (mean dilutes recent info) |
| 3-Layer ConvLSTM | Recurrent memory is essential — removing it degrades error by +68% |
| Dual Attention | Captures both "when" and "which feature" matters |
| 5D Positional Embeddings | Boundary-layer evolution context across 10-hour window |

---

## 📦 Features

### Base Features (16)

| Category | Features |
|---|---|
| Pollution | cpm25 |
| Meteorology | u10, v10, rain, t2, pblh, psfc, swdown, q2 |
| Emissions | SO2, NH3, NOx, PM25, bio, NMVOC_finn, NMVOC_e |

### Derived Features (5) — Physics-Motivated

| Feature | Formula | Why |
|---|---|---|
| wind_speed | √(u10² + v10²) | Magnitude more useful than directional components |
| vent_index | wind_speed × pblh | Captures atmospheric dispersion capacity |
| delta_cpm25 | cpm25[t] - cpm25[t-1] | Rate of change — is pollution rising or falling? |
| lat | min-max normalized | Anchors predictions to geographic terrain |
| lon | min-max normalized | Anchors predictions to geographic terrain |

**Final input tensor shape:** `(Batch, 21, 10, 140, 124)`

### Log1p Transform — Applied to 9 Features

```
cpm25, SO2, NH3, NOx, PM25, bio, NMVOC_finn, NMVOC_e, rain
```

QQ-plot R² values confirm log-normal distributions (all far from Gaussian):
```
SO2=0.023 | NOx=0.042 | PM25=0.052 | NMVOC_finn=0.009
bio=0.274  | NH3=0.319 | NMVOC_e=0.184 | rain=0.013
```

---

## 🎭 Episode Detection — STL Decomposition

Pollution episodes represent only **5.6% of all pixels** but are the most critical for public health. Standard loss functions ignore them.

```python
# Step 1: Remove trend (24hr moving average)
trend = uniform_filter1d(data, size=24)

# Step 2: Remove daily seasonal pattern
seasonal = hourly_average[t % 24]

# Step 3: Residual anomaly
residual = data - trend - seasonal

# Step 4: Mark as episode if residual > μ + 1.5σ
episode_mask = (residual > res_mean + 1.5 * res_std)
```

Episode pixels are **upweighted 7× in the loss function**.

---

## 📉 Loss Function

$$L = \text{RMSE}_{\log z} + 0.30 \cdot \text{SMAPE} + 0.15 \cdot (1 - \rho) + 7.0 \times \text{MSE}_{\text{ep}}$$

where $\rho$ is Pearson correlation and episode pixels are upweighted **7×**.

### Loss Terms

| Term | Weight | Purpose |
|---|---|---|
| $\text{RMSE}_{\log z}$ | 1.0 | Primary reconstruction in log space |
| $\text{SMAPE}$ | 0.30 | Scale-invariant % error |
| $1 - \rho$ | 0.15 | Preserves spatial pollution patterns |
| $\text{MSE}_{\text{ep}}$ | 7.0× | Upweights extreme pollution events |

--

## 📊 Results

### Model Comparison

| Model | Global RMSE (μg/m³) | Global SMAPE | Episode SMAPE | Episode Corr (ρ) |
|---|---|---|---|---|
| Persistence Baseline | ~38–45 | 0.42 | 0.51 | 0.72 |
| Plain ConvLSTM | 22.1 | 0.261 | 0.318 | 0.891 |
| **STForecaster (Ours)** | **~16.5** | **~0.200** | **~0.224** | **~0.938** |

### Computational Speedup

| System | Time per Case |
|---|---|
| WRF-Chem Numerical Solver | ~45–90 minutes |
| **STForecaster (Ours)** | **~0.8 seconds** |
| **Speedup** | **>3,000×** |

---

## 🔬 Ablation Study

| Removed Component | Episode SMAPE ↑ | Insight |
|---|---|---|
| Full Model (baseline) | 0% | — |
| Remove Temporal-Channel Attention | +12% | Moderate contribution |
| Remove Episode Boost Loss | +34% | Episode-blind networks fail on extremes |
| Replace log1p with raw normalisation | +52% | Invalidates log-normal physics of emissions |
| Replace ConvLSTM with mean-pooling | **+68%** | **Atmospheric transport requires recurrent memory** |

---

## 🌍 What the Model Learned

The attention weights reveal the model autonomously discovered real atmospheric physics:

**1. Biogenic Dominance (3.1×)**
During July monsoons, the model assigns 3.1× higher attention to biogenic/NMVOC channels — accurately emulating SOA formation when anthropogenic transport is suppressed by rainfall.

**2. Spatial Teleconnection: Punjab → Bangladesh**
Attention networks discovered non-local correlations at +14–16 hour lead times, perfectly matching westerly jet transport mechanics — without being explicitly told.

**3. Tendency Shift over Forecast Horizon**
For hours 1–4, model relies on recent PM2.5 tendencies. For longer horizons, it correctly shifts attention to fundamental meteorological drivers (wind, PBLH, temperature).

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Epochs | 55 |
| Learning Rate | 3×10⁻⁴ |
| Batch Size | 12 |
| Hidden Dim | 192 |
| Warmup Epochs | 4 |
| Early Stopping Patience | 22 |
| Weight Decay | 1×10⁻³ |
| Val Fraction | 15% |
| Temporal Buffer | 26 steps |
| Episode MSE Boost | 7.0× |
| LR Scheduler | CosineAnnealingLR |
| Ensemble Seeds | [42, 137] |

---

## 🚀 How to Run

### Requirements

```bash
pip install torch numpy scipy
```

### Data Setup

The model uses WRF-Chem atmospheric simulation data.
Update paths in the CONFIG block of the notebook:

```python
BASE_PATH = "/path/to/raw"      # folder with APRIL_16/, JULY_16/, etc.
TEST_PATH = "/path/to/test_in"  # folder with test .npy files
WORK_DIR  = "/path/to/outputs"  # output directory
```

### Training

Open `STForecaster.ipynb` and run all cells.
Training takes approximately **2–3 hours** on a single GPU.

### Config to Tune

```python
ENSEMBLE_SEEDS    = [42, 137]  # add more seeds for better ensemble
HIDDEN_DIM        = 192        # increase for more capacity (watch VRAM)
EPISODE_MSE_BOOST = 7.0        # raise if episode SMAPE is high
EPOCHS            = 55         # increase with more compute
```

---

## ⚠️ Known Limitations

- **December winter fog:** Deep winter fog causes boundary-layer collapse — model shows residual prediction bias
- **Autoregressive error accumulation:** Errors compound over the 16-hour horizon, especially beyond hour 12
- **Single-year training:** Only 2016 data used — interannual variability (ENSO) is unrepresented

---

## 🔭 Future Work

- Scale to 2015–2022 dataset for interannual ENSO robustness
- Implement Graph Neural Networks on WRF mesh for coastal/orographic boundaries (Western Ghats)
- Deploy as a real-time FastAPI microservice for sub-hourly AQI nowcasting within India's National Clean Air Programme (NCAP)
- Probabilistic forecasting with calibrated uncertainty envelopes

---

## 📄 Paper

> 📝 **Coming Soon** — arXiv preprint under preparation

*STForecaster: Physics-Informed Spatiotemporal Deep Learning for Extreme PM2.5 Forecasting over India*
Vivek N Patil | NMIMS MPSTME Shirpur, Maharashtra

---

## 🗂️ Repository Structure

```
STForecaster/
├── README.md
├── STForecaster.ipynb      # Main training + inference notebook
├── paper/
│   └── stforecaster.pdf    # Coming soon
├── assets/
│   ├── architecture.png
│   ├── wind_field.png
│   ├── ablation.png
│   └── seasonal_maps.png
└── outputs/
    └── .gitkeep
```

---

## 👤 Author

**Vivek N Patil**
SVKM's Narsee Monjee Institute of Management Studies (NMIMS)
Mukesh Patel School of Technology Management and Engineering (MPSTME)
Shirpur, Maharashtra, India

📧 vivek.patil022@nmims.in

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
Built with ☕ and a lot of NaN loss debugging.
</div>
