# Federated Learning on High-Altitude Platforms and LEO Satellites — FSPL-Based Transmission and Aggregation Strategies

## Abstract

In satellite-enabled federated learning, communication constraints often dominate system performance due to limited bandwidth, intermittent visibility, and signal degradation. This project presents a simulation framework for federated learning architectures combining High-Altitude Platforms (HAPs) and Low Earth Orbit (LEO) satellites, where Free Space Path Loss (FSPL) serves as the primary factor impacting transmission. The framework models dynamic HAP-SAT visibility, channel noise, limited capacity links, and orbital motion via physical rotation models. Two aggregation schemes are implemented: (1) standard FedAvg aggregation (`baseline`) and (2) distance-weighted aggregation (`mutual`) that incorporates inter-HAP physical proximity to improve model convergence under heterogeneous communication conditions. The simulator also supports non-IID data distributions across nodes, FedProx-based loss functions, and full parameter customization for orbital, communication, and model hyperparameters. This platform is designed for researchers investigating practical federated learning deployment in aerial and space networks.

---

## Getting Started

### Environment Setup

```bash
Required packages: tensorflow (with keras), matplotlib, plotly
Tested with: tensorflow 2.10.0 + cuda 11.2 + python 3.10
Older versions (e.g. tensorflow + python 3.9) are also supported depending on your GPU and OS compatibility.
```

### Running the Simulation

#### Option 1 — Run with default parameters:
```bash
python main.py
```

#### Option 2 — Run with custom parameters:
```bash
python main.py --aggregation [baseline|mutual] --device [auto|cpu|gpu-divide] --model [dnn|cnn] --dataset [mnist|cifar10] --rounds [int] --epochs [int]
```

**Parameter Descriptions:**

- `--aggregation`:  
  - `baseline`: Standard FedAvg aggregation  
  - `mutual`: Distance-weighted aggregation for improved convergence
- `--device`:  
  - `auto`: Let TensorFlow select device automatically  
  - `cpu`: Force CPU usage  
  - `gpu-divide`: Use GPU with VRAM limit (default 4096MB, adjustable in `config.py`) — useful for avoiding VRAM overflow while multitasking.
- `--model`: Select model type: `dnn` or `cnn`
- `--dataset`: Select dataset: `mnist` or `cifar10`
- `--rounds`: Number of global aggregation rounds (i.e. satellite orbital periods); default = 10
- `--epochs`: Number of local training epochs per round; default = 5

**Example:**
```bash
python main.py --aggregation mutual --device auto --model dnn --dataset mnist --rounds 10 --epochs 10
```

Simulation outputs include test accuracy, loss, and parameter logs, automatically saved under the `exp_log/` directory with timestamped folders.  
The `exp_log_example` directory contains historical experiment results. Subfolders such as `tx20` indicate antenna transmission power (20 dBm).

---

## Project Structure

| File         | Description |
|--------------|-------------|
| `main.py`    | Main simulation loop: FL orchestration, training, aggregation |
| `utilities.py` | Helper functions: channel modeling, model construction, logging, etc. |
| `config.py`  | Global configuration for simulation: nodes, channel, hyperparameters |

---

## Code Overview

### `config.py`

Defines simulation parameters via the `SimConfig` class:

- **Communication Parameters:** `fspl_max_db`, `snr_db`, `bandwidth_hz`, `tx_power_dbm`, etc.
- **Training Parameters:** `rounds`, `epochs`, `model_type`, `fedprox_mu`
- **Orbital Parameters:** `hap_altitude`, `sat_alt_1`, `re`, `earth_omega`, etc.

---

### `utilities.py`

#### Channel Simulation

- `fspl_db`: Compute Free Space Path Loss (FSPL)
- `calc_snr_ber`: Compute SNR and BER for the channel
- `add_ber_noise`: Simulate model weight degradation based on BER
- `shannon_capacity_bps`: Compute Shannon channel capacity (bps)

#### Orbit & Physics Calculations

- `generate_random_unit_vector`: Generate random node positions (HAP, SAT)
- `rodrigues_rotation`: Simulate Earth rotation and satellite motion using Rodrigues' rotation formula
- `check_position`: Check if nodes fall inside Earth’s surface

#### Model Construction

- `build_dnn_model_mnist/cifar10`: Build DNN models
- `build_cnn_model_cifar10`: Build CNN model
- `random_non_iid`: Generate non-IID data distribution for clients
- `FedproxLoss`: Custom FedProx loss function

#### Visualization

- `TrainingStats`: Log and visualize training metrics (accuracy/loss), plot 3D system topology

---

### `main.py`

#### Class: `FedHAP`

- `__init__()`: Initialize HAP/SAT nodes, load models and datasets
- `run_system()`: Main simulation loop
- `pre_simulate()`: Precompute SAT-HAP visibility, distance, FSPL tables
- `integrate_sat_event()`: Schedule upload/download events

#### Class: `HAP`

- `receive_sat_weights()`, `aggregate_models()`: Receive and aggregate models from SAT
- `check_aggregate_condition()`: Monitor aggregation conditions (currently unconditional)

#### Class: `SAT`

- `plan_greedy_roundtrip()`: Greedy scheduling of satellite transmissions
- `receive_hap_weight()`: Receive models from HAPs
- `get_visible_windows_list()`: Convert visibility table into time intervals

---

## Aggregation Strategies

| Mode      | Description |
|-----------|-------------|
| `baseline` | Equal-weight averaging across all HAP nodes (FedAvg) |
| `mutual`   | Distance-weighted aggregation; more stable but slightly slower convergence. You may tune `fedprox_mu` in `SimConfig` (`config.py`) to adjust convergence behavior. |

---

## Output Files

- Each run generates a folder containing all CSV logs and visualizations.
- Output includes per-round accuracy/loss for each HAP and a 3D topology visualization (HTML).

---