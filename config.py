from dataclasses import dataclass
from math import pi, sqrt

@dataclass
class SimConfig:
    # System
    num_haps: int = 5
    sats_per_hap: int = 3
    rounds: int = 10
    epochs: int = 5
    step_length: int = 1
    aggregation_dist_constant: float = 2.5e3
    log_dir: str = "exp_log"

    # Orbital parameters
    re: float = 6371
    gm: float = 3.986004418e5
    hap_altitude: float = 20.0
    sat_alt_1: float = 500.0
    sat_alt_2: float = 1000.0
    max_sat_alt: float = sat_alt_2
    max_sat_orbital_radius:float = max_sat_alt + re
    min_angular_velocity:float = sqrt(gm / max_sat_orbital_radius**3)
    total_steps: int = int((2*pi) / min_angular_velocity)
    earth_omega: float = 7.2921159e-5
    min_visible_angle: float = 0.17453293  # cos(80°)

    # Model & Dataset
    model_type: str = "dnn"      # "dnn" or "cnn"
    dataset_type: str = "mnist"  # "mnist" or "cifar10"
    non_iid_number_max: int = 3
    non_iid_number_min: int = 8
    lr:float = 0.001
    fedprox_mu:float = 0.01

    # Communication system
    snr_db: float = 15
    ber: float = 1e-5
    ka_b_mhz: float = 31000
    ka_c_mhz: float = 21200
    bandwidth_hz: float = 1e6
    fspl_max_db: float = 187

    sat_tx_power_dbm: float = 30.0
    hap_tx_power_dbm: float = 30.0
    sat_tx_gain_dbi: float = 30.0
    hap_tx_gain_dbi: float = 30.0
    sat_rx_gain_dbi: float = 30.0
    hap_rx_gain_dbi: float = 30.0
    noise_figure_db: float = 3.0
    system_loss_db: float = 2.0

    # others
    optional_vram_limit: int = 4096 # MB

import os
def log_config(cfg: SimConfig, folder_path: str):
    config_dict = vars(cfg)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "exp_config.txt")

    with open(file_path, "w") as f:
        for key, value in config_dict.items():
            f.write(f"{key}: {value}\n")