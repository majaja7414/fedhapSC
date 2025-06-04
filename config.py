from dataclasses import dataclass
from math import pi, sqrt

@dataclass
class SimConfig:
    # System
    num_haps: int = 5
    sats_per_hap: int = 3
    rounds: int = 10
    epochs: int = 10
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

import tensorflow as tf

def config_gpu_memory_limit(memory_limit_mb=3000):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)]
            )
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(f"[GPU] {len(gpus)} Physical, {len(logical_gpus)} Logical (limit {memory_limit_mb} MB)")
        except RuntimeError as e:
            print(f"[GPU config error] {e}")
