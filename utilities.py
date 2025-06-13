import numpy as np
from keras.losses import Loss
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras import Sequential
from tensorflow import reduce_sum, square, Variable

import matplotlib.pyplot as plt
import os
import math
import random
from typing import Optional

"""
Communication & Channel estimation
"""
def shannon_capacity_bps(snr_db:float, bandwidth_hz:float):
    snr = 10 ** (snr_db / 10)
    return bandwidth_hz * math.log2(1 + snr)

def add_ber_noise(weights: list[np.ndarray], ber: float) -> list[np.ndarray]:
    corrupted_weights = []
    affected_floats_total = 0
    total_floats_total = 0

    # each float32 has 32bit
    p_float = 1 - (1 - ber) ** 32   # the probability for a float32 being affected (has at least one bit get error)

    for w in weights:
        # float32 translation for debug
        w_float = w.astype(np.float32)
        total_floats = w_float.size
        total_floats_total += total_floats

        # binomial -> change this float or not change
        num_to_change = np.random.binomial(total_floats, p_float)

        w_flat = w_float.flatten()
        if num_to_change > 0:
            indices = np.random.choice(total_floats, size=num_to_change, replace=False)
            # rewrite the float to a new normal distribution float
            w_flat[indices] = np.random.randn(num_to_change).astype(np.float32)

        affected_floats_total += num_to_change

        corrupted_w = w_flat.reshape(w_float.shape)
        corrupted_weights.append(corrupted_w)

    print(f"Affected ratio: {affected_floats_total:5d}/{total_floats_total}, BER:{ber}")
    return corrupted_weights

def fspl_db(frequency_MHz, distance_km):
    fspl_db = 20 * math.log10(distance_km) + 20 * math.log10(frequency_MHz) + 32.44
    return fspl_db

def calc_snr_ber(fspl_db, bandwidth_hz: float, p_tx_dbm: float, g_tx_dbi: float, g_rx_dbi: float, noise_figure_db: float, system_loss_db: float):
    thermal_noise_dbm = -174 + 10 * math.log10(bandwidth_hz) + noise_figure_db    # Thermal Noise
    snr_db = p_tx_dbm + g_tx_dbi + g_rx_dbi - fspl_db - system_loss_db - thermal_noise_dbm
    snr = 10 ** (snr_db / 10)   # dB to linear
    ber = 0.5 * math.erfc(math.sqrt(snr))   #BPSK
    return snr_db, ber

def calculate_model_size_MB(weights)->float:
    total_bytes = sum(w.nbytes for w in weights)
    total_MB = total_bytes / (1024 ** 2)
    return total_MB

"""
Physic & orbit calculation
"""

def norm_3d_vector(vec:tuple):
    length = math.sqrt(vec[0]**2 + vec[1]**2 + vec[2]**2)
    normalized_vec = (vec[0]/length, vec[1]/length, vec[2]/length)
    return normalized_vec

def rodrigues_rotation(v, axis, angle):    
    #v_rot = v*cos(angle) + (axis x v)*sin(angle) + axis*(axis dot v)*(1 - cos(angle))

    cos_ang = math.cos(angle)
    sin_ang = math.sin(angle)
    # term1: v*cos(angle)
    term1 = v * cos_ang
    # term2: (axis x v)*sin(angle)
    cross = np.cross(axis, v)
    term2 = cross * sin_ang
    # term3: axis*(axis dot v)*(1 - cos(angle))
    dot = axis[0]*v[0] + axis[1]*v[1] + axis[2]*v[2]
    term3 = np.array([axis[0]*dot*(1 - cos_ang), axis[1]*dot*(1 - cos_ang), axis[2]*dot*(1 - cos_ang)])
    return term1 + term2 + term3

def check_position(entity_name, pos, R_earth):
    """
    if the length of pos < radius of earth, there's a problem, print the error message
    """
    length = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    if length < R_earth:
        print(f"Error: {entity_name} is inside the Earth: norm = {length:.6f} km")

def generate_random_unit_vector():
    """
    generate a random 3-d unit vector by transfing sphere coordinates to (x,y,z) coordinates, x>0, y>0, z>0
    """
    theta = random.uniform(0, math.pi/4) 
    phi = random.uniform(0, math.pi/4)
    x = math.sin(phi) * math.cos(theta)
    y = math.sin(phi) * math.sin(theta)
    z = math.cos(phi)
    return np.array([x, y, z])


"""
ML Models & algorithms
"""
def random_non_iid(x_train, y_train, max_num_class:int, min_num_class:int):
    
    num_class = np.random.randint(max_num_class, min_num_class)
    labels = np.unique(y_train)
    label_subset = np.random.choice(labels, num_class, replace=False)
    indices = np.isin(y_train.flatten(), label_subset)
    client_x, client_y = x_train[indices], y_train[indices]
    client_x, client_y = np.asarray(client_x), np.asarray(client_y)

    # Shuffle
    shuffle_indices = np.random.permutation(len(client_x))
    client_x, client_y = client_x[shuffle_indices], client_y[shuffle_indices]

    return client_x, client_y

"""
hap_weigths of DNN
<class 'list'> with elements:
(784, 64)
(64,)
(64, 32)
(32,)
(32, 10)
(10,)
"""
def build_dnn_model_mnist(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_dnn_model_cifar10(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_cnn_model_cifar10(input_shape=(32, 32, 3), num_classes=10):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    return model

class FedproxLoss(Loss):
    def __init__(self, global_weights, local_weights, base_loss_fn, mu=0.01, **kwargs):
        super(FedproxLoss, self).__init__(**kwargs)
        self.global_weights = [Variable(w, trainable=False) for w in global_weights]
        self.local_weights = [Variable(w, trainable=False) for w in local_weights]
        self.base_loss_fn = base_loss_fn
        self.mu = mu

    def call(self, y_true, y_pred):
        base_loss = self.base_loss_fn(y_true, y_pred)
        prox_loss = 0.0
        for g_w, l_w in zip(self.global_weights, self.local_weights):
            prox_loss += reduce_sum(square(l_w - g_w))
        prox_loss *= self.mu / 2.0
        return base_loss + prox_loss

    # updating the weights in this class
    # assign is provided by keras
    def update_weights(self, gw, lw):
        for var, new_w in zip(self.global_weights, gw):
            var.assign(new_w)
        for var, new_w in zip(self.local_weights, lw):
            var.assign(new_w)

"""
Plot & Data collection
"""
from collections import defaultdict
from typing import Dict, List
import plotly.graph_objects as go
import matplotlib.ticker as ticker 

# Plotting, Logs
class TrainingStats:
    def __init__(self, folder_path: str, save_period: int = 0):
        self.data: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
        self.folder_path = folder_path
        self.save_period = save_period
        os.makedirs(folder_path, exist_ok=True)

    def log(self, hap_id: int, label: str, value: float, title: Optional[str] = None):
        self.data[label][hap_id].append(value)
        if self.save_period != 0 and len(self.data[label][hap_id]) % self.save_period == 0:
            self._save_series(label, title)

    def plot_all(self):
        for label in self.data.keys():
            self._save_series(label)

    def _save_series(self, label: str, title: Optional[str] = None):
        if title is None:
            title = " ".join(piece.capitalize() for piece in label.replace("/", " ").split("_"))
        ylabel = "Accuracy" if "acc" in label else "Loss"

        plt.figure()
        csv_lines = ["Round," + ",".join(f"HAP {hap_id}" for hap_id in sorted(self.data[label]))]

        max_len = max(len(vlist) for vlist in self.data[label].values())
        for i in range(max_len):
            row = [str(i)]
            for hap_id in sorted(self.data[label]):
                hap_data = self.data[label][hap_id]
                row.append(str(hap_data[i]) if i < len(hap_data) else "")
            csv_lines.append(",".join(row))

        for hap_id, vlist in self.data[label].items():
            plt.plot(vlist, label=f"HAP {hap_id}")
        plt.xlabel("Round")
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True)

        fname = label.replace("/", "_")
        plt.savefig(os.path.join(self.folder_path, f"{fname}.png"))
        plt.close()

        # CSV file
        csv_path = os.path.join(self.folder_path, f"{fname}.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(csv_lines))

    def compare(self, label_a: str, label_b: str, hap_ids: Optional[list[int]] = None, mode = "overlay", title: Optional[str] = None):
        if label_a not in self.data or label_b not in self.data:
            missing = [l for l in (label_a, label_b) if l not in self.data]
            raise ValueError(f"Label(s) not found in stats: {missing}")

        all_haps = set(self.data[label_a].keys()) | set(self.data[label_b].keys())
        hap_ids = hap_ids or sorted(all_haps)

        if title is None:
            title = f"{label_a}  vs  {label_b}"
        ylabel = (
            "Accuracy"
            if any("acc" in l for l in (label_a, label_b))
            else "Loss"
        )

        # Plot
        if mode == "overlay":
            plt.figure()
            for hap_id in hap_ids:
                if hap_id in self.data[label_a]:
                    plt.plot(
                        self.data[label_a][hap_id],
                        linestyle="-",
                        label=f"{label_a} | HAP {hap_id}",
                    )
                if hap_id in self.data[label_b]:
                    plt.plot(
                        self.data[label_b][hap_id],
                        linestyle="--",
                        label=f"{label_b} | HAP {hap_id}",
                    )
            plt.xlabel("Round");  plt.ylabel(ylabel);  plt.title(title)
            plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
            plt.legend();  plt.grid(True)

        elif mode == "side":
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
            for ax, (lab, style) in zip(
                axes, ((label_a, "-"), (label_b, "--"))
            ):
                for hap_id in hap_ids:
                    if hap_id in self.data[lab]:
                        ax.plot(
                            self.data[lab][hap_id],
                            linestyle=style,
                            label=f"HAP {hap_id}",
                        )
                ax.set_title(lab)
                ax.set_xlabel("Round")
                plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                ax.grid(True)
            axes[0].set_ylabel(ylabel)
            axes[1].legend(loc="upper right")
            fig.suptitle(title)
        else:
            raise ValueError("mode must be 'overlay' or 'side'")
        
        fname = f"compare_{label_a.replace('/','_')}_vs_{label_b.replace('/','_')}.png"
        plt.savefig(os.path.join(self.folder_path, fname))
        plt.close()

    def plot_system(self, sat_positions:np.ndarray, hap_positions:np.ndarray, sat_trajectories:list[np.ndarray]=None, filename="3d_plot.html"):
        """
        sat_positions      : np.ndarray (N, 3)
        hap_positions      : np.ndarray (M, 3)
        sat_trajectories   : List[np.ndarray (T,3)]
        """
        fig = go.Figure()

        # SAT
        fig.add_trace(go.Scatter3d(
            x=sat_positions[:, 0], y=sat_positions[:, 1], z=sat_positions[:, 2],
            mode='markers',
            marker=dict(size=5, color='red', symbol='circle'),
            name='SATs'
        ))

        # HAP
        fig.add_trace(go.Scatter3d(
            x=hap_positions[:, 0], y=hap_positions[:, 1], z=hap_positions[:, 2],
            mode='markers',
            marker=dict(size=5, color='blue', symbol='square'),
            name='HAPs'
        ))

        # show orbits of SATs
        if sat_trajectories:
            for i, traj in enumerate(sat_trajectories):
                fig.add_trace(go.Scatter3d(
                    x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                    mode='lines',
                    line=dict(width=2, color='orange'),
                    name=f"Orbit {i}"
                ))

        # show Earth
        R = 6371  # km
        u, v = np.mgrid[0:2*np.pi:50j, 0:np.pi:25j]
        x = R * np.cos(u) * np.sin(v)
        y = R * np.sin(u) * np.sin(v)
        z = R * np.cos(v)
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            opacity=0.3,
            colorscale='Blues',
            showscale=False,
            hoverinfo='skip',
            name='Earth'
        ))

        fig.update_layout(
            title='3D HAP-SAT with Orbit Paths',
            scene=dict(
                xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                aspectmode='data',
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            legend=dict(x=0.02, y=0.98)
        )

        full_path = os.path.join(self.folder_path, filename)
        fig.write_html(full_path)
        print(f"[INFO] 3D interactive plot with orbits saved to {full_path}")

    @staticmethod
    def plot_hap_sat(haps: list, sats: list, hap_mask: Optional[list[int]] = None, sat_mask: Optional[list[int]] = None,
                     show_earth:bool=True, show_trajectory:bool=True):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        num_groups = len(haps)
        colors = plt.cm.tab10(np.linspace(0, 1, num_groups))

        # origin and axes
        ax.scatter(0, 0, 0, marker='*', s=200, color='black', label='Origin')
        axis_len = 1.25 * (6371.0 + 1000.0)
        ax.quiver(-axis_len, 0, 0, 2*axis_len, 0, 0, arrow_length_ratio=0.04, color="black")
        ax.quiver(0, -axis_len, 0, 0, 2*axis_len, 0, arrow_length_ratio=0.04, color="black")
        ax.quiver(0, 0, -axis_len, 0, 0, 2*axis_len, arrow_length_ratio=0.04, color="black")

        if show_earth:
            r = 6371
            n_lines = 12
            n_points = 100
            theta = np.linspace(0, 2*np.pi, n_points)

            ax.plot(r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta), color='gray')
            for phi in np.linspace(0, 2*np.pi, n_lines):
                x = r * np.cos(theta) * np.cos(phi)
                y = r * np.cos(theta) * np.sin(phi)
                z = r * np.sin(theta)
                ax.plot(x, y, z, color='gray', linewidth=1.5)
            for theta_ in np.linspace(0, np.pi, n_lines):
                phi = np.linspace(0, 2*np.pi, n_points)
                x = r * np.sin(theta_) * np.cos(phi)
                y = r * np.sin(theta_) * np.sin(phi)
                z = r * np.cos(theta_) * np.ones_like(phi)
                ax.plot(x, y, z, color='gray', linewidth=1.5)

        for i, hap in enumerate(haps):
            if hap_mask is not None and hap.hap_id not in hap_mask:
                continue
            color = colors[i % len(colors)]
            hap_x, hap_y, hap_z = hap.pos
            ax.scatter(hap_x, hap_y, hap_z, marker='^', s=120, color=color, label=f'HAP {hap.hap_id}')

            for sat in sats:
                if sat_mask is not None and sat.sat_id not in sat_mask:
                    continue
                assigned_hap_id = sat.sat_id // len(haps)
                if assigned_hap_id != hap.hap_id:
                    continue
                sat_x, sat_y, sat_z = sat.pos
                ax.scatter(sat_x, sat_y, sat_z, marker='o', s=60, color=color)
                ax.plot([hap_x, sat_x], [hap_y, sat_y], [hap_z, sat_z], color=color, linestyle='--', linewidth=1)
                if show_trajectory and hasattr(sat, "trajectories"):
                    traj = sat.trajectories
                    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=1.5, alpha=0.7)

        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        ax.set_zlabel("Z (km)")
        ax.set_title("3D Positions of HAPs-SATs system")

        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc='upper right')

        plt.show()
        plt.close()

