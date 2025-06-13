import numpy as np

from keras.datasets import mnist, cifar10
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

from config import SimConfig, log_config
from utilities import add_ber_noise, build_dnn_model_mnist, build_cnn_model_cifar10, random_non_iid, FedproxLoss
from utilities import rodrigues_rotation, generate_random_unit_vector, check_position
from utilities import shannon_capacity_bps, fspl_db, calc_snr_ber, calculate_model_size_MB
from utilities import TrainingStats

import math
from typing import cast
from datetime import datetime

cfg = SimConfig()

class FedHAP:
    def __init__(self, data:list[tuple], folder_path:str, build_model_fn):
        self.data = data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.folder_path = folder_path

        # Building the model for ML
        # Using the shared model to save RAM, for every training, using this model instance, only passing the weight
        self.center_model = build_model_fn()
        global_weights = self.center_model.get_weights()
        size_MB = calculate_model_size_MB(global_weights)
        self.model_size_bits = size_MB * 8 * 1024 * 1024

        base_loss_fn = SparseCategoricalCrossentropy()  # base loss function in FedProx
        self.fedprox_loss_instance = FedproxLoss(global_weights, global_weights, base_loss_fn, mu=cfg.fedprox_mu)
        self.center_model.compile(optimizer=Adam(learning_rate=cfg.lr), loss=self.fedprox_loss_instance, metrics=["accuracy"])

        # generating HAPs
        self.haps = [HAP(hap_id=i, hap_initial_weights=add_ber_noise(global_weights, cfg.ber)) for i in range(cfg.num_haps)] # HAP instances
        
        # generating SATs
        num_sats = cfg.num_haps * cfg.sats_per_hap
        self.sats = [SAT(i, add_ber_noise(global_weights, cfg.ber), random_non_iid(data[0][0], data[0][1], cfg.non_iid_number_max, cfg.non_iid_number_min)) for i in range(num_sats)]
        
        self.global_time = 0
        self.pre_simulate()

    def pre_simulate(self, duration_secs: int = cfg.total_steps, step_len: int = cfg.step_length):
        """
        Calculate and filling the following parameter for every SAT:
        sat.time_table, sat.accu_visible_time_table, sat.avg_visible_dist_table, sat.avg_fspl_ul_table, sat.avg_fspl_dl_table, sat.trajectories
        """
        num_sats, num_haps = cfg.num_haps*cfg.sats_per_hap, cfg.num_haps

        # Initialize result containers
        accu_visible_times = np.zeros((num_sats, num_haps))
        accu_visible_dist = np.zeros((num_sats, num_haps))
        accu_visible_fspl_ul = np.zeros((num_sats, num_haps))
        accu_visible_fspl_dl = np.zeros((num_sats, num_haps))
        time_tables = np.zeros((num_sats, num_haps, cfg.total_steps), dtype=bool)
        trajectories = np.empty((num_sats, cfg.total_steps, 3), dtype=np.float32)

        z_axis = np.array([0.0, 0.0, 1.0])
        print("Started pre_simulate!")
        for t in range(0, duration_secs, step_len):
            t_abs   = self.global_time + t
            theta_e = cfg.earth_omega * t_abs          # Earth rotate angle

            for hap in self.haps:
                hap.pos = rodrigues_rotation(hap.pos_ecef, z_axis, theta_e)
                hap.tangent_vector = hap.pos / np.linalg.norm(hap.pos)

            for sat in self.sats:
                rotate_angle = sat.omega * cfg.step_length
                sat.pos = rodrigues_rotation(sat.pos, sat.tangent_vector, rotate_angle)
                check_position(f"SAT{sat.sat_id}", sat.pos, cfg.re)
                trajectories[sat.sat_id, t] = sat.pos
                sat.trajectories[t] = sat.pos

                for hap in self.haps:
                    dx = sat.pos[0] - hap.pos[0]
                    dy = sat.pos[1] - hap.pos[1]
                    dz = sat.pos[2] - hap.pos[2]
                    distance = math.sqrt(dx**2 + dy**2 + dz**2)
                    cos = (dx * hap.tangent_vector[0] + dy * hap.tangent_vector[1] + dz * hap.tangent_vector[2]) / distance
                    angle = math.acos(cos)
                    visible = ((0.5 * math.pi) - angle) > cfg.min_visible_angle

                    if visible:
                        time_tables[sat.sat_id, hap.hap_id, t] = True
                        accu_visible_times[sat.sat_id, hap.hap_id] += step_len
                        accu_visible_dist[sat.sat_id, hap.hap_id] += distance
                        accu_visible_fspl_ul[sat.sat_id, hap.hap_id] += fspl_db(cfg.ka_b_mhz, distance)
                        accu_visible_fspl_dl[sat.sat_id, hap.hap_id] += fspl_db(cfg.ka_c_mhz, distance)

        # Calculate averages
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_visible_fspl_ul = np.full((num_sats, num_haps), np.inf)
            avg_visible_fspl_dl = np.full((num_sats, num_haps), np.inf)
            avg_visible_dist = np.full((num_sats, num_haps), np.inf)
            mask = accu_visible_times > 0
            avg_visible_fspl_ul[mask] = accu_visible_fspl_ul[mask] / accu_visible_times[mask]
            avg_visible_fspl_dl[mask] = accu_visible_fspl_dl[mask] / accu_visible_times[mask]
            avg_visible_dist[mask] = accu_visible_dist[mask] / accu_visible_times[mask]

        # Store to SAT instances
        for sat in self.sats:
            sid = sat.sat_id
            sat.time_table = time_tables[sid]                      # shape (num_haps, TOTAL_STEPS)
            sat.accu_visible_time_table = accu_visible_times[sid]  # shape (num_haps,)
            sat.avg_visible_dist_table = avg_visible_dist[sid]     # shape (num_haps,)
            sat.avg_fspl_ul_table = avg_visible_fspl_ul[sid]       # shape (num_haps,)
            sat.avg_fspl_dl_table = avg_visible_fspl_dl[sid]       # shape (num_haps,)
            sat.trajectories = trajectories[sid]

        self.global_time += duration_secs

    def integrate_sat_event(self):
        """
        integrates every events of SAT in a round
        return [(sat,hap,start,end,dir,ber), …] (raise sorted by start, from early to late)
        """
        for hap in self.haps:
            hap.reset_round_stats()

        evts = []
        for sat in self.sats:
            sat.event_counter = 0
            for h, td_s, td_e, tu_s, tu_e, ber_dl, ber_ul in sat.plan_greedy_roundtrip(cfg.bandwidth_hz, self.model_size_bits):
                evts.append((sat.sat_id, h, td_s, td_e, 'DL', ber_dl))
                evts.append((sat.sat_id, h, tu_s, tu_e, 'UL', ber_ul))
                sat.event_counter += 2

                self.haps[h].tx_sat_ids_round.add(sat.sat_id)
                self.haps[h].planned_cnt_round += 1
                
            print(f"SAT {sat.sat_id} planed {sat.event_counter} event")
        return sorted(evts, key=lambda x: x[2])

    def avg_fspl_between_haps(self, hap_a, hap_b, freq_mhz=cfg.ka_c_mhz):
        d = np.linalg.norm(np.array(hap_a.pos) - np.array(hap_b.pos))
        return fspl_db(freq_mhz, d)
    
    def baseline_aggregation(self)->dict:
        """
        inter-HAP model aggregation algorithm:
        1. for all active HAP, broadcast its model to others
        2. when the broadcast is over, for all active HAP, aggregating the recieve model with a weight of distance 
        3. return a model list of every aggregated HAP model
        """
        active_haps = [h for h in self.haps if h.is_active]
        if not active_haps:
            raise RuntimeError("All HAPs offline - cannot aggregate.")

        hap_models = {h.hap_id: h.last_weights for h in active_haps}
        
        received_models = {h.hap_id: [] for h in active_haps}
        received_weights = {h.hap_id: [] for h in active_haps}

        # broadcasting
        for src in active_haps:
            for dst in active_haps:
                if src == dst:
                    continue
                fspl_db = self.avg_fspl_between_haps(src, dst)
                _, ber = calc_snr_ber(fspl_db, cfg.bandwidth_hz, cfg.sat_tx_power_dbm, 
                                      cfg.sat_tx_gain_dbi, cfg.hap_rx_gain_dbi, cfg.noise_figure_db, cfg.system_loss_db)
                noisy_weights = add_ber_noise(hap_models[src.hap_id], ber)
                received_models[dst.hap_id].append(noisy_weights)

                received_weights[dst.hap_id].append(1)

        # adding its model
        for hap in active_haps:
            received_models[hap.hap_id].append(hap_models[hap.hap_id])
            received_weights[hap.hap_id].append(1)

        # Aggregating
        model_dict = {}
        for hap in active_haps:
            models = received_models[hap.hap_id]
            weights = received_weights[hap.hap_id]
            total_weight = sum(weights)
            aggregated = [np.zeros_like(layer) for layer in models[0]]
            for model, w in zip(models, weights):
                for i, layer in enumerate(model):
                    aggregated[i] += layer * (w / total_weight)
            model_dict[hap.hap_id] = aggregated

        return model_dict

    def mutual_aggregation(self, other_hap_constant, hap_self_weight)->dict:

        """
        inter-HAP model aggregation algorithm:
        1. for all active HAP, broadcast its model to others
        2. when the broadcast is over, for all active HAP, aggregating the recieve model with a weight of distance 
        3. return a model list of every aggregated HAP model
        """
        active_haps = [h for h in self.haps if h.is_active]
        if not active_haps:
            raise RuntimeError("All HAPs offline - cannot aggregate.")

        hap_models = {h.hap_id: h.last_weights for h in active_haps}
        
        received_models = {h.hap_id: [] for h in active_haps}
        received_weights = {h.hap_id: [] for h in active_haps}

        # broadcasting
        for src in active_haps:
            for dst in active_haps:
                if src == dst:
                    continue
                fspl_db = self.avg_fspl_between_haps(src, dst)
                _, ber = calc_snr_ber(fspl_db, cfg.bandwidth_hz, cfg.sat_tx_power_dbm, 
                                      cfg.sat_tx_gain_dbi, cfg.hap_rx_gain_dbi, cfg.noise_figure_db, cfg.system_loss_db)
                noisy_weights = add_ber_noise(hap_models[src.hap_id], ber)
                received_models[dst.hap_id].append(noisy_weights)

                distance = np.linalg.norm(np.array(src.pos) - np.array(dst.pos))
                weight = (1.0 / distance) * other_hap_constant
                received_weights[dst.hap_id].append(weight)

        # adding its model
        for hap in active_haps:
            received_models[hap.hap_id].append(hap_models[hap.hap_id])
            received_weights[hap.hap_id].append(hap_self_weight)

        # Aggregating
        model_dict = {}
        for hap in active_haps:
            models = received_models[hap.hap_id]
            weights = received_weights[hap.hap_id]
            total_weight = sum(weights)
            aggregated = [np.zeros_like(layer) for layer in models[0]]
            for model, w in zip(models, weights):
                for i, layer in enumerate(model):
                    aggregated[i] += layer * (w / total_weight)
            model_dict[hap.hap_id] = aggregated

        return model_dict

    def run_system(self, rounds:int, epochs:int, inter_hap_aggregation="baseline"):
        stats = TrainingStats(folder_path = self.folder_path)
        (x_train, y_train), (x_test, y_test) = self.data

        for r in range(rounds):
            self.pre_simulate()
            events = self.integrate_sat_event()
            num_events = len(events)
            print(f"Round {r+1} started, it has {num_events} events to be executed")

            # Testing
            for hap in self.haps:
                print(f"HAP {hap.hap_id:2d}: {len(hap.tx_sat_ids_round):2d} SAT(s) scheduled, total {hap.planned_cnt_round:3d} events")

            for i, (sat_id, hap_id, _, _, direction, _) in enumerate(events, start=1):
                print(f"round {r+1}/{rounds}, events {i}/{num_events}:")
                sat = cast(SAT, self.sats[sat_id])
                hap = cast(HAP, self.haps[hap_id])     
                
                if direction == 'DL':  # HAP→SAT
                    local_w = sat.weights
                    sat.receive_hap_weight(hap.hap_id, hap.last_weights, noise_option=True)

                    self.fedprox_loss_instance.update_weights(sat.weights, local_w)
                    self.center_model.set_weights(sat.weights)
                    self.center_model.fit(sat.train_data[0], sat.train_data[1], epochs=epochs, verbose=0)
                    sat.weights = self.center_model.get_weights()
                else:                  # SAT→HAP
                    hap.receive_sat_weights(sat, noise_option=True)
                    hap.check_aggregate_condition()

            # aggregating every model on HAP
            if inter_hap_aggregation == "baseline":
                transmitted_models_dict = self.baseline_aggregation()
            elif inter_hap_aggregation == "mutual":
                transmitted_models_dict = self.mutual_aggregation(1e3, 2)
            else:
                print(f"[DEBUG] Wrong inter_hap_aggregation, Using Baseline")
                transmitted_models_dict = self.baseline_aggregation()

            for hap in self.haps:
                if hap.hap_id in transmitted_models_dict:
                    hap.last_weights = transmitted_models_dict[hap.hap_id]
                else:
                    print(f"[DEBUG] HAP {hap.hap_id} is offline, it didn't attend to the inter-HAP aggregation")

            for hap in self.haps:
                self.center_model.set_weights(hap.last_weights)
                hap_test_loss, hap_test_acc = self.center_model.evaluate(x_test, y_test, verbose=1)
                stats.log(hap.hap_id, "round_per_hap_acc", hap_test_acc)
                stats.log(hap.hap_id, "round_per_hap_loss", hap_test_loss)

        # log output
        for h in self.haps:
            print(f"[SUMMARY] Number of update of HAP {h.hap_id} : {h.accu_num_sat_comm}")
        stats.plot_all()

        # system 3d plot
        sat_positions = np.array([sat.pos for sat in self.sats])
        hap_positions = np.array([hap.pos for hap in self.haps])
        sat_trajectories = [sat.trajectories for sat in self.sats]
        stats.plot_system(sat_positions, hap_positions, sat_trajectories)

class SAT:
    def __init__(self, sat_id, initial_weights, train_data:tuple):
        self.sat_id = sat_id
        self.weights = initial_weights
        self.train_data = train_data

        self.event_counter = 0
        
        # altitude and initial position of the SAT 
        if self.sat_id % 2 == 0:
            self.orbital_radius = cfg.sat_alt_1 + cfg.re
        else:
            self.orbital_radius = cfg.sat_alt_2 + cfg.re

        pos_vec = generate_random_unit_vector()
        self.pos = pos_vec * self.orbital_radius
        check_position(f"Satellite {self.sat_id}", self.pos, cfg.re)    # for debug
        
        # orbital parameters
        self.omega, self.velocity, self.tangent_vector = self.generate_omega_velocity_tangent()

        # calculated in the function FedHAP.pre_simulate()
        self.time_table = None
        self.accu_visible_time_table = None
        self.avg_visible_dist_table = None
        self.avg_fspl_ul_table = None
        self.avg_fspl_dl_table = None
        self.trajectories = np.zeros((cfg.total_steps, 3), dtype=np.float32)
        self.trajectories[0] = self.pos
        
    def generate_omega_velocity_tangent(self):
        """
        return omega(rad/s), velocity(km/s), random unit velocity vector (orbit direction) 
        """
        while(True):
            u = generate_random_unit_vector()
            rvv = np.cross(self.pos, u)
            
            length = np.linalg.norm(rvv)
            if length == 0:   # just in case that the random vector is in the same line of pos vector
                continue
            else:
                # the normalized_outer_product is the normalized velocity vector of the SAT
                ruvv = rvv / length
                omega = math.sqrt(cfg.gm / (self.orbital_radius**3))  # (rad/s)
                velocity = omega * self.orbital_radius    # v=wR (km/s)
                return omega, velocity, ruvv

    def receive_hap_weight(self, hap_id, hap_weights, noise_option:bool):
        """
        Getting the model (weights) from the HAP
        """
        print(f"DL: HAP {hap_id} -> SAT {self.sat_id}, transmitting")
        if noise_option:
            snr_db, ber = calc_snr_ber(self.avg_fspl_dl_table[hap_id], cfg.bandwidth_hz, cfg.hap_tx_power_dbm, 
                                       cfg.hap_tx_gain_dbi, cfg.sat_rx_gain_dbi, cfg.noise_figure_db, cfg.system_loss_db)
            self.weights = add_ber_noise(hap_weights, ber)
        else:
            self.weights = hap_weights

    def get_visible_windows_list(self, hap_id: int, step_len=cfg.step_length):
        """
        transfer the time_table(bool list) to a time_window list [(start,end), …](esc)
        """
        windows_list, s = [], None
        for i, v in enumerate(self.time_table[hap_id]):
            if v and s is None:
                s = i
            elif not v and s is not None:
                windows_list.append((s*step_len, i*step_len)); s = None
        if s is not None:
            windows_list.append((s*step_len, len(self.time_table[hap_id])*step_len))
        return windows_list

    def plan_greedy_roundtrip(self, bandwidth_hz: float, model_size_bit: float):
        """
        Use earliest-finish-greedy algorithm to find out best event schedule,
        filtering the possible event with 2 rule to better the model quality:
        rule1: the visible window should be long enough for UL + DL + training (shannon limit)
        rule2: the average FSPL of UL+DL should <= FSPL_MAX_DB
        return: events = [(hap_id, td_s, td_e, tu_s, tu_e, ber_dl, ber_ul), …], raise sorted by td_s (from early to late)
        """
        cand = []   # (finish,start,hap,ber_dl,ber_ul,t_dl,t_ul)

        for hap_id in range(len(self.avg_fspl_ul_table)):
            if self.accu_visible_time_table[hap_id] == 0:
                continue

            snr_dl_db, ber_dl = calc_snr_ber(self.avg_fspl_dl_table[hap_id], cfg.bandwidth_hz, cfg.sat_tx_power_dbm, 
                                             cfg.sat_tx_gain_dbi, cfg.hap_rx_gain_dbi, cfg.noise_figure_db, cfg.system_loss_db)
            snr_ul_db, ber_ul = calc_snr_ber(self.avg_fspl_dl_table[hap_id], cfg.bandwidth_hz, cfg.sat_tx_power_dbm, 
                                             cfg.sat_tx_gain_dbi, cfg.hap_rx_gain_dbi, cfg.noise_figure_db, cfg.system_loss_db)
            cap_dl = shannon_capacity_bps(snr_dl_db, bandwidth_hz)
            cap_ul = shannon_capacity_bps(snr_ul_db, bandwidth_hz)

            t_dl  = model_size_bit / cap_dl
            t_train = 90
            t_ul  = model_size_bit / cap_ul
            t_total = t_dl + t_ul + t_train

            for start, end in self.get_visible_windows_list(hap_id):
                if end - start < t_total:                # filtering rule 1: the time window is too small
                    continue

                avg_ul = self.avg_fspl_ul_table[hap_id]
                avg_dl = self.avg_fspl_dl_table[hap_id]
                if (avg_ul + avg_dl) / 2 > cfg.fspl_max_db:  # filtering rule2: the FSPL is too large
                    continue
                cand.append((start + t_total, start, hap_id, ber_dl, ber_ul, t_dl, t_ul))

        if not cand:
            return []

        # Earliest finish greedy algorithm
        cand.sort()     # python sort it from the first index(finish) to the last index(t_ul)
        events, last_finish = [], -1
        for end, start, h, ber_dl, ber_ul, t_dl, t_ul in cand:
            if start >= last_finish:
                td_s, td_e = start, start + t_dl
                tu_s, tu_e = td_e, end
                events.append((h, td_s, td_e, tu_s, tu_e, ber_dl, ber_ul))
                last_finish = end
        return events

class HAP:
    def __init__(self, hap_id, hap_initial_weights):
        self.hap_id = hap_id
        self.sats:list[SAT] = []
        self.is_active = True

        self.weights_buffer = []
        self.weights_of_weights_buffer = []
        
        # generate a random position for HAP
        u = generate_random_unit_vector()
        self.pos = ((cfg.re + cfg.hap_altitude)*u[0],(cfg.re + cfg.hap_altitude)*u[1],(cfg.re + cfg.hap_altitude)*u[2]) # position vector of the HAP
        self.pos_ecef = np.array(self.pos, dtype=np.float64)
        check_position(f"HAP{self.hap_id}", self.pos, cfg.re)

        # calculating the normal vector of the hap
        self.pos_length = math.sqrt(self.pos[0]**2 + self.pos[1]**2 + self.pos[2]**2)
        self.tangent_vector = (self.pos[0]/self.pos_length, self.pos[1]/self.pos_length, self.pos[2]/self.pos_length)

        self.last_weights = hap_initial_weights

        self.tx_sat_ids_round: set[int] = set()   # recording the SAT that communicate with this HAP in this round
        self.tx_sat_ids_total: set[int] = set()   # recording the SAT that communicate with this HAP accrossing every round
        self.planned_cnt_round = 0
        self.accu_num_sat_comm = 0

    def receive_sat_weights(self, sat:SAT, noise_option:bool):
        print(f"UL: SAT {sat.sat_id} -> HAP {self.hap_id}, transmitting")

        if noise_option:
            snr_db, ber = calc_snr_ber(sat.avg_fspl_ul_table[self.hap_id], cfg.bandwidth_hz, cfg.sat_tx_power_dbm, 
                                       cfg.sat_tx_gain_dbi, cfg.hap_rx_gain_dbi, cfg.noise_figure_db, cfg.system_loss_db)
            noisy_w = add_ber_noise(sat.weights, ber)
            self.weights_buffer.append(noisy_w)
        else:
            self.weights_buffer.append(sat.weights)
        
        distance_km = sat.avg_visible_dist_table[self.hap_id]
        w_dist = 1 / distance_km
        w = cfg.aggregation_dist_constant * (w_dist) + len(sat.train_data[0])
        self.weights_of_weights_buffer.append(w)

    def clean_weights_buffer(self):
        self.weights_buffer.clear()
        self.weights_of_weights_buffer.clear()
    
    def aggregate_models(self):
        if len(self.weights_buffer) == 0:
            print("[INFO] This round of aggregation did nothings(weights is empty)")
            return
        
        self.weights_buffer.append(self.last_weights)
        hap_w = (cfg.non_iid_number_max + cfg.non_iid_number_min) / 2 + cfg.aggregation_dist_constant/500 * 2
        self.weights_of_weights_buffer.append(hap_w)

        weighted_sum = [np.zeros_like(layer) for layer in self.weights_buffer[0]]
        total_weight = sum(self.weights_of_weights_buffer)

        for model, weight in zip(self.weights_buffer, self.weights_of_weights_buffer):
            for i, layer in enumerate(model):
                weighted_sum[i] += layer * (weight / total_weight)
        self.last_weights = weighted_sum
        self.clean_weights_buffer()

    def check_aggregate_condition(self):
        self.aggregate_models()

    def reset_round_stats(self):
        self.accu_num_sat_comm += self.planned_cnt_round
        self.planned_cnt_round = 0
        self.tx_sat_ids_round.clear()

# multi-processing, depends on your RAM, VRAM volumn: 32GB RAM -> at most 2 expriment at the same time
import multiprocessing as mp
import tensorflow as tf

def run_exp(mode: str, gpu_mode="auto", model_type=None, dataset_type=None, rounds=None, epochs=None):
    cfg.model_type = model_type if model_type else cfg.model_type
    cfg.dataset_type = dataset_type if dataset_type else cfg.dataset_type
    cfg.rounds = rounds if rounds else cfg.rounds
    cfg.epochs = epochs if epochs else cfg.epochs

    print(f"[INFO] Aggregation: {mode}")
    print(f"[INFO] Device Mode: {gpu_mode}")
    print(f"[INFO] Model: {cfg.model_type}, Dataset: {cfg.dataset_type}")
    print(f"[INFO] Rounds: {cfg.rounds}, Epochs: {cfg.epochs}")
    if gpu_mode == "gpu-divide":
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=cfg.optional_vram_limit)])
        print("[INFO] dividing VRAM on a single gpu")
    elif gpu_mode == "cpu":
        tf.config.set_visible_devices([], "GPU")
        print(f"[INFO] [{mode}] forced to CPU only")
    elif gpu_mode == "auto":
        print(f"[INFO] [{mode}] using default tensorflow GPU config")

    # Load dataset
    if cfg.dataset_type == "mnist":
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
    elif cfg.dataset_type == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
    else:
        raise ValueError("[ERROR] Unsupported dataset")
    
    data = [(x_train, y_train), (x_test, y_test)]

    # Set model
    if cfg.model_type == "dnn":
        build_model_fn = build_dnn_model_mnist
    elif cfg.model_type == "cnn":
        build_model_fn = build_cnn_model_cifar10
    else:
        raise ValueError("[ERROR] Unsupported model type")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_path = f"{cfg.log_dir}/{mode}_{timestamp}"

    log_config(cfg=cfg, folder_path=folder_path)
    fedhap = FedHAP(data = data, folder_path=folder_path, build_model_fn=build_model_fn)
    try:
        fedhap.run_system(rounds=cfg.rounds, epochs=cfg.epochs, inter_hap_aggregation=mode)
    except Exception as e:
        print(f"[ERROR] [{mode}] crash: {e}")

import argparse
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description="FedHAP Simulation")
    parser.add_argument("--aggregation", type=str, choices=["baseline", "mutual"], default="baseline")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "gpu-divide"], default="auto")
    parser.add_argument("--model", type=str, choices=["dnn", "cnn"], default="dnn")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    p = mp.Process(target=run_exp, args=(args.aggregation, args.device, args.model, args.dataset, args.rounds, args.epochs))
    p.start()
    p.join()

#指令範例 python main.py --aggregation mutual --device auto --model dnn --dataset mnist --rounds 10 --epochs 5