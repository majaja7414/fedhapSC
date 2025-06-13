# 高空平台結合LEO衛星之聯邦學習 - 基於FSPL的傳輸與聚合策略

本專題模擬高空平台（HAP）與低軌道衛星（LEO-SAT）組成之聯邦學習架構，聚焦於模型在傳輸通道中受 FSPL 與 BER 影響下的傳送與聚合策略。

---

## 使用方式

### 環境

```bash
需要安裝 tensorflow(加上keras), matplotlib, plotly
tensorflow的部分最新版本測試到2.10.0 + cuda11.2 + python 3.10
舊版本的tensorflow + python 3.9也可以，根據自己的GPU或作業系統選擇好版本即可
```

### 執行方式

#### 1. 直接執行main.py，會用預設的參數運行
#### 2. Bash (Terminal)輸入參數執行，格式為：
```bash
python main.py --aggregation [baseline|mutual] --device [auto|cpu|gpu-divide] --model [dnn|cnn] --dataset [mnist|cifar10] --rounds [int] --epochs [int]
```
--aggregation:選擇聚合方式，baseline為純fed-avg，mutual加入距離權重(結果會更好)  
--device: auto為讓tensorflow自己抓，cpu使用cpu，gpu-divide使用gpu並且限制最大VRAM，不然邊看球賽邊跑程式可能會爆VRAM，預設4096MB，可在config.py中修改  
--model:選擇模型種類，cnn或dnn  
--dataset:選擇訓練資料，mnist或cifar10  
--rounds:全局要聚合多少次(HAP之間)，也就是衛星繞多少圈地球，預設為10  
--epochs:衛星的每一次訓練要跑多少rounds，mnist少一點，cifar10需要多一點，預設為5

範例：
```bash
python main.py --aggregation mutual --device auto --model dnn --dataset mnist --rounds 10 --epochs 10
```

執行結果包含測試準確率、loss、設定，自動產生時間和檔名存於 `exp_log/` 下

---

## 主要檔案說明

| 檔案名稱           | 功能簡述                          |
| -------------- | ----------------------------- |
| `main.py`      | 模擬主程式，建構聯邦學習架構、執行輪次、訓練與聚合 |
| `utilities.py` | 其他helper functions：包含通道計算、模型建構、紀錄等功能 |
| `config.py`    | 模擬全域設定（節點數、通道參數、超參數等）      |

---

## 程式碼說明
函式、參數簡單概要，每個函式的詳細功能在內的註解
### `config.py`

透過class `SimConfig`，預設模擬所需的所有參數：

* 通訊參數：`fspl_max_db`, `snr_db`, `bandwidth_hz`, `tx_power_dbm` 等
* 模型訓練參數：`rounds`, `epochs`, `model_type`, `fedprox_mu`
* 軌道模擬參數：`hap_altitude`, `sat_alt_1`, `re`, `earth_omega` 等

---

### `utilities.py`

#### 通訊模擬

* `fspl_db`：計算自由空間路徑損失 (FSPL)
* `calc_snr_ber`：計算通道的 SNR 與 BER
* `add_ber_noise`：根據 BER 對模型權量進行模擬破壞
* `shannon_capacity_bps`：給出shannon極限下的最高傳輸速率(bps)

#### 衛星軌道 & 物理計算

* `generate_random_unit_vector`：產生随機方向向量，用來隨機生成節點的初始位置(HAP、SAT)  
* `rodrigues_rotation`：Rodrigues 旋轉公式模擬地球自轉、SAT軌道運動
* `check_position`：檢查節點是否陷入地球

#### 模型建構

* `build_dnn_model_mnist/cifar10`：建立 DNN 模型
* `build_cnn_model_cifar10`：建立 CNN 模型
* `random_non_iid`：為各 client 隨機產生 non-IID 訓練資料
* `FedproxLoss`：自訂義 FedProx 損失函數，包含權量更新功能

#### 視覺化

* `TrainingStats`：記錄與繪製每輪測試結果圖（acc/loss），或顯示出3D系統拓樸

---

### `main.py`

#### class：`FedHAP`

* `__init__()`：初始化 HAP/SAT，載入模型與資料
* `run_system()`：執行模擬主迴圈
* `pre_simulate()`：模擬 SAT 與 HAP 的可視時間/距離/FSPL
* `integrate_sat_event()`：要上傳下載的事件規劃

#### class：`HAP`

* `receive_sat_weights()`, `aggregate_models()`：接收並聚合 SAT 模型
* `check_aggregate_condition()`：監測是否聚合（目前預設為無條件）

#### class：`SAT`

* `plan_greedy_roundtrip()`：最早結束策略來規劃傳輸時機
* `receive_hap_weight()`：下載 HAP 模型
* `get_visible_windows_list()`：轉換可視表為時間區間

---

## 聚合方式比較

| 模式         | 說明                             |
| ---------- | ------------------------------ |
| `baseline` | 所有 HAP 互傳模型並等權平均               |
| `mutual`   | 加入 inter-HAP 距離作為聚合權量，較穩健但收敘慢一點，可以去調整Fedprox中的mu (config.py，class Simconfig中的fedprox_mu)加速聚合 |

---

## 輸出

* 每次執行後產生一個包含所有圖表與CSV的資料夾，可透過CSV進行後續數據操作
* 檔案包含每個HAP在每輪的測試資料集的準確率、損失，與系統3D拓樸 (HTML檔案)

---
