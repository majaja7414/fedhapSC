# HAP-SAT 聯邦學習系統

此倉庫實作高空平台 (HAP) 與衛星 (SAT) 之聯邦學習模擬環境，
支援 DNN 與 CNN 兩種模型，以及 MNIST、CIFAR10 等常見資料集。

## 環境需求
- Python 3.10 以上
- 先執行 `pip install -r requirements.txt` 安裝所有套件

## 基本使用方式
1. 調整 `config.py` 中的重要參數，例如 `rounds`、`epochs`、`model_type` 與 `dataset_type`。
2. 進行單次實驗：
   ```bash
   python main.py
   ```
3. 執行多事件/多處理流程：
   ```bash
   python test_multi_evt.py
   ```

## 輸出結果
- 所有執行紀錄與圖表會儲存於 `exp_log/` 目錄下，
  其中包含每回合的準確率與損失等資料。

## 其他腳本
- `test_multi_evt.py`：示範以多處理方式同時執行多個模擬。
- 其他自訂測試腳本亦可依照以上方式執行。

