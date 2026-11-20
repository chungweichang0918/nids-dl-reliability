# Mac mini 執行說明

這個資料夾包含在 Mac 上補跑 RNN 種子所需的全部檔案，已自帶資料集 (NSL-KDD、UNSW-NB15)。

## 步驟

1. 把整個 `mac_package` 資料夾複製到 Mac，開終端機並進入該資料夾。
2. 安裝環境 (只需一次，約幾分鐘):
   ```
   bash setup_mac.sh
   ```
   最後會印出 `MPS (Apple GPU) 可用: True/False`。
   若腳本提示找不到 `.venv/bin/python`，請在 Mac 上重跑此步驟。安裝腳本會先備份無法啟動的舊環境。
3. 測速（時間依機型而異）:
   ```
   ./bench.sh
   ```
   完成後會印出平均每個 epoch 的秒數。筆電約 40 秒，**60 秒以內值得用**。
   測速與正式訓練預設用 CPU。若要測 Apple GPU，可用 `KERAS_TORCH_DEVICE=mps ./bench.sh`。
4. 正式訓練 RNN 種子 4、5 (筆電負責 1–3):
   ```
   ./run_mac.sh
   ```
   可隨時按 Ctrl+C 中斷，再執行一次會從沒完成的接續。
   若 Apple GPU 測速成功並決定使用它，請用 `KERAS_TORCH_DEVICE=mps ./run_mac.sh`。
5. 訓練完成後打包結果:
   ```
   ./collect.sh
   ```
   把產生的 `mac_results.zip` 複製回筆電的專案資料夾解壓縮 (合併進 `results/`)，告訴 Claude 即可。

## 注意

- 若 `./setup_mac.sh` 顯示 Permission denied，先執行 `chmod +x *.sh`。
- 每個種子要跑「二元」和「五類」各一次 (共 2 個 RNN 工作)，種子 4、5 共 4 個。
- 不要和筆電同時跑同一個種子，否則會重複 (兩台機器的鎖定檔不互通)。目前筆電在跑種子 1–3 的 RNN。
- 不同硬體的結果會有些微差異，報告中會註明部分種子在 Apple 晶片上執行。
