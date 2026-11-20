"""量測六個模型在這台機器各種裝置上的訓練速度，推估三個資料集每個 epoch 要多久。

用假資料測，不需要資料集。會分別測 CPU、Apple GPU (MPS，不支援的運算退回 CPU)、NVIDIA GPU (若有)。

用法: python bench_devices.py              (約 5-15 分鐘)
      python bench_devices.py --quick      (只測 DNN / CNN / LSTM)
"""
import json
import os
import subprocess
import sys
import time

MODELS = ["DNN", "CNN", "LSTM", "CNN-LSTM", "GRU", "RNN"]
# (資料集, 特徵數, 訓練筆數 = 扣除測試與驗證後實際拿來訓練的筆數)
DATASETS = [("NSL-KDD", 122, 80_622), ("UNSW-NB15", 67, 140_273), ("CIC-IDS2017", 77, 1_177_417),
            ("CSE-CIC-IDS2018", 77, 1_910_191)]
STEPS = 7           # 每個模型量 7 個 batch，前 2 個當暖機


def worker(device, models):
    os.environ["KERAS_BACKEND"] = "torch"
    os.environ["KERAS_TORCH_DEVICE"] = device
    if device == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    import warnings
    warnings.filterwarnings("ignore")
    import keras
    import numpy as np
    import nids_core  # noqa: F401  MPS 時自動套用 orthogonal 初始化的修正
    import nids_repro as nr

    class Timer(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            self.t = [time.perf_counter()]

        def on_train_batch_end(self, batch, logs=None):
            self.t.append(time.perf_counter())

    out = {}
    rng = np.random.default_rng(0)
    for name in models:
        for n_feat in sorted({d[1] for d in DATASETS}):           # 兩個 CIC 資料集特徵數相同，只量一次
            key = f"{name}/{n_feat}"
            try:
                recurrent = name in nr.RECURRENT_MODELS
                batch = nr.MICRO_BATCH if recurrent else nr.BATCH_SIZE
                model = nr.build_model(name, "binary", n_features=n_feat,
                                       accumulation_steps=5 if recurrent else None)
                X = rng.standard_normal((batch * STEPS, n_feat)).astype("float32")
                if name in nr.SEQUENCE_MODELS:
                    X = X[..., None]
                y = rng.integers(0, 2, len(X))
                timer = Timer()
                model.fit(X, y, batch_size=batch, epochs=1, verbose=0, callbacks=[timer])
                step = float(np.median(np.diff(timer.t)[2:]))
                out[key] = dict(sec_per_step=step, batch=batch)
            except Exception as e:                      # 例如 MPS 不支援某個運算
                out[key] = dict(error=f"{type(e).__name__}: {str(e)[:120]}")
            print(f"  [{device}] {key:14s} " + (f"{out[key]['sec_per_step'] * 1000:8.1f} ms/batch"
                                                if "error" not in out[key] else out[key]["error"]),
                  file=sys.stderr, flush=True)
    print("RESULT " + json.dumps(out))


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--worker":
        return worker(sys.argv[2], sys.argv[3].split(","))
    models = ["DNN", "CNN", "LSTM"] if "--quick" in sys.argv else MODELS

    import torch
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    if "--cuda" in sys.argv:                             # 雲端主機: 只量顯示卡，省下 CPU 測試的時間
        devices = ["cuda"]
    print(f"torch {torch.__version__}，可用裝置: {devices}，CPU 執行緒 {torch.get_num_threads()}")

    results = {}
    for device in devices:
        print(f"\n測試 {device} ...", flush=True)
        p = subprocess.run([sys.executable, __file__, "--worker", device, ",".join(models)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        sys.stdout.write(p.stderr if p.returncode == 0 else p.stderr[-2000:])
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")]
        results[device] = json.loads(line[-1][7:]) if line else {}

    print("\n=== 每個 epoch 的估計時間 (分鐘) ===")
    header = f"{'模型':10s}" + "".join(f"{d + ' ' + dev:>22s}" for d, _, _ in DATASETS for dev in devices)
    print(header)
    for name in models:
        row = f"{name:10s}"
        for _, n_feat, n_train in DATASETS:
            for dev in devices:
                r = results.get(dev, {}).get(f"{name}/{n_feat}")
                if not r or "error" in r:
                    row += f"{'失敗':>22s}"
                else:
                    row += f"{r['sec_per_step'] * n_train / r['batch'] / 60:22.2f}"
        print(row)
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(f"\n結果已存到 {out}，請把上面的表格貼給 Claude。")


if __name__ == "__main__":
    main()
