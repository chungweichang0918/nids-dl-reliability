"""診斷 LSTM / GRU / CNN-LSTM 為何沒有重現論文結果的實驗。

每個實驗同時記錄「最後一個 epoch」與「驗證集最佳 epoch」兩種權重的測試結果，
用來區分訓練不穩定 (最後 epoch 較差) 與模型本身能力不足 (兩者都差)。

變體:
  base_s1   : 與主實驗相同設定，只換隨機種子 (量測種子變異)
  clip      : 梯度裁剪 clipnorm=1.0 (檢驗梯度暴衝假設)
  lr3e-4    : 學習率 3e-4
  plain5000 : 不做梯度累積，直接 batch 5000 (檢驗梯度累積實作假設，只有 LSTM 二元放得下)

用法: .venv/Scripts/python.exe experiments.py
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import json
import time
import warnings

import keras
import numpy as np
from sklearn.model_selection import train_test_split

import nids_repro as nr

warnings.filterwarnings("ignore", message="RNN module weights are not part")

OUTDIR = nr.OUT / "ablation"
JOBS = [  # (task, model) 依訓練時間由短到長
    ("binary", "LSTM"),
    ("multi", "CNN-LSTM"),
    ("binary", "GRU"),
]
VARIANTS = {
    "base_s1": dict(seed=1),
    "clip": dict(seed=42, clipnorm=1.0),
    "lr3e-4": dict(seed=42, lr=3e-4),
    "plain5000": dict(seed=42, plain=True),
}


class KeepBest(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best, self.epoch, self.weights = -1.0, 0, None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_accuracy"] > self.best:
            self.best, self.epoch = logs["val_accuracy"], epoch + 1
            self.weights = self.model.get_weights()


def run(task, name, variant, cfg, splits, epochs=100):
    path = OUTDIR / f"{task}_{name}_{variant}.json"
    if path.exists():
        return
    X_train, X_test, y_train, y_test = splits[task]
    keras.utils.set_random_seed(cfg["seed"])
    plain = cfg.get("plain", False)
    if name in nr.RECURRENT_MODELS and not plain:
        batch, steps = nr.MICRO_BATCH, nr.BATCH_SIZE // nr.MICRO_BATCH
    else:
        batch, steps = nr.BATCH_SIZE, None
    model = nr.build_model(name, task)
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=cfg.get("lr", 1e-3), clipnorm=cfg.get("clipnorm"),
            gradient_accumulation_steps=steps),
        loss=model.loss, metrics=["accuracy"])
    X_train, X_test = X_train[..., None], X_test[..., None]

    keep = KeepBest()
    t0 = time.time()
    hist = model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                     validation_split=0.2, verbose=0, callbacks=[keep])
    seconds = time.time() - t0

    last, _, _ = nr.evaluate(task, y_test, model.predict(X_test, batch_size=batch, verbose=0))
    model.set_weights(keep.weights)
    best, _, _ = nr.evaluate(task, y_test, model.predict(X_test, batch_size=batch, verbose=0))
    val = hist.history["val_accuracy"]
    out = dict(task=task, model=name, variant=variant, cfg=cfg, seconds=round(seconds),
               last=last, best=best, best_epoch=keep.epoch,
               val_accuracy=[float(v) for v in val],
               val_loss=[float(v) for v in hist.history["val_loss"]],
               # 訓練是否「崩落」: 後段驗證準確率比前面最高值低超過 5 個百分點的次數
               collapses=int(sum(1 for i in range(1, len(val)) if val[i] < max(val[:i]) - 0.05)))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"[{task}] {name:8s} {variant:9s} last={last['accuracy']*100:.2f} "
          f"best={best['accuracy']*100:.2f}@{keep.epoch}  collapses={out['collapses']} "
          f"({seconds:.0f}s)", flush=True)


def main():
    X, y_bin, y_multi = nr.load_data()
    splits = {
        "binary": train_test_split(X, y_bin, test_size=0.2, random_state=nr.SEED),
        "multi": train_test_split(X, y_multi, test_size=0.2, random_state=nr.SEED),
    }
    for variant, cfg in VARIANTS.items():
        for task, name in JOBS:
            if cfg.get("plain") and (task, name) != ("binary", "LSTM"):
                continue  # 其他模型 batch 5000 放不進 6GB 顯示卡
            run(task, name, variant, cfg, splits)


if __name__ == "__main__":
    main()
