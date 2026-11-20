"""雲端上的 FT-Transformer 檢查 (約 3-5 分鐘): 建模、weight decay 排除、短訓練、權重重新載入、速度估算。"""
import os
import time

os.environ.setdefault("KERAS_BACKEND", "torch")
import numpy as np

import nids_core as core
import nids_data
import tune

b = nids_data.load("nsl")
print("試驗 0 (原論文預設):", tune.trial_hp("nsl", "binary", "FTT", 0))
m = core.build(b, "FTT", "binary")
vs = m.trainable_variables
no_decay = [v.path for v in vs if not m.optimizer._use_weight_decay(v)]
print(f"參數 {m.count_params():,}；不做 weight decay 的變數 {len(no_decay)}/{len(vs)}，例如 {no_decay[:3]}")

sub = dict(b, train={k: v[:20000] for k, v in b["train"].items()})
t = time.time()
model, info, _ = core.train(sub, "FTT", "binary", seed=1, epochs=2, title="FTT 測試")
print(f"2 萬筆 x 2 輪: {time.time() - t:.0f}s  驗證準確率 {[round(v * 100, 2) for v in info['val_accuracy']]}")
p1 = core.predict(model, "FTT", b["tests"]["random"]["X"][:3000])
model.save_weights("/tmp/ftt.weights.h5")
m2 = core.build(b, "FTT", "binary")
m2.load_weights("/tmp/ftt.weights.h5")
print("重新載入後預測相同:", bool(np.allclose(p1, core.predict(m2, "FTT", b["tests"]["random"]["X"][:3000]), atol=1e-6)))

# 速度: 原論文預設設定在 CSE-CIC-IDS2018 上 1 個 epoch 的時間 (用 20 萬筆推算)
c = nids_data.load("cic18")
n_full = int(len(c["train"]["X"]) * 0.8)
sub = dict(c, train={k: v[:200_000] for k, v in c["train"].items()})
t = time.time()
core.train(sub, "FTT", "binary", seed=1, epochs=1, title="FTT 速度", verbose=False, grad_norm=False)
sec = (time.time() - t) / 160_000 * n_full
print(f"CSE-CIC-IDS2018 預設設定: 每個 epoch 約 {sec / 60:.1f} 分鐘 (與其他模型共用顯示卡時)")
