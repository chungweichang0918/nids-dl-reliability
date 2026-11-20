"""訓練與評估核心: 三個資料集、六個模型共用。

所有 v2 實驗 (調參、多種子、評估條件、機制分析) 都透過 train() 與 evaluate()，
確保訓練流程完全一致。結果存放在 results/v2/。
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import gc
import json
import time
import warnings

warnings.filterwarnings("ignore", message="RNN module weights are not part")
warnings.filterwarnings("ignore", message="Skipping variable loading for optimizer")

import keras
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

import nids_repro as nr

from pathlib import Path


def patch_orthogonal_for_mps():
    """Apple GPU (MPS) 不支援 QR 分解，循環層的 orthogonal 初始化會失敗。
    改用 NumPy 在 CPU 上算出正交矩陣再轉成張量 (數學上等價，只影響初始化)。"""
    from keras.src.initializers.random_initializers import Orthogonal as OrthogonalInitializer

    def call(self, shape, dtype=None):
        rows, cols = int(np.prod(shape[:-1])), int(shape[-1])
        seed = self.seed if isinstance(self.seed, int) else None
        a = np.random.default_rng(seed).standard_normal((max(rows, cols), min(rows, cols)))
        q, r = np.linalg.qr(a)
        q *= np.sign(np.diag(r))
        if rows < cols:
            q = q.T
        return keras.ops.convert_to_tensor((self.gain * q).reshape(shape), dtype=dtype or "float32")

    OrthogonalInitializer.__call__ = call


if os.environ.get("KERAS_TORCH_DEVICE") == "mps":
    patch_orthogonal_for_mps()

# NIDS_TF32=1: 矩陣乘法使用 TF32 (Ampere 以後的顯示卡)。只用於 FT-Transformer 的行程，論文中註明。
if os.environ.get("NIDS_TF32") == "1":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# NIDS_OUT 環境變數可把結果導到別的資料夾 (測試用)
OUT = Path(os.environ.get("NIDS_OUT") or nr.OUT / "v2")
MODELS = ["DNN", "CNN", "LSTM", "CNN-LSTM", "GRU", "RNN"]      # 由快到慢


def device():
    from keras.src.backend.torch import core
    return str(core.get_device())


def shape_input(name, X):
    return X[..., None] if name in nr.SEQUENCE_MODELS else X


def batching(name):
    """循環模型與 FT-Transformer 用 1000 筆 x 累積 5 次梯度 (等效 batch 5000)，其餘直接 batch 5000。"""
    if name in nr.MICRO_BATCH_MODELS:
        return nr.MICRO_BATCH, nr.BATCH_SIZE // nr.MICRO_BATCH
    return nr.BATCH_SIZE, None


def build(bundle, name, task, hp=None):
    batch, steps = batching(name)
    return nr.build_model(name, task, n_features=bundle["n_features"], accumulation_steps=steps,
                          n_classes=len(bundle["classes"]), hp=hp)


class Monitor(keras.callbacks.Callback):
    """記錄最佳驗證權重、每個 epoch 的梯度範數，並顯示進度。"""

    def __init__(self, title, epochs, keep_best=True, grad_norm=True, verbose=True):
        super().__init__()
        self.title, self.epochs, self.verbose = title, epochs, verbose
        self.keep_best, self.track_grad = keep_best, grad_norm
        self.best, self.best_epoch, self.best_weights = -1.0, 0, None
        self.grad_max, self.grad_mean, self._norms = [], [], []

    def on_train_begin(self, logs=None):
        self.t0 = time.time()
        if self.verbose:
            print(f"{self.title} 開始訓練 ({time.strftime('%H:%M')}, {device()})", flush=True)

    def on_train_batch_end(self, batch, logs=None):
        if not self.track_grad:
            return
        # torch 後端: 反向傳播後梯度留在各變數的 .grad (裁剪前的原始梯度)
        total = 0.0
        for v in self.model.trainable_variables:
            g = v.value.grad
            if g is not None:
                total += float(g.detach().pow(2).sum())
        self._norms.append(total ** 0.5)

    def on_epoch_end(self, epoch, logs=None):
        if self._norms:
            self.grad_max.append(float(np.max(self._norms)))
            self.grad_mean.append(float(np.mean(self._norms)))
            self._norms = []
        if self.keep_best and logs["val_accuracy"] > self.best:
            self.best, self.best_epoch = logs["val_accuracy"], epoch + 1
            self.best_weights = self.model.get_weights()
        if self.verbose:
            used = time.time() - self.t0
            left = used / (epoch + 1) * (self.epochs - epoch - 1)
            clock = lambda s: f"{int(s) // 60:d}:{int(s) % 60:02d}"
            print(f"\r  epoch {epoch + 1:3d}/{self.epochs}  val_acc={logs['val_accuracy'] * 100:6.2f}%  "
                  f"已用 {clock(used)}  剩餘約 {clock(left)}   ", end="", flush=True)

    def on_train_end(self, logs=None):
        if self.verbose:
            print("\r" + " " * 78 + "\r", end="")


def collapses(val):
    """驗證準確率比先前最高值低 5 個百分點以上的 epoch 數。"""
    return int(sum(1 for i in range(1, len(val)) if val[i] < max(val[:i]) - 0.05))


def train(bundle, name, task, seed, hp=None, epochs=None, title="", verbose=True,
          keep_best=True, grad_norm=True, X=None, y=None):
    """訓練一個模型。回傳 (model, info)；info 含訓練曲線、梯度範數與崩落次數。

    X, y 可覆寫訓練資料 (例如打亂特徵順序或對抗訓練)，預設使用 bundle 的訓練集。
    """
    import nids_data
    epochs = epochs or nids_data.EPOCHS[bundle["name"]]
    X = bundle["train"]["X"] if X is None else X
    y = bundle["train"][task] if y is None else y
    batch, _ = batching(name)
    keras.utils.set_random_seed(seed)
    model = build(bundle, name, task, hp)
    monitor = Monitor(title, epochs, keep_best=keep_best, grad_norm=grad_norm, verbose=verbose)
    t0 = time.time()
    hist = model.fit(shape_input(name, X), y, epochs=epochs, batch_size=batch,
                     validation_split=0.2, verbose=0, callbacks=[monitor])
    val = [float(v) for v in hist.history["val_accuracy"]]
    info = dict(model=name, task=task, seed=seed, epochs=epochs, hp=hp, device=device(),
                seconds=round(time.time() - t0), params=model.count_params(),
                val_accuracy=val, val_loss=[float(v) for v in hist.history["val_loss"]],
                train_accuracy=[float(v) for v in hist.history["accuracy"]],
                train_loss=[float(v) for v in hist.history["loss"]],
                grad_norm_max=monitor.grad_max, grad_norm_mean=monitor.grad_mean,
                nan_loss=bool(not np.isfinite(hist.history["loss"]).all()),
                best_epoch=monitor.best_epoch, collapses=collapses(val))
    return model, info, monitor.best_weights


def predict(model, name, X):
    # 循環模型在 6 GB 顯示卡上用 batch 4000 預測要約 4.5 GB，訓練完的模型還佔著記憶體時會不夠；
    # 預測結果與 batch 大小無關，所以循環模型用 1000、其他模型用 20000
    batch, steps = batching(name)
    return model.predict(shape_input(name, X), batch_size=batch if steps else batch * 4, verbose=0)


def metrics(task, y, prob, classes):
    """準確率、macro-F1、各類別 recall、混淆矩陣；二元另含攻擊偵測率與誤報率。"""
    if task == "binary":
        pred = (prob.ravel() > 0.5).astype(int)
        names = nr.BINARY_CLASSES
    else:
        pred = prob.argmax(axis=1)
        names = classes
    labels = list(range(len(names)))
    cm = confusion_matrix(y, pred, labels=labels)
    support = cm.sum(axis=1)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(len(names)), where=support > 0)
    present = [i for i in labels if support[i] > 0]
    out = dict(accuracy=float((pred == y).mean()),
               macro_f1=float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
               recall={n: float(r) for n, r in zip(names, recall)},
               support={n: int(s) for n, s in zip(names, support)},
               confusion_matrix=cm.tolist())
    if task == "binary":
        out.update(detection_rate=float(recall[0]), false_alarm_rate=float(1 - recall[1]))
    return out, pred


def evaluate(bundle, model, name, task, save_prob=None):
    """在 bundle 的所有測試集上評估。有 unseen 遮罩時另外計算已知 / 未知攻擊的偵測率。"""
    out, probs = {}, {}
    for key, t in bundle["tests"].items():
        prob = predict(model, name, t["X"])
        m, pred = metrics(task, t[task], prob, bundle["classes"])
        if task == "binary" and "unseen" in t:
            attack, unseen = t["binary"] == 0, t["unseen"]
            m["detection_seen"] = float((pred[attack & ~unseen] == 0).mean())
            m["detection_unseen"] = float((pred[unseen] == 0).mean()) if unseen.any() else None
        if task == "binary" and "fine" in t:
            fine_labels = bundle["fine_labels"]
            m["detection_by_attack"] = {
                fine_labels[f]: [float((pred[t["fine"] == f] == 0).mean()), int((t["fine"] == f).sum())]
                for f in np.unique(t["fine"]) if fine_labels[f] != "Benign"}
        out[key] = m
        probs[key] = prob.astype("float16")
    if save_prob is not None:
        np.savez_compressed(save_prob, **probs)
    return out


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except RuntimeError:            # CUDA 已經出錯時，釋放快取本身也會失敗；交給外層重啟行程
            pass


# ---------------------------------------------------------------------------- 工作佇列
def claim(path):
    """以 lock 檔認領工作；已完成或已被其他行程認領則回傳 None。"""
    if path.exists():
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    try:
        lock.open("x").close()
    except FileExistsError:
        return None
    return lock


def write_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(path)


def progress(done, total):
    return f"[已完成 {done}/{total}]"
