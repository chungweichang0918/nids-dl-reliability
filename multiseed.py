"""多種子重複實驗 + 官方 KDDTest+ 評估。

每個 (種子, 任務, 模型) 以與主實驗完全相同的設定訓練一次，同時在
  (a) KDDTrain+ 內部 20% 隨機切分測試集 (論文的評估方式)
  (b) 官方 KDDTest+ (含 17 種訓練時沒出現過的攻擊)
上評估，並儲存權重供後續 SHAP / 對抗樣本分析使用。

可同時開多個行程，以 lock 檔分工、不會重複訓練:
    .venv/Scripts/python.exe multiseed.py
    .venv/Scripts/python.exe multiseed.py --seeds 1 2 --models DNN CNN --epochs 2
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import json
import time
import warnings

warnings.filterwarnings("ignore", message="RNN module weights are not part")

import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import nids_repro as nr

TEST_FILE = nr.ROOT / "data" / "KDDTest+.txt"
OUTDIR = nr.OUT / "seeds"
MODELDIR = nr.OUT / "models"
SEEDS = [1, 2, 3, 4, 5]
# 由快到慢，快的先完成，後續分析可以先開始
MODEL_ORDER = ["DNN", "CNN", "LSTM", "CNN-LSTM", "GRU", "RNN"]


def load_both():
    """回傳 KDDTrain+ 與 KDDTest+，前處理 (scaler、one-hot 欄位) 只用 KDDTrain+ 擬合。"""
    train = pd.read_csv(nr.DATA, header=None, names=nr.COLUMNS).drop(columns="difficulty")
    test = pd.read_csv(TEST_FILE, header=None, names=nr.COLUMNS).drop(columns="difficulty")
    numeric = [c for c in train.columns if c not in nr.CATEGORICAL + ["label"]]

    scaler = StandardScaler().fit(train[numeric].astype("float64"))
    frames = {}
    for key, df in (("train", train), ("test", test)):
        df = df.copy()
        df[numeric] = scaler.transform(df[numeric].astype("float64"))
        frames[key] = pd.get_dummies(df.drop(columns="label"), columns=nr.CATEGORICAL)
    columns = list(frames["train"].columns)
    # KDDTest+ 沒出現的類別補 0；訓練時沒看過的類別值會變成全 0
    frames["test"] = frames["test"].reindex(columns=columns, fill_value=0)

    out = dict(features=columns, numeric=numeric, scaler=scaler,
               train_attacks=sorted(set(train["label"])))
    for key, df in (("train", train), ("test", test)):
        category = df["label"].map(nr.ATTACK_CATEGORY)
        assert category.notna().all(), df.loc[category.isna(), "label"].unique()
        out[key] = dict(
            X=frames[key].to_numpy(dtype="float32"),
            binary=(category == "Normal").to_numpy().astype("int64"),
            multi=category.map({c: i for i, c in enumerate(nr.MULTI_CLASSES)}).to_numpy(),
            label=df["label"].to_numpy(), category=category.to_numpy())
    assert out["train"]["X"].shape[1] == 122
    return out


def split(data, task):
    """與主實驗相同的 80/20 切分 (random_state=42)。"""
    return train_test_split(data["train"]["X"], data["train"][task],
                            test_size=0.2, random_state=nr.SEED)


class KeepBest(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best, self.epoch, self.weights = -1.0, 0, None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_accuracy"] > self.best:
            self.best, self.epoch = logs["val_accuracy"], epoch + 1
            self.weights = self.model.get_weights()


class Progress(keras.callbacks.Callback):
    """每個 epoch 在同一行更新進度: 目前 epoch、驗證準確率、已用時間與預估剩餘時間。"""

    def __init__(self, title, epochs):
        super().__init__()
        self.title, self.epochs = title, epochs

    def on_train_begin(self, logs=None):
        self.t0 = time.time()
        print(f"{self.title} 開始訓練 ({time.strftime('%H:%M')})", flush=True)

    def on_epoch_end(self, epoch, logs=None):
        used = time.time() - self.t0
        left = used / (epoch + 1) * (self.epochs - epoch - 1)
        clock = lambda s: f"{int(s) // 60:d}:{int(s) % 60:02d}"
        print(f"\r  epoch {epoch + 1:3d}/{self.epochs}  val_acc={logs['val_accuracy'] * 100:6.2f}%  "
              f"已用 {clock(used)}  剩餘約 {clock(left)}   ", end="", flush=True)

    def on_train_end(self, logs=None):
        print("\r" + " " * 78 + "\r", end="")


def job_title(folder, jobs, task, name, seed):
    """例: [已完成 12/60] s1 binary GRU"""
    done = sum((folder / f"{t}_{m}_s{s}.json").exists() for m, s, t in jobs)
    return f"[已完成 {done}/{len(jobs)}] s{seed} {task:6s} {name:8s}"


def shape_input(name, X):
    return X[..., None] if name in nr.SEQUENCE_MODELS else X


def weights_path(task, name, seed):
    return MODELDIR / f"{task}_{name}_s{seed}.weights.h5"


def load_model(task, name, seed):
    model = nr.build_model(name, task)
    model.load_weights(weights_path(task, name, seed))
    return model


def summarize(task, y, prob):
    metrics, cm, report = nr.evaluate(task, y, prob)
    macro = report["macro avg"]
    return dict(**metrics, macro_f1=float(macro["f1-score"]),
                confusion_matrix=cm.tolist(),
                per_class={c: dict(precision=report[c]["precision"], recall=report[c]["recall"],
                                   f1=report[c]["f1-score"], support=int(report[c]["support"]))
                           for c in (nr.BINARY_CLASSES if task == "binary" else nr.MULTI_CLASSES)})


def run(task, name, seed, data, epochs, jobs):
    path = OUTDIR / f"{task}_{name}_s{seed}.json"
    lock = path.with_suffix(".lock")
    if path.exists():
        return
    try:
        lock.open("x").close()      # 原子性建立；已被其他行程認領就跳過
    except FileExistsError:
        return
    try:
        X_train, X_test, y_train, y_test = split(data, task)
        keras.utils.set_random_seed(seed)
        if name in nr.RECURRENT_MODELS:
            batch, steps = nr.MICRO_BATCH, nr.BATCH_SIZE // nr.MICRO_BATCH
        else:
            batch, steps = nr.BATCH_SIZE, None
        model = nr.build_model(name, task, accumulation_steps=steps)
        keep = KeepBest()
        t0 = time.time()
        hist = model.fit(shape_input(name, X_train), y_train, epochs=epochs, batch_size=batch,
                         validation_split=0.2, verbose=0,
                         callbacks=[keep, Progress(job_title(OUTDIR, jobs, task, name, seed), epochs)])
        seconds = time.time() - t0

        sets = {"split": (X_test, y_test),
                "kddtest": (data["test"]["X"], data["test"][task])}
        result, probs = {}, {}
        for which in ("last", "best"):
            if which == "best":
                last_weights = model.get_weights()
                model.set_weights(keep.weights)
            for key, (X, y) in sets.items():
                prob = model.predict(shape_input(name, X), batch_size=batch, verbose=0)
                result[f"{key}_{which}"] = summarize(task, y, prob)
                if which == "last":
                    probs[key] = prob.astype("float32")
        model.set_weights(last_weights)

        MODELDIR.mkdir(parents=True, exist_ok=True)
        model.save_weights(weights_path(task, name, seed))
        np.savez_compressed(OUTDIR / f"{task}_{name}_s{seed}.npz", **probs)
        val = [float(v) for v in hist.history["val_accuracy"]]
        out = dict(task=task, model=name, seed=seed, epochs=epochs, seconds=round(seconds),
                   best_epoch=keep.epoch, val_accuracy=val,
                   val_loss=[float(v) for v in hist.history["val_loss"]],
                   collapses=int(sum(1 for i in range(1, len(val))
                                     if val[i] < max(val[:i]) - 0.05)),
                   **result)
        path.write_text(json.dumps(out, indent=1))
        print(f"  完成: split={out['split_last']['accuracy']*100:.2f} "
              f"kddtest={out['kddtest_last']['accuracy']*100:.2f} "
              f"collapses={out['collapses']} ({seconds:.0f}s)", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--tasks", nargs="+", default=["binary", "multi"])
    parser.add_argument("--epochs", type=int, default=nr.EPOCHS)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load_both()
    print(f"train={data['train']['X'].shape}  kddtest={data['test']['X'].shape}", flush=True)
    jobs = [(name, seed, task) for name in args.models for seed in args.seeds
            for task in args.tasks]
    for name, seed, task in jobs:
        run(task, name, seed, data, args.epochs, jobs)
    print(f"全部完成 ({time.strftime('%H:%M')})")


if __name__ == "__main__":
    main()
