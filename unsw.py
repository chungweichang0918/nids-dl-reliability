"""跨資料集驗證: 把同樣的六個模型與超參數搬到 UNSW-NB15。

UNSW-NB15 (Moustafa & Slay, 2015) 有官方的訓練/測試切分 (175,341 / 82,332 筆)，
測試集不是從訓練集隨機切出來的，比 NSL-KDD 的內部切分更接近實際部署情境。
前處理沿用論文流程 (標準化 + one-hot)，但 scaler 只用訓練集擬合。

可同時開多個行程 (lock 檔分工):
    .venv/Scripts/python.exe unsw.py
    .venv/Scripts/python.exe unsw.py --seeds 1 --models DNN --epochs 2
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

import nids_repro as nr
from multiseed import KeepBest, MODEL_ORDER, Progress, job_title, shape_input

DIR = nr.ROOT / "data" / "unsw"
# 鏡像站的檔名沿用官方檔名的已知錯置: 175,341 筆的才是訓練集
TRAIN_FILE, TEST_FILE = DIR / "test.csv", DIR / "train.csv"
OUTDIR = nr.OUT / "unsw"
CATEGORICAL = ["proto", "service", "state"]
TOP_PROTO = 5       # proto 有 133 種值，只保留最常見的 5 種，其餘歸為 other
SEEDS = [1, 2, 3]
RNN_SEEDS = [1]     # SimpleRNN 一次約 1 小時以上，預設只跑一個種子


def load():
    train = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig").drop(columns="id")
    test = pd.read_csv(TEST_FILE, encoding="utf-8-sig").drop(columns="id")
    assert len(train) == 175341 and len(test) == 82332, (len(train), len(test))
    numeric = [c for c in train.columns if c not in CATEGORICAL + ["attack_cat", "label"]]

    keep = {"proto": list(train["proto"].value_counts().index[:TOP_PROTO])}
    for col in ("service", "state"):
        keep[col] = sorted(train[col].unique())
    scaler = StandardScaler().fit(train[numeric].astype("float64"))
    classes = sorted(train["attack_cat"].unique())

    out = dict(classes=classes)
    columns = None
    for key, df in (("train", train), ("test", test)):
        df = df.copy()
        for col in CATEGORICAL:
            df[col] = df[col].where(df[col].isin(keep[col]), "other")
        df[numeric] = scaler.transform(df[numeric].astype("float64"))
        X = pd.get_dummies(df[numeric + CATEGORICAL], columns=CATEGORICAL)
        columns = columns or list(X.columns)
        X = X.reindex(columns=columns, fill_value=0)
        out[key] = dict(
            X=X.to_numpy(dtype="float32"),
            # 與 NSL-KDD 實驗一致: normal = 1 (positive)
            binary=(df["label"] == 0).to_numpy().astype("int64"),
            multi=df["attack_cat"].map({c: i for i, c in enumerate(classes)}).to_numpy())
    out["features"] = columns
    return out


def summarize(task, classes, y, prob):
    pred = (prob.ravel() > 0.5).astype(int) if task == "binary" else prob.argmax(axis=1)
    names = nr.BINARY_CLASSES if task == "binary" else classes
    labels = list(range(len(names)))
    report = classification_report(y, pred, labels=labels, target_names=names,
                                   zero_division=0, output_dict=True)
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(report["macro avg"]["f1-score"]),
                weighted_f1=float(report["weighted avg"]["f1-score"]),
                confusion_matrix=confusion_matrix(y, pred, labels=labels).tolist(),
                per_class={c: dict(precision=report[c]["precision"], recall=report[c]["recall"],
                                   f1=report[c]["f1-score"], support=int(report[c]["support"]))
                           for c in names})


def run(task, name, seed, data, epochs, jobs):
    path = OUTDIR / f"{task}_{name}_s{seed}.json"
    lock = path.with_suffix(".lock")
    if path.exists():
        return
    try:
        lock.open("x").close()
    except FileExistsError:
        return
    try:
        # 訓練集的 CSV 依類別排序，validation_split 取最後 20% 會全是同一類，先固定打亂
        order = np.random.default_rng(nr.SEED).permutation(len(data["train"]["X"]))
        X_train, y_train = data["train"]["X"][order], data["train"][task][order]
        X_test, y_test = data["test"]["X"], data["test"][task]
        keras.utils.set_random_seed(seed)
        if name in nr.RECURRENT_MODELS:
            batch, steps = nr.MICRO_BATCH, nr.BATCH_SIZE // nr.MICRO_BATCH
        else:
            batch, steps = nr.BATCH_SIZE, None
        model = nr.build_model(name, task, n_features=X_train.shape[1],
                               accumulation_steps=steps, n_classes=len(data["classes"]))
        keep = KeepBest()
        t0 = time.time()
        hist = model.fit(shape_input(name, X_train), y_train, epochs=epochs, batch_size=batch,
                         validation_split=0.2, verbose=0,
                         callbacks=[keep, Progress(job_title(OUTDIR, jobs, task, name, seed), epochs)])
        seconds = time.time() - t0

        last = summarize(task, data["classes"], y_test,
                         model.predict(shape_input(name, X_test), batch_size=batch, verbose=0))
        model.set_weights(keep.weights)
        best = summarize(task, data["classes"], y_test,
                         model.predict(shape_input(name, X_test), batch_size=batch, verbose=0))
        val = [float(v) for v in hist.history["val_accuracy"]]
        out = dict(task=task, model=name, seed=seed, epochs=epochs, seconds=round(seconds),
                   params=model.count_params(), n_features=int(X_train.shape[1]),
                   best_epoch=keep.epoch, val_accuracy=val,
                   train_accuracy=[float(v) for v in hist.history["accuracy"]],
                   collapses=int(sum(1 for i in range(1, len(val))
                                     if val[i] < max(val[:i]) - 0.05)),
                   last=last, best=best)
        path.write_text(json.dumps(out, indent=1))
        print(f"  完成: val={val[-1]*100:.2f} test={last['accuracy']*100:.2f} "
              f"macroF1={last['macro_f1']*100:.2f} collapses={out['collapses']} "
              f"({seconds:.0f}s)", flush=True)
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=MODEL_ORDER)
    parser.add_argument("--tasks", nargs="+", default=["binary", "multi"])
    parser.add_argument("--epochs", type=int, default=nr.EPOCHS)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    data = load()
    print(f"train={data['train']['X'].shape}  test={data['test']['X'].shape}  "
          f"classes={data['classes']}", flush=True)
    jobs = [(name, seed, task) for name in args.models
            for seed in args.seeds or (RNN_SEEDS if name == "RNN" else SEEDS)
            for task in args.tasks]
    for name, seed, task in jobs:
        run(task, name, seed, data, args.epochs, jobs)
    print(f"全部完成 ({time.strftime('%H:%M')})")


if __name__ == "__main__":
    main()
