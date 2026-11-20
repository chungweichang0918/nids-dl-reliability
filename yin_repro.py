"""第二個案例: 重現 Yin et al. 2017 (RNN-IDS, IEEE Access 5:21954–21961)。

論文的設定 (詳見 papers/yin2017_notes.md):
  KDDTrain+ 訓練，KDDTest+ 與 KDDTest-21 測試；one-hot 後 122 維；duration、src_bytes、dst_bytes 取 log10(x+1)，
  再 min-max 到 [0, 1]；單層 Elman RNN (sigmoid 隱藏層、softmax 輸出)，SGD；
  超參數格點: 隱藏節點 {20, 60, 80, 120, 240} x 學習率 {0.01, 0.1, 0.5}；
  二元 100 epochs (最佳 80 節點、lr 0.1)，五類 80 epochs (最佳 80 節點、lr 0.5)。
  論文以 KDDTest+ 的準確率挑選超參數。

我們的重現
  - 固定把 KDDTrain+ 的 20% 留作驗證集，模型都在其餘 80% 上訓練，同一批模型可同時比較
    「用 KDDTest+ 挑選」(論文的做法) 與「用驗證集挑選」(正確做法)
  - 論文沒寫的設定: batch size 64、不使用動量、PyTorch 預設初始化、min-max 只用訓練資料計算
  - 輸入如何組成序列論文沒有說明，我們測兩種解讀:
      record   每筆紀錄是長度 1 的序列。此時 Elman RNN 的隱藏層與一層 sigmoid MLP 完全相同，
               因此以 PyTorch 直接實作 (快)，完整跑 15 組格點 x 5 個種子 x 2 種任務
      featseq  122 個特徵當成長度 122 的序列 (Elsayed 等人的做法)。訓練成本高，
               只跑論文選出的設定 x 3 個種子 (Keras SimpleRNN, sigmoid)
  - 另以同樣的訓練資料訓練隨機森林與 XGBoost (預設參數) 作對照

用法:
    python yin_repro.py --interp record            # 約 2-3 小時 (只用 CPU，預設 4 個執行緒)
    python yin_repro.py --interp featseq            # 建議在 Mac 執行 (SimpleRNN 在 Mac CPU 較快)
    python yin_repro.py --interp trees              # 幾分鐘
    python yin_repro.py --summary
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = Path(os.environ.get("NIDS_OUT") or ROOT / "results" / "v2") / "yin"
HIDDEN = [20, 60, 80, 120, 240]
LRS = [0.01, 0.1, 0.5]
PAPER_CFG = {"binary": (80, 0.1), "multi": (80, 0.5)}
EPOCHS = {"binary": 100, "multi": 80}
PAPER_RESULT = {"binary": {"KDDTest+": 83.28, "KDDTest-21": 68.55},
                "multi": {"KDDTest+": 81.29, "KDDTest-21": 64.67}}
BATCH = 64
SEEDS = [1, 2, 3, 4, 5]
LOG_COLS = ["duration", "src_bytes", "dst_bytes"]
TESTS = {"KDDTest+": "KDDTest+.txt", "KDDTest-21": "KDDTest-21.txt"}


# ---------------------------------------------------------------------------- 資料
def load():
    import nids_repro as nr
    read = lambda f: pd.read_csv(DATA / f, header=None, names=nr.COLUMNS).drop(columns="difficulty")
    train = read("KDDTrain+.txt")
    tests = {k: read(f) for k, f in TESTS.items()}
    cat = ["protocol_type", "service", "flag"]
    numeric = [c for c in train.columns if c not in cat + ["label"]]

    def encode(df):
        df = df.copy()
        for c in LOG_COLS:
            df[c] = np.log10(df[c].astype("float64") + 1.0)
        return pd.get_dummies(df.drop(columns="label"), columns=cat)

    X_train = encode(train)
    columns = X_train.columns
    lo = X_train[numeric].min()
    span = (X_train[numeric].max() - lo).replace(0, 1.0)

    def scale(X):
        X = X.reindex(columns=columns, fill_value=0).astype("float64")
        X[numeric] = (X[numeric] - lo) / span           # min-max 只用訓練資料計算
        return X.to_numpy(dtype="float32")

    def labels(df):
        category = df["label"].map(nr.ATTACK_CATEGORY)
        assert category.notna().all()
        # 二元: 0 = normal, 1 = anomaly (論文的定義，正類為異常)
        return dict(binary=(category != "Normal").to_numpy().astype("int64"),
                    multi=category.map({c: i for i, c in enumerate(nr.MULTI_CLASSES)}).to_numpy())

    data = dict(train=dict(X=scale(X_train), **labels(train)),
                tests={k: dict(X=scale(encode(v)), **labels(v)) for k, v in tests.items()})
    assert data["train"]["X"].shape[1] == 122
    # 固定的 80/20 切分 (與種子無關)，所有設定共用
    order = np.random.default_rng(42).permutation(len(train))
    n_val = int(round(len(train) * 0.2))
    data["fit_idx"], data["val_idx"] = order[n_val:], order[:n_val]
    return data


def accuracy(prob, y):
    return float((prob.argmax(1) == y).mean() * 100)


# ---------------------------------------------------------------------------- record 解讀 (PyTorch)
def train_record(data, task, hidden, lr, seed, epochs, threads):
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    n_out = 2 if task == "binary" else 5
    # 長度 1 的 Elman RNN: h = sigmoid(W_hx x + b_h)，y = softmax(W_yh h + b_y)
    model = torch.nn.Sequential(torch.nn.Linear(122, hidden), torch.nn.Sigmoid(), torch.nn.Linear(hidden, n_out))
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    X = torch.as_tensor(data["train"]["X"][data["fit_idx"]])
    y = torch.as_tensor(data["train"][task][data["fit_idx"]])
    gen = torch.Generator().manual_seed(seed)
    val_curve = []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), BATCH):
            idx = perm[i:i + BATCH]
            opt.zero_grad()
            loss_fn(model(X[idx]), y[idx]).backward()
            opt.step()
        val_curve.append(evaluate_record(model, data, task)["validation"])
    return model, val_curve


def evaluate_record(model, data, task):
    import torch
    model.eval()
    out = {}
    with torch.no_grad():
        prob = lambda X: torch.softmax(model(torch.as_tensor(X)), 1).numpy()
        out["validation"] = accuracy(prob(data["train"]["X"][data["val_idx"]]), data["train"][task][data["val_idx"]])
        for k, t in data["tests"].items():
            out[k] = accuracy(prob(t["X"]), t[task])
    return out


# ---------------------------------------------------------------------------- featseq 解讀 (Keras SimpleRNN)
def train_featseq(data, task, hidden, lr, seed, epochs):
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    from keras import layers
    keras.utils.set_random_seed(seed)
    n_out = 2 if task == "binary" else 5
    model = keras.Sequential([keras.Input((122, 1)), layers.SimpleRNN(hidden, activation="sigmoid"),
                              layers.Dense(n_out, activation="softmax")])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    X = data["train"]["X"][..., None]
    hist = model.fit(X[data["fit_idx"]], data["train"][task][data["fit_idx"]], batch_size=BATCH, epochs=epochs,
                     validation_data=(X[data["val_idx"]], data["train"][task][data["val_idx"]]), verbose=2)
    prob = lambda A: model.predict(A[..., None], batch_size=4096, verbose=0)
    out = dict(validation=accuracy(prob(data["train"]["X"][data["val_idx"]]), data["train"][task][data["val_idx"]]))
    for k, t in data["tests"].items():
        out[k] = accuracy(prob(t["X"]), t[task])
    return out, [float(v) * 100 for v in hist.history["val_accuracy"]]


# ---------------------------------------------------------------------------- 執行
def claim(path):
    if path.exists():
        return None
    lock = path.with_suffix(".lock")
    try:
        lock.open("x").close()
    except FileExistsError:
        return None
    return lock


def run(interp, data, tasks, seeds, threads, epochs_override=None):
    OUT.mkdir(parents=True, exist_ok=True)
    if interp == "record":
        jobs = [(t, h, lr, s) for t in tasks for h in HIDDEN for lr in LRS for s in seeds]
    else:
        jobs = [(t, *PAPER_CFG[t], s) for t in tasks for s in seeds]
    for n, (task, hidden, lr, seed) in enumerate(jobs, 1):
        path = OUT / f"{interp}_{task}_h{hidden}_lr{lr}_s{seed}.json"
        lock = claim(path)
        if lock is None:
            continue
        try:
            epochs = epochs_override or EPOCHS[task]
            t0 = time.time()
            if interp == "record":
                model, curve = train_record(data, task, hidden, lr, seed, epochs, threads)
                res = evaluate_record(model, data, task)
            else:
                res, curve = train_featseq(data, task, hidden, lr, seed, epochs)
            out = dict(interp=interp, task=task, hidden=hidden, lr=lr, seed=seed, epochs=epochs, batch=BATCH,
                       seconds=round(time.time() - t0), val_curve=curve, **res)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
            tmp.replace(path)
            print(f"[{n}/{len(jobs)}] {interp} {task:6s} h={hidden:3d} lr={lr:<4} s{seed}  "
                  f"val={res['validation']:.2f}  KDDTest+={res['KDDTest+']:.2f}  "
                  f"KDDTest-21={res['KDDTest-21']:.2f}  ({out['seconds']}s)", flush=True)
        finally:
            lock.unlink(missing_ok=True)


def run_trees(data, tasks, seeds):
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb
    OUT.mkdir(parents=True, exist_ok=True)
    X, fit, val = data["train"]["X"], data["fit_idx"], data["val_idx"]
    for task in tasks:
        y = data["train"][task]
        for seed in seeds:
            for name in ("RF", "XGB"):
                path = OUT / f"trees_{task}_{name}_s{seed}.json"
                if path.exists():
                    continue
                model = RandomForestClassifier(n_jobs=4, random_state=seed) if name == "RF" else \
                    xgb.XGBClassifier(tree_method="hist", n_jobs=4, random_state=seed, verbosity=0)
                model.fit(X[fit], y[fit])
                res = dict(validation=accuracy(model.predict_proba(X[val]), y[val]))
                for k, t in data["tests"].items():
                    res[k] = accuracy(model.predict_proba(t["X"]), t[task])
                path.write_text(json.dumps(dict(interp="trees", task=task, model=name, seed=seed, **res)),
                                encoding="utf-8")
                print(f"trees {task:6s} {name} s{seed}  KDDTest+={res['KDDTest+']:.2f}  "
                      f"KDDTest-21={res['KDDTest-21']:.2f}", flush=True)


# ---------------------------------------------------------------------------- 彙整
def summary():
    rows = [json.loads(p.read_text(encoding="utf-8")) for p in OUT.glob("*.json")]
    if not rows:
        print("還沒有結果。")
        return
    df = pd.DataFrame(rows)
    res = {}
    for task in ("binary", "multi"):
        rec = df[(df.interp == "record") & (df.task == task)]
        r = dict(paper=PAPER_RESULT[task])
        if len(rec):
            h, lr = PAPER_CFG[task]
            pc = rec[(rec.hidden == h) & (rec.lr == lr)]
            r["paper_config"] = {k: dict(mean=float(pc[k].mean()), sd=float(pc[k].std(ddof=1)) if len(pc) > 1 else 0.0,
                                         max=float(pc[k].max()), n=int(len(pc))) for k in TESTS}
            sel = []
            for seed, g in rec.groupby("seed"):
                if len(g) < len(HIDDEN) * len(LRS):
                    continue
                by_test = g.loc[g["KDDTest+"].idxmax()]
                by_val = g.loc[g["validation"].idxmax()]
                sel.append(dict(seed=int(seed), test_selected=float(by_test["KDDTest+"]),
                                test_selected_cfg=(int(by_test.hidden), float(by_test.lr)),
                                val_selected=float(by_val["KDDTest+"]),
                                val_selected_cfg=(int(by_val.hidden), float(by_val.lr)),
                                val_selected_21=float(by_val["KDDTest-21"]),
                                test_selected_21=float(g.loc[g["KDDTest-21"].idxmax()]["KDDTest-21"])))
            r["selection"] = sel
            if sel:
                r["selection_gap"] = float(np.mean([s["test_selected"] - s["val_selected"] for s in sel]))
            r["best_single_run"] = float(rec["KDDTest+"].max())
            r["grid_mean"] = (rec.groupby(["hidden", "lr"])["KDDTest+"].mean().round(2)
                              .reset_index().to_dict("records"))
        fs = df[(df.interp == "featseq") & (df.task == task)]
        if len(fs):
            r["featseq"] = {k: dict(mean=float(fs[k].mean()), sd=float(fs[k].std(ddof=1)) if len(fs) > 1 else 0.0,
                                    n=int(len(fs))) for k in TESTS}
        tr = df[(df.interp == "trees") & (df.task == task)]
        if len(tr):
            r["trees"] = {m: {k: float(g[k].mean()) for k in TESTS} for m, g in tr.groupby("model")}
        res[task] = r
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for task, r in res.items():
        print(f"\n== {task}  (論文: KDDTest+ {r['paper']['KDDTest+']}%, KDDTest-21 {r['paper']['KDDTest-21']}%)")
        if "paper_config" in r:
            pc = r["paper_config"]
            print(f"  論文設定 {PAPER_CFG[task]}，{pc['KDDTest+']['n']} 個種子: KDDTest+ {pc['KDDTest+']['mean']:.2f} ± "
                  f"{pc['KDDTest+']['sd']:.2f} (最高 {pc['KDDTest+']['max']:.2f})，KDDTest-21 {pc['KDDTest-21']['mean']:.2f}")
        for s in r.get("selection", []):
            print(f"  種子 {s['seed']}: 用測試集挑 {s['test_selected']:.2f} {s['test_selected_cfg']}  vs  "
                  f"用驗證集挑 {s['val_selected']:.2f} {s['val_selected_cfg']}")
        if "selection_gap" in r:
            print(f"  用測試集挑選平均多報 {r['selection_gap']:.2f} 個百分點；所有訓練中最高 {r['best_single_run']:.2f}")
        if "featseq" in r:
            print(f"  featseq 解讀: KDDTest+ {r['featseq']['KDDTest+']['mean']:.2f} ± {r['featseq']['KDDTest+']['sd']:.2f}")
        for m, v in r.get("trees", {}).items():
            print(f"  {m} (預設參數): KDDTest+ {v['KDDTest+']:.2f}，KDDTest-21 {v['KDDTest-21']:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interp", choices=["record", "featseq", "trees"])
    parser.add_argument("--tasks", nargs="+", default=["binary", "multi"])
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=None, help="只用於測試")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary or not args.interp:
        return summary()
    data = load()
    print(f"訓練 {len(data['fit_idx']):,} 筆、驗證 {len(data['val_idx']):,} 筆；"
          + "、".join(f"{k} {len(t['X']):,} 筆" for k, t in data["tests"].items()), flush=True)
    seeds = args.seeds or (SEEDS if args.interp != "featseq" else SEEDS[:3])
    if args.interp == "trees":
        run_trees(data, args.tasks, seeds)
    else:
        run(args.interp, data, args.tasks, seeds, args.threads, args.epochs)
    summary()


if __name__ == "__main__":
    main()
