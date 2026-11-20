"""樹模型基準線: 隨機森林 (RF) 與 XGBoost。

和深度模型完全相同的資料、切分、評估條件與種子數，並用相同的調參預算 (20 次隨機搜尋，
第 0 次為函式庫預設值，只用驗證集挑選)。驗證集與 Keras 的 validation_split=0.2 相同: 訓練資料的最後 20%。

階段:
    python baselines_v2.py --stage tune          調參 (每個資料集 x 任務 x 模型 20 次試驗)
    python baselines_v2.py --stage run           用最佳超參數在所有評估條件與種子上訓練與評估
    python baselines_v2.py --stage tune run      兩者
    python baselines_v2.py --status

只用 CPU。預設使用一半的 CPU 執行緒 (--jobs 可調整)，避免拖慢同時進行的深度模型訓練。
可同時開多個行程 (lock 檔分工)，可隨時中斷後重跑接續。
"""
import argparse
import json
import math
import os
import time
import zlib

import numpy as np

import nids_core as core
import nids_data
import nids_repro as nr

DIR = core.OUT / "baselines"
BEST = DIR / "tuned_hp.json"
MODELS = ["RF", "XGB"]
TRIALS = 20
SEEDS = {"nsl": 5, "unsw": 5, "cic": 3, "cic18": 3, "cic18b": 3}
BOTH, BINARY = ["binary", "multi"], ["binary"]
CONFIGS = [("nsl", "default", BOTH), ("unsw", "default", BOTH), ("cic", "improved", BOTH),
           ("cic", "temporal", BINARY), ("cic", "orig", BOTH), ("cic", "dedup", BOTH),
           ("cic18", "improved", BOTH), ("cic18", "temporal", BINARY),
           ("cic18b", "improved", BOTH), ("cic18b", "temporal", BINARY)]
JOBS = max(1, (os.cpu_count() or 2) // 2)


def sample(name, rng):
    pick = lambda options: options[int(rng.integers(len(options)))]
    if name == "RF":
        return dict(n_estimators=pick([100, 200, 300]), max_depth=pick([None, 10, 20, 40]),
                    min_samples_leaf=pick([1, 2, 5]), max_features=pick(["sqrt", "log2", 0.5]),
                    max_samples=pick([0.3, 0.6, 1.0]))
    return dict(n_estimators=pick([100, 200, 400]), max_depth=pick([3, 6, 9]),
                learning_rate=pick([0.03, 0.1, 0.3]), subsample=pick([0.7, 1.0]),
                colsample_bytree=pick([0.5, 0.8, 1.0]), min_child_weight=pick([1, 5]))


DEFAULT = {"RF": dict(n_estimators=100, max_depth=None, min_samples_leaf=1, max_features="sqrt",
                      max_samples=1.0),
           "XGB": dict(n_estimators=100, max_depth=6, learning_rate=0.3, subsample=1.0,
                       colsample_bytree=1.0, min_child_weight=1)}


def trial_hp(dataset, task, name, trial):
    if trial == 0:
        return dict(DEFAULT[name])
    seed = zlib.crc32(f"tree/{dataset}/{task}/{name}/{trial}".encode())
    return sample(name, np.random.default_rng(seed))


def make(name, hp, seed, n_classes):
    if name == "RF":
        from sklearn.ensemble import RandomForestClassifier
        h = dict(hp)
        h["max_samples"] = None if h["max_samples"] >= 1.0 else h["max_samples"]
        return RandomForestClassifier(**h, n_jobs=JOBS, random_state=seed)
    import xgboost as xgb
    objective = dict(objective="binary:logistic") if n_classes == 2 else \
        dict(objective="multi:softprob", num_class=n_classes)
    return xgb.XGBClassifier(**hp, **objective, tree_method="hist", device="cpu", n_jobs=JOBS,
                             random_state=seed, verbosity=0)


def fit(bundle, task, name, hp, seed):
    """在訓練資料的前 80% 訓練，回傳 (predict_fn, 驗證準確率)。predict_fn 回傳與深度模型相同格式的機率。"""
    X, y = bundle["train"]["X"], bundle["train"][task]
    split_at = int(math.floor(len(X) * 0.8))
    present = np.unique(y[:split_at])
    to_idx = {c: i for i, c in enumerate(present)}
    n_total = 2 if task == "binary" else len(bundle["classes"])
    model = make(name, hp, seed, len(present))
    model.fit(X[:split_at], np.vectorize(to_idx.get)(y[:split_at]))

    def predict(X_new):
        proba = np.concatenate([model.predict_proba(X_new[i:i + 200_000])
                                for i in range(0, len(X_new), 200_000)])
        if task == "binary":                            # 與深度模型相同: P(normal)
            return proba[:, to_idx[1]] if 1 in to_idx else np.zeros(len(proba))
        full = np.zeros((len(proba), n_total), dtype="float32")
        full[:, present] = proba
        return full

    val = float(((predict(X[split_at:]) > 0.5) == y[split_at:]).mean()) if task == "binary" \
        else float((predict(X[split_at:]).argmax(1) == y[split_at:]).mean())
    return predict, val


def evaluate(bundle, predict, task):
    """與 nids_core.evaluate 相同的指標 (含已知 / 未知攻擊與各攻擊類型偵測率)。"""
    out = {}
    for key, t in bundle["tests"].items():
        m, pred = core.metrics(task, t[task], predict(t["X"]), bundle["classes"])
        if task == "binary" and "unseen" in t:
            attack, unseen = t["binary"] == 0, t["unseen"]
            m["detection_seen"] = float((pred[attack & ~unseen] == 0).mean())
            m["detection_unseen"] = float((pred[unseen] == 0).mean()) if unseen.any() else None
        if task == "binary" and "fine" in t:
            labels = bundle["fine_labels"]
            m["detection_by_attack"] = {labels[f]: [float((pred[t["fine"] == f] == 0).mean()),
                                                    int((t["fine"] == f).sum())]
                                        for f in np.unique(t["fine"]) if labels[f] != "Benign"}
        out[key] = m
    return out


# ---------------------------------------------------------------------------- 階段
def tune_stage(datasets, models, trials):
    jobs = [(d, t, m, i) for d, c, ts in CONFIGS if c == nids_data.MAIN_CONDITION[d] and d in datasets
            for m in models for t in ts for i in range(trials)]
    path_of = lambda d, t, m, i: DIR / "tune" / f"{d}_{t}_{m}_t{i}.json"
    for dataset in dict.fromkeys(j[0] for j in jobs):
        bundle = nids_data.load(dataset)
        print(nids_data.describe(bundle), flush=True)
        for _, task, name, trial in [j for j in jobs if j[0] == dataset]:
            path = path_of(dataset, task, name, trial)
            lock = core.claim(path)
            if lock is None:
                continue
            try:
                done = sum(path_of(*j).exists() for j in jobs)
                hp = trial_hp(dataset, task, name, trial)
                t0 = time.time()
                _, val = fit(bundle, task, name, hp, seed=0)
                core.write_json(path, dict(dataset=dataset, task=task, model=name, trial=trial, hp=hp,
                                           score=val, seconds=round(time.time() - t0)))
                print(f"{core.progress(done, len(jobs))} {dataset} {task:6s} {name:4s} 試驗 {trial:2d}  "
                      f"驗證準確率 {val * 100:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
            finally:
                lock.unlink(missing_ok=True)
        del bundle
        core.cleanup()
    summarize(trials)


def summarize(trials=TRIALS, verbose=False):
    best = {}
    for f in sorted((DIR / "tune").glob("*_t*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        e = best.setdefault(f"{r['dataset']}/{r['task']}/{r['model']}", dict(n=0, score=-1, paper_score=None))
        e["n"] += 1
        if r["trial"] == 0:
            e["paper_score"] = r["score"]          # 這裡的 "paper" 指函式庫預設值
        if r["score"] > e["score"]:
            e.update(score=r["score"], trial=r["trial"], hp=r["hp"])
    if best:
        core.write_json(BEST, best)
    if verbose:
        for k, e in best.items():
            print(f"{k:22s} {e['n']:3d}/{trials}  預設 {e['paper_score'] * 100:.2f}  最佳 {e['score'] * 100:.2f}")
    return best


def run_stage(datasets, models, seeds_override):
    best = json.loads(BEST.read_text(encoding="utf-8")) if BEST.exists() else {}
    jobs = [(d, c, t, m, s) for d, c, ts in CONFIGS if d in datasets for m in models for t in ts
            for s in range(1, (seeds_override or SEEDS[d]) + 1)]
    path_of = lambda d, c, t, m, s: DIR / "runs" / f"{d}_{c}_{t}_{m}_s{s}.json"
    for key in dict.fromkeys((j[0], j[1]) for j in jobs):
        todo = [j for j in jobs if j[:2] == key and not path_of(*j).exists()]
        if not todo:
            continue
        bundle = nids_data.load(*key)
        print(nids_data.describe(bundle), flush=True)
        for dataset, condition, task, name, seed in todo:
            b = best.get(f"{dataset}/{task}/{name}")
            if b is None or b["n"] < TRIALS:
                print(f"略過 {dataset}/{task}/{name}: 調參未完成", flush=True)
                continue
            path = path_of(dataset, condition, task, name, seed)
            lock = core.claim(path)
            if lock is None:
                continue
            try:
                done = sum(path_of(*j).exists() for j in jobs)
                t0 = time.time()
                predict, val = fit(bundle, task, name, b["hp"], seed)
                tests = evaluate(bundle, predict, task)
                core.write_json(path, dict(dataset=dataset, condition=condition, task=task, model=name,
                                           seed=seed, hp=b["hp"], val_accuracy=val, tests=tests,
                                           seconds=round(time.time() - t0)))
                print(f"{core.progress(done, len(jobs))} {dataset}/{condition} {task:6s} {name:4s} s{seed}  "
                      + "  ".join(f"{k}={m['accuracy'] * 100:.2f}" for k, m in tests.items())
                      + f"  ({time.time() - t0:.0f}s)", flush=True)
            finally:
                lock.unlink(missing_ok=True)
        del bundle
        core.cleanup()


def main():
    global JOBS
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", nargs="+", default=["tune", "run"], choices=["tune", "run"])
    parser.add_argument("--datasets", nargs="+", default=["nsl", "unsw", "cic"])
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--seeds", type=int, default=None, help="覆寫種子數 (測試用)")
    parser.add_argument("--jobs", type=int, default=JOBS, help="CPU 執行緒數")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    JOBS = args.jobs
    (DIR / "tune").mkdir(parents=True, exist_ok=True)
    (DIR / "runs").mkdir(parents=True, exist_ok=True)
    if args.status:
        summarize(args.trials, verbose=True)
        print("runs:", len(list((DIR / "runs").glob("*.json"))))
        return
    if "tune" in args.stage:
        tune_stage(args.datasets, args.models, args.trials)
    if "run" in args.stage:
        run_stage(args.datasets, args.models, args.seeds)
    print("樹模型基準線完成。")


if __name__ == "__main__":
    main()
