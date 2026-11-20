"""公平的超參數搜尋: 每個 (資料集, 任務, 模型) 用相同的試驗次數與訓練預算。

- 隨機搜尋，預設 20 次試驗；第 0 次固定為原論文的最佳超參數 (學習率 1e-3、不裁剪)
- 搜尋空間除了原論文的結構超參數，另外對所有模型開放學習率與梯度裁剪
- 每次試驗只訓練完整輪數的 1/4，以最後一個 epoch 的驗證準確率挑選 (從不使用測試集)
- 可同時開多個行程，也可隨時中斷後重跑接續

用法:
    python tune.py                                  # 全部
    python tune.py --models CNN LSTM CNN-LSTM GRU    # 筆電 (顯示卡)
    python tune.py --models RNN DNN                  # Mac mini
    python tune.py --summary                         # 只列出目前的最佳超參數
"""
import argparse
import json
import zlib

import numpy as np

import nids_core as core
import nids_data
import nids_repro as nr

DIR = core.OUT / "tune"
BEST = core.OUT / "tuned_hp.json"
TRIALS = 20
UNITS = [8, 16, 32, 50, 64, 128]
DROPOUT = [0.1, 0.2, 0.3, 0.4, 0.5]
FILTERS = [8, 16, 32, 64]
KERNEL = [3, 5, 7]
DENSE = [16, 25, 32, 50, 64]
LR = [3e-4, 1e-3, 3e-3]
CLIP = [None, 1.0]


def sample(name, rng):
    pick = lambda options: options[int(rng.integers(len(options)))]
    if name == "DNN":
        hp = dict(units_1=pick(UNITS), units_2=pick(UNITS))
    elif name in ("RNN", "LSTM", "GRU"):
        hp = dict(units_1=pick(UNITS), dropout_1=pick(DROPOUT), units_2=pick(UNITS),
                  dropout_2=pick(DROPOUT), units_3=pick(UNITS), dense_units=pick(DENSE))
    elif name == "FTT":
        # 參考 Gorishniy et al. (2021) 的搜尋空間，為了計算量取離散值並限制層數與 token 維度
        hp = dict(n_blocks=pick([1, 2, 3]), d_token=pick([64, 128, 192, 256]),
                  attention_dropout=pick([0.0, 0.1, 0.2, 0.3]), ffn_dropout=pick([0.0, 0.1, 0.2, 0.3]),
                  residual_dropout=pick([0.0, 0.1]), ffn_factor=pick([2 / 3, 4 / 3, 2.0]),
                  weight_decay=pick([1e-6, 1e-5, 1e-4]))
        hp.update(lr=pick([3e-5, 1e-4, 3e-4, 1e-3]), clipnorm=pick(CLIP))
        return hp
    elif name == "CNN":
        hp = dict(filters_1=pick(FILTERS), kernel_size_1=pick(KERNEL), dropout_1=pick(DROPOUT),
                  filters_2=pick(FILTERS), kernel_size_2=pick(KERNEL), dropout_2=pick(DROPOUT),
                  dense_units=pick(DENSE))
    else:
        hp = dict(filters=pick(FILTERS), kernel_size=pick(KERNEL), units=pick(UNITS[2:]),
                  dropout=pick(DROPOUT))
    hp.update(lr=pick(LR), clipnorm=pick(CLIP))
    return hp


def trial_hp(dataset, task, name, trial):
    if trial == 0:
        if name == "FTT":                      # FT-Transformer 原論文的預設設定 (含自己的學習率)
            return dict(nr.HP[task][name])
        return dict(nr.HP[task][name], lr=1e-3, clipnorm=None)
    seed = zlib.crc32(f"{dataset}/{task}/{name}/{trial}".encode())
    return sample(name, np.random.default_rng(seed))


def trial_path(dataset, task, name, trial):
    return DIR / f"{dataset}_{task}_{name}_t{trial}.json"


def tune_epochs(dataset):
    return max(1, nids_data.EPOCHS[dataset] // 4)


def run(bundle, task, name, trial, jobs, epochs):
    dataset = bundle["name"]
    path = trial_path(dataset, task, name, trial)
    lock = core.claim(path)
    if lock is None:
        return
    try:
        done = sum(trial_path(*j).exists() for j in jobs)
        hp = trial_hp(dataset, task, name, trial)
        title = f"{core.progress(done, len(jobs))} {dataset} {task:6s} {name:8s} 試驗 {trial:2d}"
        try:
            _, info, _ = core.train(bundle, name, task, seed=0, hp=hp, epochs=epochs, title=title,
                                    keep_best=False, grad_norm=False)
            val = info["val_accuracy"]
            score = float(val[-1]) if np.isfinite(val[-1]) and not info["nan_loss"] else -1.0
            out = dict(dataset=dataset, task=task, model=name, trial=trial, hp=hp, score=score,
                       val_accuracy=val, collapses=info["collapses"], seconds=info["seconds"],
                       epochs=epochs)
        except RuntimeError as e:                       # 例如顯示卡記憶體不足
            out = dict(dataset=dataset, task=task, model=name, trial=trial, hp=hp, score=-1.0,
                       error=str(e)[:200], epochs=epochs)
        core.write_json(path, out)
        print(f"  完成: 驗證準確率 {out['score'] * 100:.2f}%  "
              + (f"({out['seconds']}s)  " if "seconds" in out else f"失敗: {out['error'][:60]}  ")
              + json.dumps({k: v for k, v in hp.items()}), flush=True)
    finally:
        lock.unlink(missing_ok=True)
        core.cleanup()


def summarize(trials=TRIALS, verbose=True):
    """挑出每個 (資料集, 任務, 模型) 的最佳試驗，寫入 tuned_hp.json。"""
    best = {}
    for f in sorted(DIR.glob("*_t*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        key = f"{r['dataset']}/{r['task']}/{r['model']}"
        entry = best.setdefault(key, dict(n=0, score=-2.0, paper_score=None))
        entry["n"] += 1
        if r["trial"] == 0:
            entry["paper_score"] = r["score"]
        if r["score"] > entry["score"] or (r["score"] == entry["score"] and r["trial"] < entry["trial"]):
            entry.update(score=r["score"], trial=r["trial"], hp=r["hp"])
    if best:
        core.write_json(BEST, best)
    if verbose:
        print(f"{'資料集/任務/模型':28s} {'試驗數':>6s} {'原論文超參數':>12s} {'最佳':>8s}  最佳超參數")
        for key, e in best.items():
            paper = f"{e['paper_score'] * 100:.2f}" if e["paper_score"] is not None else "—"
            print(f"{key:28s} {e['n']:>4d}/{trials:<3d} {paper:>12s} {e['score'] * 100:8.2f}  "
                  f"(試驗 {e['trial']}) {json.dumps(e['hp'])}")
    return best


def best_hp(dataset, task, name):
    """回傳調參後的最佳超參數；找不到時報錯，避免不小心用到未調參的設定。"""
    best = json.loads(BEST.read_text(encoding="utf-8"))
    return best[f"{dataset}/{task}/{name}"]["hp"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["nsl", "unsw", "cic"])
    parser.add_argument("--models", nargs="+", default=core.MODELS)
    parser.add_argument("--tasks", nargs="+", default=["binary", "multi"])
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--epochs", type=int, default=None, help="只用於測試")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.summary:
        return summarize(args.trials)

    jobs = [(d, t, m, i) for d in args.datasets for m in args.models for t in args.tasks
            for i in range(args.trials)]
    for dataset in args.datasets:
        bundle = nids_data.load(dataset)
        print(nids_data.describe(bundle), flush=True)
        for name in args.models:
            for task in args.tasks:
                for trial in range(args.trials):
                    run(bundle, task, name, trial, jobs, args.epochs or tune_epochs(dataset))
        del bundle
        core.cleanup()
    summarize(args.trials)
    print("調參完成。")


if __name__ == "__main__":
    main()
