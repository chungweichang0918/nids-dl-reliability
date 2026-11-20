"""正式實驗 (RQ1 可重現性、RQ2 訓練穩定性、RQ3 評估陷阱)。

實驗群組 (--groups)，依重要性排序:
  main        調參後的超參數，主要條件: NSL / UNSW 各 5 個種子、CIC (修正標記版) 3 個種子
  paper       原論文超參數 (學習率 1e-3、不裁剪)，同樣的資料與種子
  conditions  CIC-IDS2017 其他評估條件: 原始版、去重、時間切分 (調參後的超參數，3 個種子)
  clip        原論文超參數 + 梯度裁剪 (只有閘門循環模型)，檢驗裁剪是否消除崩落
  perm        把特徵欄位順序隨機重排後重新訓練 (二元分類，5 種排列)，檢驗「把表格當序列」的假設

每次訓練都記錄: 各測試集的指標 (最後 epoch 與最佳驗證 epoch 兩組權重)、訓練曲線、
每個 epoch 的梯度範數、崩落次數。可同時開多個行程，可隨時中斷後重跑接續。

用法:
    python run_v2.py --models CNN LSTM CNN-LSTM GRU      # 筆電
    python run_v2.py --models RNN DNN                    # Mac mini
    python run_v2.py --groups main --datasets nsl        # 只跑一部分
    python run_v2.py --status                            # 查看進度
"""
import argparse
import json

import nids_core as core
import nids_data
import nids_repro as nr
import tune

RUNS = core.OUT / "runs"
WEIGHTS = core.OUT / "models"
GROUPS = ["main", "paper", "conditions", "clip", "perm"]
# temporal: 只有 conditions 群組中的時序切分 (FT-Transformer 只跑主要條件與分布外測試時使用)
OPTIONAL_GROUPS = ["temporal"]
SEEDS = {"nsl": 5, "unsw": 5, "cic": 3, "cic18": 3, "cic18b": 3}
GATED = ["LSTM", "CNN-LSTM", "GRU"]
N_PERM = 5


def job_id(j):
    perm = f"_p{j['perm']}" if j.get("perm") else ""
    return (f"{j['dataset']}_{j['condition']}_{j['hpset']}_{j['task']}_{j['model']}"
            f"_s{j['seed']}{perm}")


def all_jobs(groups, datasets, models):
    jobs = []

    def add(group, dataset, condition, hpset, tasks, names, seeds, perm=None):
        if dataset not in datasets:
            return
        for name in names:
            if name not in models:
                continue
            for seed in seeds:
                for task in tasks:
                    jobs.append(dict(group=group, dataset=dataset, condition=condition, hpset=hpset,
                                     task=task, model=name, seed=seed, perm=perm))

    both = ["binary", "multi"]
    for group in groups:
        if group in ("main", "paper"):
            hpset = "tuned" if group == "main" else "paper"
            for ds in ("nsl", "unsw", "cic", "cic18", "cic18b"):
                n = SEEDS[ds] if group == "main" else min(SEEDS[ds], 5 if ds == "nsl" else 3)
                names = core.MODELS + ["FTT"] if group == "main" else core.MODELS   # FTT 沒有「原論文設定」
                add(group, ds, nids_data.MAIN_CONDITION[ds], hpset, both, names, range(1, n + 1))
        elif group == "conditions":
            for cond, tasks in (("temporal", ["binary"]), ("orig", both), ("dedup", both)):
                add(group, "cic", cond, "tuned", tasks, core.MODELS, range(1, SEEDS["cic"] + 1))
            for ds in ("cic18", "cic18b"):
                add(group, ds, "temporal", "tuned", ["binary"], core.MODELS, range(1, SEEDS[ds] + 1))
        elif group == "temporal":
            for ds in ("cic", "cic18", "cic18b"):
                add("conditions", ds, "temporal", "tuned", ["binary"], core.MODELS + ["FTT"],
                    range(1, SEEDS[ds] + 1))
        elif group == "clip":
            add(group, "nsl", "default", "paper_clip", both, GATED, range(1, 6))
            add(group, "unsw", "default", "paper_clip", both, GATED, range(1, 4))
        elif group == "perm":
            for p in range(1, N_PERM + 1):
                for ds in ("nsl", "unsw"):
                    add(group, ds, "default", "tuned", ["binary"], core.MODELS, [1], perm=p)
    return jobs


def get_hp(j):
    """回傳 (超參數, 錯誤訊息)。調參尚未完成的組合不執行，避免混用未調完的設定。"""
    dataset, task, name = j["dataset"], j["task"], j["model"]
    if j["hpset"] == "paper":
        return dict(nr.HP[task][name], lr=1e-3, clipnorm=None), None
    if j["hpset"] == "paper_clip":
        return dict(nr.HP[task][name], lr=1e-3, clipnorm=1.0), None
    if not tune.BEST.exists():
        return None, "找不到 tuned_hp.json"
    best = json.loads(tune.BEST.read_text(encoding="utf-8")).get(f"{dataset}/{task}/{name}")
    if best is None or best["n"] < tune.TRIALS:
        return None, f"調參未完成 ({0 if best is None else best['n']}/{tune.TRIALS})"
    return best["hp"], None


def run(j, bundle, jobs, epochs=None):
    path = RUNS / f"{job_id(j)}.json"
    if path.exists():
        return
    hp, problem = get_hp(j)
    if hp is None:
        print(f"略過 {job_id(j)}: {problem}", flush=True)
        return
    lock = core.claim(path)
    if lock is None:
        return
    try:
        done = sum((RUNS / f"{job_id(x)}.json").exists() for x in jobs)
        name, task = j["model"], j["task"]
        title = f"{core.progress(done, len(jobs))} {j['group']:10s} {job_id(j)}"
        model, info, best_weights = core.train(bundle, name, task, j["seed"], hp=hp, epochs=epochs,
                                               title=title)
        result = dict(j, **info, tests_last=core.evaluate(bundle, model, name, task))
        if j["group"] in ("main", "conditions", "paper"):
            WEIGHTS.mkdir(parents=True, exist_ok=True)
            model.save_weights(WEIGHTS / f"{job_id(j)}.weights.h5")
        model.set_weights(best_weights)
        result["tests_best"] = core.evaluate(bundle, model, name, task)
        core.write_json(path, result)
        summary = "  ".join(f"{k}={m['accuracy'] * 100:.2f}" for k, m in result["tests_last"].items())
        print(f"  完成: {summary}  崩落 {info['collapses']} 次  ({info['seconds']}s)", flush=True)
    finally:
        lock.unlink(missing_ok=True)
        core.cleanup()


def load_model(j, bundle):
    """載入已訓練的模型 (供捷徑分析與對抗實驗使用)。"""
    result = json.loads((RUNS / f"{job_id(j)}.json").read_text(encoding="utf-8"))
    model = core.build(bundle, j["model"], j["task"], result["hp"])
    model.load_weights(WEIGHTS / f"{job_id(j)}.weights.h5")
    return model, result


def status(jobs):
    table = {}
    for j in jobs:
        key = (j["group"], j["dataset"], j["condition"])
        done, total = table.get(key, (0, 0))
        table[key] = (done + (RUNS / f"{job_id(j)}.json").exists(), total + 1)
    for (group, ds, cond), (done, total) in table.items():
        print(f"{group:11s} {ds:5s} {cond:9s} {done:4d}/{total}")
    print(f"合計 {sum(d for d, _ in table.values())}/{sum(t for _, t in table.values())}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", nargs="+", default=GROUPS, choices=GROUPS + OPTIONAL_GROUPS)
    parser.add_argument("--datasets", nargs="+", default=["nsl", "unsw", "cic"])
    parser.add_argument("--models", nargs="+", default=core.MODELS)
    parser.add_argument("--epochs", type=int, default=None, help="只用於測試")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()

    tune.summarize(verbose=False)
    jobs = all_jobs(args.groups, args.datasets, args.models)
    if args.status:
        return status(jobs)

    # 依 (群組, 資料集, 條件, 排列) 分批，資料只載入一次
    batches = {}
    for j in jobs:
        batches.setdefault((j["group"], j["dataset"], j["condition"], j["perm"]), []).append(j)
    for (group, dataset, condition, perm), batch in batches.items():
        if all((RUNS / f"{job_id(j)}.json").exists() for j in batch):
            continue
        bundle = nids_data.load(dataset, condition)
        if perm:
            bundle = nids_data.permuted(bundle, perm)
        print(nids_data.describe(bundle) + (f"  [特徵排列 {perm}]" if perm else ""), flush=True)
        for j in batch:
            run(j, bundle, jobs, args.epochs)
        del bundle
        core.cleanup()
    status(jobs)


if __name__ == "__main__":
    main()
