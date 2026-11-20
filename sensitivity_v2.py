"""調參敏感度分析: 調參排名前 k 的設定，各用完整訓練輪數與多個種子重跑。

動機: NSL-KDD 二元 LSTM 選中的設定 (調參只訓練 1/4 輪數) 在完整 100 輪時有 2/5 個種子崩落。
這裡檢查排名第 2、3 名的設定在完整訓練下是否穩定，以區分「調參協定的問題」與「LSTM 本身的問題」。
排名第 1 的設定沿用 run_v2 已跑完的 5 個種子 (results/v2/runs)，不重跑。

用法: python sensitivity_v2.py                              (預設 nsl/binary/LSTM，前 3 名，種子 1-5)
      python sensitivity_v2.py --key nsl/multi/GRU --top 3
輸出: results/v2/sensitivity/{dataset}_{task}_{model}_r{rank}_s{seed}.json
"""
import argparse
import json

import nids_core as core
import nids_data
import tune

DIR = core.OUT / "sensitivity"


def ranked_trials(dataset, task, name):
    """依調參分數由高到低排列 (同分時試驗編號小的優先，與 tune.summarize 相同)。"""
    rows = [json.loads(f.read_text(encoding="utf-8")) for f in tune.DIR.glob(f"{dataset}_{task}_{name}_t*.json")]
    return sorted(rows, key=lambda r: (-r["score"], r["trial"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default="nsl/binary/LSTM")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    dataset, task, name = args.key.split("/")
    DIR.mkdir(parents=True, exist_ok=True)
    ranked = ranked_trials(dataset, task, name)
    assert len(ranked) >= tune.TRIALS, f"調參未完成: {len(ranked)}/{tune.TRIALS}"
    for rank, r in enumerate(ranked[:args.top], 1):
        print(f"第 {rank} 名: 試驗 {r['trial']}  調參分數 {r['score'] * 100:.2f}%  {json.dumps(r['hp'])}", flush=True)
    bundle = nids_data.load(dataset)
    print(nids_data.describe(bundle), flush=True)
    jobs = [(rank, r, s) for rank, r in enumerate(ranked[:args.top], 1) if rank > 1 for s in args.seeds]
    for i, (rank, r, seed) in enumerate(jobs):
        path = DIR / f"{dataset}_{task}_{name}_r{rank}_s{seed}.json"
        if path.exists():
            continue
        lock = core.claim(path)
        if lock is None:
            continue
        try:
            title = f"{core.progress(i, len(jobs))} 敏感度 {args.key} 第{rank}名 (試驗 {r['trial']}) 種子 {seed}"
            model, info, best_weights = core.train(bundle, name, task, seed, hp=r["hp"], title=title)
            result = {**info, **dict(dataset=dataset, condition=nids_data.MAIN_CONDITION[dataset], task=task,
                                     model=name, seed=seed, rank=rank, trial=r["trial"], tune_score=r["score"],
                                     hp=r["hp"], tests_last=core.evaluate(bundle, model, name, task))}
            model.set_weights(best_weights)
            result["tests_best"] = core.evaluate(bundle, model, name, task)
            core.write_json(path, result)
            print("  完成: " + "  ".join(f"{k}={m['accuracy'] * 100:.2f}" for k, m in result["tests_last"].items())
                  + f"  崩落 {info['collapses']} 次  ({info['seconds']}s)", flush=True)
        finally:
            lock.unlink(missing_ok=True)
            core.cleanup()
    print("完成。")


if __name__ == "__main__":
    main()
