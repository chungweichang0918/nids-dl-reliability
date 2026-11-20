"""CSE-CIC-IDS2018 (修正版, Liu/Engelen et al., IEEE CNS 2022) 前處理: 分層抽樣成與 CIC-IDS2017 同一組 77 個特徵。

完整資料有 6,320 萬筆 (CIC-IDS2017 修正版的 30 倍)，無法以相同協定訓練，因此依「方案 A」抽樣:
  - 每一種細分類攻擊最多保留 300,000 筆 (DoS Hulk、DDoS-HOIC 被截斷，其餘全部保留)
  - 正常流量 (含 Attempted，依作者建議視為正常) 保留 2,200,000 筆
  - 每一天、每一種標籤依相同比例抽樣 (分層)，固定種子，可完全重現
  - 抽樣在讀檔時串流完成 (每次讀 1,000,000 筆)，記憶體需求約 3 GB

輸出 data/cicids2018/improved.npz 與 .json，格式與 cicids.py 相同:
  X      float32 (n, 77)  未標準化的特徵，欄位順序與 CIC-IDS2017 相同
  fine   int16            細分類標籤 (對應 json 的 fine_labels)
  day    int8             0 = 02-14 ... 9 = 03-02 (依日期排序，用於依時間切分)
  dup    bool             是否為重複紀錄 (特徵與標籤完全相同，保留第一筆)

用法: python cic2018.py                                      (主要樣本，種子 2018 -> improved.npz)
      python cic2018.py --seed 2019 --variant improved_s2019   (第二份獨立樣本，檢查結論是否依賴抽樣)
"""
import json
import time
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from cicids import RENAME

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "data" / "cicids2018"
ZIP = DIR / "CSECICIDS2018_improved.zip"
FEATURES = json.loads((ROOT / "data" / "cicids2017" / "improved.json").read_text())["features"]
SEED = 2018
ATTACK_CAP = 300_000
BENIGN_TARGET = 2_200_000
CHUNK = 1_000_000

CLASSES = ["Benign", "DoS", "DDoS", "BruteForce", "WebAttack", "Botnet", "Infiltration"]


def day_key(filename):
    """'Wednesday-14-02-2018.csv' -> (2018, 2, 14)，用於依日期排序。"""
    _, d, m, y = Path(filename).stem.split("-")
    return int(y), int(m), int(d)


def clean_label(label, attempted):
    if attempted or label == "BENIGN":
        return "Benign"
    return " ".join(str(label).replace(" - ", " ").split())


def to_class(label):
    low = label.lower()
    if low == "benign":
        return "Benign"
    for key, cls in (("ddos", "DDoS"), ("dos", "DoS"), ("bruteforce", "BruteForce"), ("web attack", "WebAttack"),
                     ("botnet", "Botnet"), ("infiltration", "Infiltration")):
        if key in low:
            return cls
    raise ValueError(label)


def quotas(seed=SEED):
    """由 label_counts.json (count2018 統計) 算出每天、每種標籤要抽的筆數。"""
    counts = json.loads((DIR / "label_counts.json").read_text(encoding="utf-8"))
    per_day = {}
    for f, c in counts.items():
        if f.startswith("_"):
            continue
        d = Counter()
        for raw, n in c.items():
            att = raw.startswith("(Attempted) ")
            d[clean_label(raw.replace("(Attempted) ", ""), att)] += n
        per_day[f] = d
    total = Counter()
    for d in per_day.values():
        total.update(d)
    rate = {l: (BENIGN_TARGET / n if l == "Benign" else min(1.0, ATTACK_CAP / n)) for l, n in total.items()}
    rng = np.random.default_rng(seed)
    masks = {}
    for f in sorted(per_day, key=day_key):
        for l, n in sorted(per_day[f].items()):
            k = int(round(n * rate[l]))
            m = np.zeros(n, bool)
            m[rng.choice(n, k, replace=False) if k < n else slice(None)] = True
            masks[(f, l)] = m
    return per_day, rate, masks


def read_sampled(masks, per_day):
    inv = {v: k for k, v in RENAME.items()}
    usecols = [inv.get(c, c) for c in FEATURES] + ["Timestamp", "Label", "Attempted Category"]
    zf = zipfile.ZipFile(ZIP)
    frames = []
    for day, f in enumerate(sorted(per_day, key=day_key)):
        t0, pos, seen = time.time(), Counter(), Counter()
        parts = []
        for chunk in pd.read_csv(zf.open(f), usecols=usecols, chunksize=CHUNK, low_memory=False):
            chunk = chunk.rename(columns=RENAME)
            att = (chunk["Attempted Category"] != -1).to_numpy()
            labels = np.array([clean_label(l, a) for l, a in zip(chunk["Label"].to_numpy(), att)], dtype=object)
            keep = np.zeros(len(chunk), bool)
            for l in np.unique(labels):
                rows = np.flatnonzero(labels == l)
                keep[rows] = masks[(f, l)][pos[l]:pos[l] + len(rows)]
                pos[l] += len(rows)
            sel = chunk.loc[keep, FEATURES + ["Timestamp"]].copy()
            sel["Label"] = labels[keep]
            parts.append(sel)
            seen.update({l: int(k) for l, k in zip(*np.unique(labels[keep], return_counts=True))})
        assert dict(pos) == dict(per_day[f]), (f, dict(pos), dict(per_day[f]))   # 與統計結果完全一致
        df = pd.concat(parts, ignore_index=True).sort_values("Timestamp", kind="stable")
        df["day"] = day
        frames.append(df.drop(columns="Timestamp"))
        print(f"[{day}] {f}: 讀 {sum(pos.values()):,} 筆，抽出 {len(df):,} 筆 ({time.time() - t0:.0f}s)  "
              + "  ".join(f"{l}={n:,}" for l, n in seen.most_common()), flush=True)
    return pd.concat(frames, ignore_index=True)


def load(variant="improved"):
    """回傳 (arrays, meta)，格式與 cicids.load 相同 (binary: 1 = Benign)。"""
    path = DIR / f"{variant}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，請先執行 python cic2018.py")
    meta = json.loads((DIR / f"{variant}.json").read_text())
    z = np.load(path)
    arrays = {k: z[k] for k in z.files}
    lut = np.array([CLASSES.index(c) for c in meta["fine_to_class"]], dtype="int64")
    arrays["cls"] = lut[arrays["fine"]]
    arrays["binary"] = (arrays["cls"] == 0).astype("int64")
    return arrays, meta


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--variant", default="improved")
    args = parser.parse_args()
    t0 = time.time()
    per_day, rate, masks = quotas(args.seed)
    df = read_sampled(masks, per_day)
    n_raw = len(df)
    X = df[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    bad = ~np.isfinite(X).all(axis=1)
    X, df = X[~bad].astype("float32"), df.loc[~bad].reset_index(drop=True)
    labels = df["Label"]
    fine_labels = sorted(labels.unique(), key=lambda s: (s != "Benign", s))
    fine = labels.map({l: i for i, l in enumerate(fine_labels)}).to_numpy(dtype="int16")
    key = pd.util.hash_pandas_object(pd.DataFrame(X).assign(y=fine), index=False)
    dup = key.duplicated().to_numpy()
    days = sorted(per_day, key=day_key)
    meta = dict(variant=args.variant, source=ZIP.name, features=FEATURES, fine_labels=fine_labels,
                fine_to_class=[to_class(l) for l in fine_labels], classes=CLASSES,
                sampling=dict(seed=args.seed, attack_cap=ATTACK_CAP, benign_target=BENIGN_TARGET,
                              rate={l: float(r) for l, r in sorted(rate.items())},
                              population={l: int(sum(d[l] for d in per_day.values())) for l in rate}),
                days=[Path(f).stem for f in days],
                n_raw=int(n_raw), n_dropped_nonfinite=int(bad.sum()), n=int(len(X)), n_duplicates=int(dup.sum()),
                counts={l: int((fine == i).sum()) for i, l in enumerate(fine_labels)},
                counts_dedup={l: int(((fine == i) & ~dup).sum()) for i, l in enumerate(fine_labels)},
                per_day=[int((df["day"] == d).sum()) for d in range(len(days))])
    np.savez(DIR / f"{args.variant}.npz", X=X, fine=fine, day=df["day"].to_numpy(dtype="int8"), dup=dup)
    (DIR / f"{args.variant}.json").write_text(json.dumps(meta, indent=1))
    print(f"\n抽樣後 {n_raw:,} 筆，移除 NaN/Inf {int(bad.sum()):,} 筆，剩 {len(X):,} 筆；"
          f"重複 {int(dup.sum()):,} 筆 ({dup.mean() * 100:.1f}%)，共 {time.time() - t0:.0f}s")
    for l in fine_labels:
        print(f"  {l:42s} {meta['counts'][l]:>10,}  去重後 {meta['counts_dedup'][l]:>10,}")


if __name__ == "__main__":
    main()
