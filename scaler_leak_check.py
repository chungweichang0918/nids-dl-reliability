"""前處理洩漏的規模: 隨機切分條件下，標準化參數用「整個訓練檔 (含隨機測試集)」擬合，
與只用「切分後的訓練部分」擬合相比，相差多少。不訓練任何模型，只比較統計量。

輸出 results/v2/analysis/scaler_leak.json，每個資料集:
  max_dmean   max_j |mean_train - mean_full| / std_full
  max_dstd    max_j |std_train / std_full - 1|
  test_shift  隨機測試集上標準化後數值的差 |z_train - z_full| 的中位數、99 百分位與最大值
用法: python scaler_leak_check.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import nids_data
import nids_repro as nr

OUT = Path(__file__).resolve().parent / "results" / "v2" / "analysis" / "scaler_leak.json"


def stats(X, idx_train, idx_test):
    X = np.asarray(X, dtype="float64")
    m_f, s_f = X.mean(0), X.std(0)
    m_t, s_t = X[idx_train].mean(0), X[idx_train].std(0)
    ok = s_f > 0
    s_t = np.where(s_t > 0, s_t, 1.0)
    z_f = (X[idx_test][:, ok] - m_f[ok]) / s_f[ok]
    z_t = (X[idx_test][:, ok] - m_t[ok]) / s_t[ok]
    d = np.abs(z_t - z_f)
    return dict(n_features=int(ok.sum()), max_dmean=float(np.max(np.abs(m_t - m_f)[ok] / s_f[ok])),
                max_dstd=float(np.max(np.abs(s_t[ok] / s_f[ok] - 1))),
                test_shift=dict(median=float(np.median(d)), p99=float(np.quantile(d, 0.99)), max=float(d.max())))


def main():
    res = {}
    # NSL-KDD: KDDTrain+ 的數值欄位，與主實驗相同的 train_test_split(random_state=42)
    train = pd.read_csv(nr.DATA, header=None, names=nr.COLUMNS).drop(columns="difficulty")
    numeric = [c for c in train.columns if c not in nr.CATEGORICAL + ["label"]]
    idx = np.arange(len(train))
    i_tr, i_te = train_test_split(idx, test_size=0.2, random_state=nr.SEED)
    res["nsl"] = stats(train[numeric].to_numpy(), i_tr, i_te)
    # UNSW-NB15: 官方訓練集的數值欄位，nids_data._split
    import unsw
    tr = pd.read_csv(unsw.TRAIN_FILE, encoding="utf-8-sig").drop(columns="id")
    numeric = [c for c in tr.columns if c not in unsw.CATEGORICAL + ["attack_cat", "label"]]
    i_tr, i_te = nids_data._split(len(tr))
    res["unsw"] = stats(tr[numeric].to_numpy(), i_tr, i_te)
    # CIC-IDS2017 修正版與 CSE-CIC-IDS2018 (去重後)，nids_data._split
    import cicids
    import cic2018
    for name, src in (("cic", cicids), ("cic18", cic2018)):
        arrays, _ = src.load("improved")
        X = arrays["X"][~arrays["dup"]]
        i_tr, i_te = nids_data._split(len(X))
        res[name] = stats(X, i_tr, i_te)
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for k, v in res.items():
        print(f"{k:6s} max|Δmean|/σ={v['max_dmean']:.4f}  max|σ'/σ-1|={v['max_dstd']:.4f}  "
              f"測試集 |Δz| 中位數={v['test_shift']['median']:.5f}  99%={v['test_shift']['p99']:.4f}  "
              f"最大={v['test_shift']['max']:.3f}")


if __name__ == "__main__":
    main()
