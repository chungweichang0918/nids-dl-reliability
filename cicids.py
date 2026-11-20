"""CIC-IDS2017 前處理: 把原始版與修正版整理成同一組 77 個特徵的快取檔。

兩個版本
  original : UNB 發布的 MachineLearningCSV (Sharafaldin et al., 2018)，共 2,830,743 筆
  improved : Liu, Engelen 等人 (IEEE CNS 2022) 以修正後的 CICFlowMeter 重新產生並重新標記的版本
             依作者建議，"Attempted" (有攻擊意圖但流量本身無惡意行為) 一律視為 Benign

輸出 data/cicids2017/{original,improved}.npz 與 .json:
  X      float32 (n, 77)  未標準化的特徵，欄位順序兩版一致
  fine   int16            細分類標籤 (對應 json 的 fine_labels)
  day    int8             0=Mon ... 4=Fri (用於依時間切分)
  dup    bool             是否為重複紀錄 (特徵與標籤完全相同，保留第一筆)

用法: python cicids.py            (只需執行一次，約 5-10 分鐘)
"""
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DIR = ROOT / "data" / "cicids2017"
ZIPS = {"original": DIR / "original" / "MachineLearningCSV.zip",
        "improved": DIR / "improved" / "CICIDS2017_improved.zip"}
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]

# 修正版欄位名稱 -> 原始版欄位名稱 (其餘欄位名稱相同)
RENAME = {
    "Dst Port": "Destination Port", "Total Fwd Packet": "Total Fwd Packets",
    "Total Bwd packets": "Total Backward Packets",
    "Total Length of Fwd Packet": "Total Length of Fwd Packets",
    "Total Length of Bwd Packet": "Total Length of Bwd Packets",
    "Packet Length Min": "Min Packet Length", "Packet Length Max": "Max Packet Length",
    "CWR Flag Count": "CWE Flag Count",
    "Fwd Segment Size Avg": "Avg Fwd Segment Size", "Bwd Segment Size Avg": "Avg Bwd Segment Size",
    "Fwd Bytes/Bulk Avg": "Fwd Avg Bytes/Bulk", "Fwd Packet/Bulk Avg": "Fwd Avg Packets/Bulk",
    "Fwd Bulk Rate Avg": "Fwd Avg Bulk Rate", "Bwd Bytes/Bulk Avg": "Bwd Avg Bytes/Bulk",
    "Bwd Packet/Bulk Avg": "Bwd Avg Packets/Bulk", "Bwd Bulk Rate Avg": "Bwd Avg Bulk Rate",
    "FWD Init Win Bytes": "Init_Win_bytes_forward", "Bwd Init Win Bytes": "Init_Win_bytes_backward",
    "Fwd Act Data Pkts": "act_data_pkt_fwd", "Fwd Seg Size Min": "min_seg_size_forward",
}

# 細分類 -> 類別 (多元分類用)
CLASSES = ["Benign", "DoS", "DDoS", "PortScan", "BruteForce", "WebAttack", "Botnet",
           "Infiltration", "Heartbleed"]


def to_class(label):
    low = label.lower()
    if low == "benign":
        return "Benign"
    for key, cls in (("ddos", "DDoS"), ("dos", "DoS"), ("infiltration", "Infiltration"),
                     ("portscan", "PortScan"), ("patator", "BruteForce"), ("web attack", "WebAttack"),
                     ("bot", "Botnet"), ("heartbleed", "Heartbleed")):
        if key in low:
            return cls
    raise ValueError(label)


def clean_label(label):
    """統一兩個版本的標籤寫法 (原始版的 Web Attack 標籤含有亂碼字元)。"""
    label = "".join(c if c.isascii() else " " for c in str(label))
    label = " ".join(label.replace(" - ", " ").replace("-", "-").split())
    fix = {"BENIGN": "Benign", "Bot": "Botnet", "Portscan": "PortScan",
           "DoS slowloris": "DoS Slowloris", "Web Attack Sql Injection": "Web Attack SQL Injection"}
    return fix.get(label, label)


def read_original():
    zf = zipfile.ZipFile(ZIPS["original"])
    frames = []
    for info in zf.infolist():
        if not info.filename.endswith(".csv"):
            continue
        name = info.filename.split("/")[-1].lower()
        df = pd.read_csv(zf.open(info), encoding="latin-1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df.drop(columns="Fwd Header Length.1")       # 原始檔中重複的欄位
        df["day"] = next(i for i, d in enumerate(DAYS) if name.startswith(d))
        # 同一天有多個檔案時維持上午 -> 下午的順序
        df["part"] = 0 if "morning" in name else 1
        frames.append(df)
    df = pd.concat(frames, ignore_index=True).sort_values(["day", "part"], kind="stable")
    return df.drop(columns="part").reset_index(drop=True)


def read_improved(features):
    zf = zipfile.ZipFile(ZIPS["improved"])
    frames = []
    for i, day in enumerate(DAYS):
        df = pd.read_csv(zf.open(f"{day}.csv"), low_memory=False).rename(columns=RENAME)
        # 作者建議: Attempted 的流量沒有實際惡意行為，視為 Benign
        df.loc[df["Attempted Category"] != -1, "Label"] = "BENIGN"
        df = df.sort_values("Timestamp", kind="stable")
        df["day"] = i
        frames.append(df[features + ["Label", "day"]])
    return pd.concat(frames, ignore_index=True)


def finalize(df, features, variant):
    n_raw = len(df)
    X = df[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
    bad = ~np.isfinite(X).all(axis=1)          # Flow Bytes/s 等欄位含有 NaN / Infinity
    X, df = X[~bad].astype("float32"), df.loc[~bad].reset_index(drop=True)
    labels = df["Label"].map(clean_label)
    fine_labels = sorted(labels.unique(), key=lambda s: (s != "Benign", s))
    fine = labels.map({l: i for i, l in enumerate(fine_labels)}).to_numpy(dtype="int16")
    # 重複紀錄: 特徵與標籤完全相同
    key = pd.util.hash_pandas_object(pd.DataFrame(X).assign(y=fine), index=False)
    dup = key.duplicated().to_numpy()
    meta = dict(variant=variant, features=features, fine_labels=fine_labels,
                fine_to_class=[to_class(l) for l in fine_labels], classes=CLASSES,
                n_raw=int(n_raw), n_dropped_nonfinite=int(bad.sum()), n=int(len(X)),
                n_duplicates=int(dup.sum()),
                counts={l: int((fine == i).sum()) for i, l in enumerate(fine_labels)},
                counts_dedup={l: int(((fine == i) & ~dup).sum()) for i, l in enumerate(fine_labels)},
                per_day=[int((df["day"] == d).sum()) for d in range(5)])
    np.savez(DIR / f"{variant}.npz", X=X, fine=fine, day=df["day"].to_numpy(dtype="int8"), dup=dup)
    (DIR / f"{variant}.json").write_text(json.dumps(meta, indent=1))
    return meta


def load(variant):
    """回傳 (arrays, meta)。arrays 另含 cls (類別索引) 與 binary (1 = Benign，與其他資料集一致)。"""
    path = DIR / f"{variant}.npz"
    if not path.exists():
        raise FileNotFoundError(f"{path} 不存在，請先執行 python cicids.py")
    meta = json.loads((DIR / f"{variant}.json").read_text())
    z = np.load(path)
    arrays = {k: z[k] for k in z.files}
    lut = np.array([CLASSES.index(c) for c in meta["fine_to_class"]], dtype="int64")
    arrays["cls"] = lut[arrays["fine"]]
    arrays["binary"] = (arrays["cls"] == 0).astype("int64")
    return arrays, meta


def main():
    original = read_original()
    features = [c for c in original.columns if c not in ("Label", "day")]
    assert len(features) == 77, len(features)
    for variant, df in (("original", original), ("improved", None)):
        if df is None:
            df = read_improved(features)
        m = finalize(df, features, variant)
        print(f"[{variant}] 原始 {m['n_raw']:,} 筆，移除 NaN/Inf {m['n_dropped_nonfinite']:,} 筆，"
              f"剩 {m['n']:,} 筆；重複 {m['n_duplicates']:,} 筆 ({m['n_duplicates'] / m['n'] * 100:.1f}%)")
        print("  各天筆數:", dict(zip(DAYS, m["per_day"])))
        for label in m["fine_labels"]:
            print(f"  {label:28s} {m['counts'][label]:>9,}  去重後 {m['counts_dedup'][label]:>9,}")
        del df


if __name__ == "__main__":
    main()
