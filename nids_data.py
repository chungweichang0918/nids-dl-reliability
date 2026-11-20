"""三個資料集的統一資料層。

load(dataset, condition) 回傳同一種結構 (bundle):
    name, condition, features, classes, tasks
    train = {X, binary, multi}          已固定打亂；訓練時取最後 20% 當驗證集
    tests = {測試集名稱: {X, binary, multi, ...}}

所有資料集的二元標籤都是 1 = 正常、0 = 攻擊。

資料集與評估條件
    nsl  / default   訓練: KDDTrain+ 的 80%      測試: random (其餘 20%)、official (KDDTest+)
    unsw / default   訓練: 官方訓練集的 80%      測試: random (其餘 20%)、official (官方測試集)
    cic  / orig      原始版全部資料，隨機 80/20                     (文獻中最常見的做法)
    cic  / dedup     原始版去除重複紀錄，隨機 80/20
    cic  / improved  修正標記版 (去重)，隨機 80/20                   (調參與主要比較使用)
    cic  / temporal  修正標記版 (去重)，週一至週三訓練、週四週五測試  (只做二元分類)
    cic18 / improved CSE-CIC-IDS2018 修正版的分層抽樣 (cic2018.py，方案 A)，去重，隨機 80/20
    cic18 / temporal 同上，前五天 (02-14 至 02-21) 訓練、後五天測試  (只做二元分類)
    cic18b           第二份獨立抽樣 (種子 2019)，只用來檢查結論是否依賴抽樣 (DNN 與樹模型)

前處理沿用被檢視的原論文流程: 數值特徵標準化、類別特徵 one-hot；隨機切分的條件是「先標準化再切分」。
"""
import numpy as np
from sklearn.preprocessing import StandardScaler

import nids_repro as nr

SPLIT_SEED = 42
CONDITIONS = {"nsl": ["default"], "unsw": ["default"],
              "cic": ["orig", "dedup", "improved", "temporal"],
              "cic18": ["improved", "temporal"], "cic18b": ["improved", "temporal"]}
MAIN_CONDITION = {"nsl": "default", "unsw": "default", "cic": "improved", "cic18": "improved", "cic18b": "improved"}
# 訓練輪數: 以權重更新次數而非 epoch 數為準。CIC-IDS2017 每個 epoch 的更新次數是 NSL-KDD 的十幾倍
# CSE-CIC-IDS2018 抽樣後的訓練集是 CIC-IDS2017 修正版的 1.6 倍，12 個 epoch 的更新次數與 CIC-IDS2017 的 20 個相當
EPOCHS = {"nsl": 100, "unsw": 100, "cic": 20, "cic18": 12, "cic18b": 12}
# 依時間切分時最後一個訓練日 (含): CIC-IDS2017 週一至週三；CSE-CIC-IDS2018 前五天 (02-14 至 02-20)
TEMPORAL_LAST_TRAIN_DAY = {"cic": 2, "cic18": 4, "cic18b": 4}


def _split(n, test_size=0.2):
    """固定的隨機 80/20 切分，回傳 (訓練索引, 測試索引)，訓練索引已打亂。"""
    order = np.random.default_rng(SPLIT_SEED).permutation(n)
    n_test = int(round(n * test_size))
    return order[n_test:], order[:n_test]


def _part(X, binary, multi, idx=None, **extra):
    if idx is None:
        return dict(X=X, binary=binary, multi=multi, **extra)
    return dict(X=X[idx], binary=binary[idx], multi=multi[idx],
                **{k: v[idx] for k, v in extra.items()})


def _nsl():
    import multiseed as ms
    d = ms.load_both()
    X_train, X_test, yb_train, yb_test = ms.split(d, "binary")      # 與先前實驗完全相同的切分
    _, _, ym_train, ym_test = ms.split(d, "multi")
    t = d["test"]
    unseen = (t["binary"] == 0) & ~np.isin(t["label"], d["train_attacks"])
    return dict(
        features=d["features"], classes=nr.MULTI_CLASSES, tasks=["binary", "multi"],
        numeric=d["numeric"], scaler=d["scaler"],
        train=dict(X=X_train, binary=yb_train, multi=ym_train),
        tests={"random": dict(X=X_test, binary=yb_test, multi=ym_test),
               "official": dict(X=t["X"], binary=t["binary"], multi=t["multi"], unseen=unseen)})


def _unsw():
    import unsw
    d = unsw.load()
    tr, te = d["train"], d["test"]
    idx_train, idx_test = _split(len(tr["X"]))
    return dict(
        features=d["features"], classes=d["classes"], tasks=["binary", "multi"],
        train=_part(tr["X"], tr["binary"], tr["multi"], idx_train),
        tests={"random": _part(tr["X"], tr["binary"], tr["multi"], idx_test),
               "official": _part(te["X"], te["binary"], te["multi"])})


def _cic(condition, dataset="cic"):
    if dataset in ("cic18", "cic18b"):
        import cic2018 as source
    else:
        import cicids as source
    variant = ("improved_s2019" if dataset == "cic18b"
               else "original" if condition in ("orig", "dedup") else "improved")
    arrays, meta = source.load(variant)
    keep = np.ones(len(arrays["X"]), bool) if condition == "orig" else ~arrays["dup"]
    X, yb, ym = arrays["X"][keep], arrays["binary"][keep], arrays["cls"][keep]
    day, fine = arrays["day"][keep], arrays["fine"][keep]
    info = dict(features=meta["features"], classes=meta["classes"], fine_labels=meta["fine_labels"])

    if condition == "temporal":
        is_train = day <= TEMPORAL_LAST_TRAIN_DAY[dataset]
        scaler = StandardScaler().fit(X[is_train])                  # 只用訓練日擬合
        X = scaler.transform(X).astype("float32")
        idx_train = np.flatnonzero(is_train)
        idx_train = idx_train[np.random.default_rng(SPLIT_SEED).permutation(len(idx_train))]
        idx_test = np.flatnonzero(~is_train)
        train_fine = set(np.unique(fine[is_train]))
        unseen = (yb == 0) & ~np.isin(fine, list(train_fine))        # 訓練日沒出現過的攻擊
        return dict(**info, tasks=["binary"], scaler=scaler,
                    train=_part(X, yb, ym, idx_train),
                    tests={"temporal": _part(X, yb, ym, idx_test, fine=fine, unseen=unseen)})

    scaler = StandardScaler().fit(X)                                # 先標準化再切分 (常見做法)
    X = scaler.transform(X).astype("float32")
    idx_train, idx_test = _split(len(X))
    return dict(**info, tasks=["binary", "multi"], scaler=scaler,
                train=_part(X, yb, ym, idx_train),
                tests={"random": _part(X, yb, ym, idx_test, fine=fine)})


def load(dataset, condition=None):
    condition = condition or MAIN_CONDITION[dataset]
    assert condition in CONDITIONS[dataset], (dataset, condition)
    if dataset in ("cic", "cic18", "cic18b"):
        bundle = _cic(condition, dataset)
    else:
        bundle = {"nsl": _nsl, "unsw": _unsw}[dataset]()
    bundle.update(name=dataset, condition=condition, n_features=bundle["train"]["X"].shape[1])
    bundle["groups"] = feature_groups(dataset, bundle["features"])
    return bundle


CATEGORICAL = {"nsl": ["protocol_type", "service", "flag"], "unsw": ["proto", "service", "state"],
               "cic": [], "cic18": [], "cic18b": []}


def feature_groups(dataset, features):
    """每個欄位所屬的原始特徵名稱 (one-hot 欄位合併回原本的類別特徵)。"""
    def original(col):
        for c in CATEGORICAL[dataset]:
            if col.startswith(c + "_"):
                return c
        return col
    return [original(c) for c in features]


def feature_scaling(bundle):
    """每個欄位的 (mean, scale, 是否為標準化過的數值欄位)，用來在標準化空間與原始數值之間換算。

    one-hot 欄位 (mean=0, scale=1, numeric=False) 不做換算。只支援 nsl 與 cic。
    """
    n, features = bundle["n_features"], bundle["features"]
    mean, scale, numeric = np.zeros(n), np.ones(n), np.zeros(n, bool)
    if bundle["name"] == "nsl":
        cols = [features.index(c) for c in bundle["numeric"]]
        mean[cols], scale[cols], numeric[cols] = bundle["scaler"].mean_, bundle["scaler"].scale_, True
    elif bundle["name"] in ("cic", "cic18", "cic18b"):
        mean, scale, numeric = bundle["scaler"].mean_.copy(), bundle["scaler"].scale_.copy(), ~numeric
    else:
        raise NotImplementedError(bundle["name"])
    return mean, scale, numeric


def permuted(bundle, perm_seed):
    """回傳特徵欄位順序被隨機重排的 bundle (訓練與測試用同一個排列)。"""
    perm = np.random.default_rng(1000 + perm_seed).permutation(bundle["n_features"])
    out = dict(bundle)
    out["train"] = dict(bundle["train"], X=bundle["train"]["X"][:, perm])
    out["tests"] = {k: dict(t, X=t["X"][:, perm]) for k, t in bundle["tests"].items()}
    out["features"] = [bundle["features"][i] for i in perm]
    out["groups"] = [bundle["groups"][i] for i in perm]
    return out


def describe(bundle):
    tr = bundle["train"]
    lines = [f"{bundle['name']}/{bundle['condition']}: {bundle['n_features']} 維特徵，"
             f"訓練 {len(tr['X']):,} 筆 (正常 {tr['binary'].mean() * 100:.1f}%)"]
    for name, t in bundle["tests"].items():
        extra = f"，其中未見過的攻擊 {int(t['unseen'].sum()):,} 筆" if "unseen" in t else ""
        lines.append(f"  測試 {name}: {len(t['X']):,} 筆 (正常 {t['binary'].mean() * 100:.1f}%){extra}")
    return "\n".join(lines)


if __name__ == "__main__":
    for ds, conds in CONDITIONS.items():
        for c in conds:
            print(describe(load(ds, c)))
