"""對抗樣本: 小幅修改攻擊流量的特徵，六個二元模型會不會把攻擊誤判為 normal？

威脅模型
  - 攻擊者只修改「攻擊」樣本，目標是被判成 normal (逃避偵測)
  - 只能動數值特徵，類別特徵 (protocol / service / flag) 不變
  - 兩種情境: all = 38 個數值特徵都能動 (理論上界)
              controllable = 只動攻擊者實際能控制的 14 個特徵 (位元組數、連線次數等)
  - 擾動上限 eps 以「標準差」為單位 (L-inf)，且不超出訓練資料的最小/最大值
攻擊方法: PGD (Madry et al., 2018)，20 步；另測試由 DNN 產生的對抗樣本轉移到其他模型的成功率。

用法: .venv/Scripts/python.exe adversarial.py        (需先跑完 multiseed.py)
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import multiseed as ms
import nids_repro as nr

OUTDIR = nr.OUT / "adversarial"
FIG = nr.OUT / "figures"
plt.rcParams.update({"font.family": ["Microsoft JhengHei", "DejaVu Sans"],
                     "axes.unicode_minus": False, "font.size": 10})
MODELS = list(nr.HP["binary"])
COLORS = dict(zip(MODELS, ["#2a78d6", "#eb6834", "#2e9e6b", "#7b4fc9", "#c9353b", "#8a8f98"]))
EPSILONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
STEPS = 20
# 攻擊者可以直接控制的特徵: 自己送出的流量大小、時間、連線頻率與登入後的行為
CONTROLLABLE = ["duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
                "num_failed_logins", "num_compromised", "num_root", "num_file_creations",
                "num_shells", "num_access_files", "count", "srv_count"]


def shape(name, x):
    return x.unsqueeze(-1) if name in nr.SEQUENCE_MODELS else x


def predict(model, name, x, batch=2000):
    with torch.no_grad():
        return torch.cat([model(shape(name, x[i:i + batch]), training=False).reshape(-1)
                          for i in range(0, len(x), batch)])


def pgd(model, name, x, mask, eps, lo, hi, steps=STEPS, batch=1000):
    """讓 P(normal) 變大的 L-inf PGD。mask 為可修改欄位 (1/0)。"""
    out = []
    # cuDNN 的 RNN 在推論模式不能反向傳播，攻擊時改用一般實作
    with torch.backends.cudnn.flags(enabled=False):
        for i in range(0, len(x), batch):
            x0 = x[i:i + batch]
            adv = x0.clone()
            for _ in range(steps):
                adv.requires_grad_(True)
                p = model(shape(name, adv), training=False).reshape(-1)
                loss = torch.log(p.clamp_min(1e-12)).sum()
                grad, = torch.autograd.grad(loss, adv)
                adv = adv.detach() + (2.5 * eps / steps) * grad.sign() * mask
                adv = torch.min(torch.max(adv, x0 - eps), x0 + eps)
                adv = torch.min(torch.max(adv, lo), hi)
            out.append(adv.detach())
    return torch.cat(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--samples", type=int, default=2000)
    args = parser.parse_args()

    data = ms.load_both()
    features = data["features"]
    numeric = [c for c in features if c in data["numeric"]]
    masks = {"all": np.isin(features, numeric).astype("float32"),
             "controllable": np.isin(features, CONTROLLABLE).astype("float32")}
    X_train, X_test, y_train, y_test = ms.split(data, "binary")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lo = torch.as_tensor(X_train.min(axis=0), device=device)
    hi = torch.as_tensor(X_train.max(axis=0), device=device)

    models = {m: ms.load_model("binary", m, args.seed) for m in MODELS}
    # 取所有模型都成功偵測的攻擊樣本，才能公平比較
    attacks = torch.as_tensor(X_test[y_test == 0], device=device)
    detected = torch.ones(len(attacks), dtype=torch.bool, device=device)
    clean = {}
    for m, model in models.items():
        hit = predict(model, m, attacks) <= 0.5
        clean[m] = float(hit.float().mean())
        detected &= hit
    pool = torch.where(detected)[0]
    pick = pool[torch.randperm(len(pool), generator=torch.Generator().manual_seed(0))
                [:args.samples].to(device)]
    x = attacks[pick]
    print(f"攻擊樣本 {len(attacks)} 筆，六個模型都偵測到的 {len(pool)} 筆，取 {len(x)} 筆", flush=True)

    result = dict(seed=args.seed, samples=int(len(x)), epsilons=EPSILONS, steps=STEPS,
                  controllable=CONTROLLABLE, clean_detection=clean, evasion={}, transfer={})
    dnn_adv = {}
    for scenario, mask in masks.items():
        mask_t = torch.as_tensor(mask, device=device)
        result["evasion"][scenario] = {}
        for m, model in models.items():
            rates = []
            for eps in EPSILONS:
                adv = pgd(model, m, x, mask_t, eps, lo, hi)
                rates.append(float((predict(model, m, adv) > 0.5).float().mean()))
                if m == "DNN" and scenario == "all":
                    dnn_adv[eps] = adv
            result["evasion"][scenario][m] = rates
            print(f"[{scenario:12s}] {m:8s} " +
                  "  ".join(f"{e:g}:{r*100:5.1f}%" for e, r in zip(EPSILONS, rates)), flush=True)

    # 轉移攻擊 (黑箱): 用 DNN 產生的對抗樣本去騙其他模型
    for m, model in models.items():
        result["transfer"][m] = [float((predict(model, m, dnn_adv[eps]) > 0.5).float().mean())
                                 for eps in EPSILONS]
        print(f"[DNN 轉移    ] {m:8s} " + "  ".join(
            f"{e:g}:{r*100:5.1f}%" for e, r in zip(EPSILONS, result["transfer"][m])), flush=True)

    # 對照組: 同樣大小的隨機擾動
    rng = torch.Generator(device=device).manual_seed(0)
    result["random"] = {}
    for m, model in models.items():
        rates = []
        for eps in EPSILONS:
            noise = (torch.randint(0, 2, x.shape, generator=rng, device=device) * 2 - 1) * eps
            adv = torch.min(torch.max(x + noise * torch.as_tensor(masks["all"], device=device), lo), hi)
            rates.append(float((predict(model, m, adv) > 0.5).float().mean()))
        result["random"][m] = rates

    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"binary_s{args.seed}.json").write_text(json.dumps(result, indent=1))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    titles = {"all": "(a) 可修改全部 38 個數值特徵",
              "controllable": f"(b) 只修改攻擊者可控的 {len(CONTROLLABLE)} 個特徵"}
    for ax, scenario in zip(axes, masks):
        for m in MODELS:
            ax.plot(EPSILONS, np.array(result["evasion"][scenario][m]) * 100, "o-",
                    color=COLORS[m], label=m, ms=4)
        ax.set_xscale("log")
        ax.set_xticks(EPSILONS, [f"{e:g}" for e in EPSILONS])
        ax.set_xlabel("擾動上限 ε（標準差）")
        ax.set_title(titles[scenario])
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("逃避偵測成功率 (%)")
    axes[1].legend(frameon=False, ncol=2)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "adversarial.png", dpi=200)
    plt.close(fig)
    print("已輸出", OUTDIR / f"binary_s{args.seed}.json")


if __name__ == "__main__":
    main()
