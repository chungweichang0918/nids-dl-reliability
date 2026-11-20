"""RQ5 對抗穩健性: 符合流量限制的攻擊、不依賴梯度的攻擊、轉移攻擊與對抗訓練。

威脅模型
  unconstrained  所有數值特徵都可修改 (理論上界，與先前的實驗相同)
  realistic      只能修改攻擊者端可控制的特徵，而且
                 - 只能增加 (可以多送封包、多送位元組、拖長時間，但無法收回已送出的流量)
                 - 計數類特徵必須是非負整數 (先在原始數值空間取整數，再換回標準化空間)
                 - 不超出訓練資料出現過的範圍 (TCP 視窗大小另限制在 0–65535)
                 特徵之間的衍生關係 (例如平均值、速率) 不重新計算，這是特徵空間攻擊的已知限制
擾動上限 ε 以標準化後的標準差為單位 (L∞)。

攻擊
  pgd   白箱 PGD (Madry et al.)，隨機起點、20 步
  rs    不依賴梯度的隨機搜尋 (只查詢模型輸出)，300 次查詢；若 rs 明顯比 pgd 強，代表梯度被遮蔽
  轉移  以 DNN 產生的對抗樣本攻擊其他模型 (黑箱)

階段 (--stage)
  attack     攻擊 run_v2.py main 群組訓練好的二元模型 (NSL-KDD 與 CIC-IDS2017，種子 1–3)
  advtrain   以 realistic 威脅模型、ε=1 的 PGD-3 做對抗訓練，再評估乾淨準確率與穩健性
  transfer   轉移攻擊 (需要所有模型的權重在同一台機器上，最後在筆電執行)

用法:
    python adversarial_v2.py --stage attack --models CNN LSTM CNN-LSTM GRU      # 筆電
    python adversarial_v2.py --stage attack --models RNN DNN                    # Mac mini
    python adversarial_v2.py --stage advtrain --models ...                      # 同上分工
    python adversarial_v2.py --stage transfer                                   # 筆電，最後
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import contextlib
import json
import time

import keras
import numpy as np
import torch

import nids_core as core
import nids_data
import nids_repro as nr
import run_v2
import tune

DIR = core.OUT / "adversarial"
DATASETS = ["nsl", "cic"]
SEEDS = [1, 2, 3]
EPS = [0.1, 0.25, 0.5, 1.0, 2.0]
THREATS = ["unconstrained", "realistic"]
POOL = 2000
PGD_STEPS = 20
RS_QUERIES = 300
AT_EPS, AT_STEPS = 1.0, 3

# 攻擊者端可控制的特徵: 名稱 -> (方向 +1=只能增加 / 0=可增可減, 原始數值下限, 原始數值上限)
CONTROLLABLE = {
    "nsl": {name: (+1, 0, None) for name in [
        "duration", "src_bytes", "wrong_fragment", "urgent", "hot", "num_failed_logins",
        "num_compromised", "num_root", "num_file_creations", "num_shells", "num_access_files",
        "count", "srv_count"]},
    "cic": {**{name: (+1, 0, None) for name in [
        "Flow Duration", "Total Fwd Packets", "Total Length of Fwd Packets", "Fwd Packet Length Max",
        "Fwd IAT Total", "Fwd IAT Max", "Fwd PSH Flags", "Fwd URG Flags", "Fwd Header Length",
        "Subflow Fwd Packets", "Subflow Fwd Bytes", "act_data_pkt_fwd"]},
        "Init_Win_bytes_forward": (0, 0, 65535)},
}
CONTROLLABLE["cic18"] = CONTROLLABLE["cic"]          # 兩者使用相同的 77 個 CICFlowMeter 特徵

# consistent 威脅模型 (只用於 CICFlowMeter 特徵): realistic 的可控特徵去掉兩個 subflow 特徵
# (修正版中它們是每個 subflow 的平均值，無法由總量推得，因此固定不變)，
# 並在每次查詢模型前依下列公式重新計算衍生特徵。公式均已在 CIC-IDS2017 修正版原始資料上驗證 100% 成立。
CONSISTENT_PRIMARY = [n for n in CONTROLLABLE["cic"] if not n.startswith("Subflow")]
CONSISTENT_DERIVED = ["Fwd Packet Length Mean", "Avg Fwd Segment Size", "Packet Length Mean",
                      "Average Packet Size", "Flow Bytes/s", "Flow Packets/s", "Fwd Packets/s",
                      "Bwd Packets/s", "Down/Up Ratio", "Fwd IAT Mean", "Flow IAT Mean",
                      "Max Packet Length", "PSH Flag Count", "URG Flag Count"]
TIMING = ["Flow Duration", "Fwd IAT Total", "Fwd IAT Max"]
MIN_HEADER_BYTES = 8          # 每個新增封包至少增加的標頭位元組 (資料中每封包標頭長度的第 1 百分位數)


def dev():
    return torch.device(core.device())


def no_cudnn():
    """cuDNN 的循環層在推論模式不能反向傳播，算輸入梯度時改用一般實作。"""
    return torch.backends.cudnn.flags(enabled=False) if torch.cuda.is_available() \
        else contextlib.nullcontext()


class Constraints:
    def __init__(self, bundle, threat):
        mean, scale, numeric = nids_data.feature_scaling(bundle)
        X = bundle["train"]["X"]
        lo, hi = X.min(axis=0).astype("float64"), X.max(axis=0).astype("float64")
        n = bundle["n_features"]
        allowed, direction, integer = np.zeros(n, bool), np.zeros(n), np.zeros(n, bool)
        self.features = bundle["features"]
        if threat == "unconstrained":
            allowed = numeric.copy()
        else:
            ctrl = CONTROLLABLE[bundle["name"]]
            if threat.startswith("consistent"):
                assert bundle["name"] in ("cic", "cic18"), "consistent 只支援 CICFlowMeter 特徵"
                ctrl = {k: v for k, v in ctrl.items() if k in CONSISTENT_PRIMARY}
                if threat == "consistent_notime":       # 診斷: 不允許改變時間 (不延後封包)
                    ctrl = {k: v for k, v in ctrl.items() if k not in TIMING}
            for name, (d, raw_lo, raw_hi) in ctrl.items():
                i = bundle["features"].index(name)
                allowed[i], direction[i], integer[i] = True, d, True
                if raw_lo is not None:
                    lo[i] = max(lo[i], (raw_lo - mean[i]) / scale[i])
                if raw_hi is not None:
                    hi[i] = min(hi[i], (raw_hi - mean[i]) / scale[i])
        t = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=dev())
        self.threat, self.n_allowed = threat, int(allowed.sum())
        self.allowed, self.dir, self.integer = t(allowed, torch.bool), t(direction), t(integer, torch.bool)
        self.lo, self.hi, self.mean, self.scale = t(lo), t(hi), t(mean), t(scale)

    def project(self, x, x0, eps):
        d = torch.where(self.allowed, x - x0, torch.zeros_like(x)).clamp(-eps, eps)
        d = torch.where(self.dir > 0, d.clamp(min=0), d)
        lo, hi = torch.minimum(self.lo, x0), torch.maximum(self.hi, x0)   # 原本就超出範圍的不強迫拉回
        return torch.minimum(torch.maximum(x0 + d, lo), hi)

    def finalize(self, x, x0):
        """整數特徵在原始數值空間朝原值方向取整數，保證不超過 ε 也不違反只能增加的限制。"""
        if not bool(self.integer.any()):
            return x
        # 用 float64 換算: src_bytes 等特徵可達 10^9，float32 無法精確表示整數
        mean, scale = self.mean.double(), self.scale.double()
        r0 = torch.round(x0.double() * scale + mean)
        r = x.double() * scale + mean
        r = torch.where(self.integer, r0 + torch.trunc(r - r0), r)
        out = ((r - mean) / scale).float()
        return torch.where(self.allowed & self.integer, out, torch.where(self.allowed, x, x0))

    def random_start(self, x0, eps):
        u = (torch.rand_like(x0) * 2 - 1) * eps
        return self.project(x0 + u, x0, eps)

    def complete(self, x, x0):
        """consistent 威脅模型: 先修正主要特徵之間的可行性，再重新計算衍生特徵 (可微分)。
        其他威脅模型原樣傳回。所有修正只會把數值往原值方向拉回，因此不違反 ε 與「只能增加」。"""
        if not self.threat.startswith("consistent"):
            return x
        f = self.features
        i = {n: f.index(n) for n in CONSISTENT_PRIMARY + CONSISTENT_DERIVED
             + ["Total Backward Packets", "Total Length of Bwd Packets", "Bwd Packet Length Max",
                "Bwd PSH Flags", "Bwd URG Flags"]}
        mean, scale = self.mean.double(), self.scale.double()
        r = x.double() * scale + mean
        r0 = x0.double() * scale + mean
        col = lambda t, n: t[:, i[n]]
        nb, bb = col(r0, "Total Backward Packets"), col(r0, "Total Length of Bwd Packets")
        dur, fit, fimax = col(r, "Flow Duration"), col(r, "Fwd IAT Total"), col(r, "Fwd IAT Max")
        nf, bf, fmax = col(r, "Total Fwd Packets"), col(r, "Total Length of Fwd Packets"), col(r, "Fwd Packet Length Max")
        hdr, act, psh, urg = (col(r, "Fwd Header Length"), col(r, "act_data_pkt_fwd"),
                              col(r, "Fwd PSH Flags"), col(r, "Fwd URG Flags"))
        # 可行性: 新增的封包需要標頭位元組；IAT 不超過持續時間；平均長度不超過最大長度；計數不超過封包數
        nf = torch.minimum(nf, col(r0, "Total Fwd Packets") + torch.floor((hdr - col(r0, "Fwd Header Length")) / MIN_HEADER_BYTES))
        # 持續時間只能在流量至少有 2 個封包時拉長 (延後送出某個封包)；前向 IAT 需要至少 2 個前向封包
        dur = torch.where(nf + nb >= 2, dur, col(r0, "Flow Duration"))
        fit = torch.where(nf >= 2, fit, col(r0, "Fwd IAT Total"))
        fimax = torch.where(nf >= 2, fimax, col(r0, "Fwd IAT Max"))
        fit = torch.minimum(fit, dur)
        fimax = torch.minimum(fimax, fit)
        bf = torch.minimum(bf, nf * fmax)
        act, psh, urg = torch.minimum(act, nf), torch.minimum(psh, nf), torch.minimum(urg, nf)
        # 衍生特徵以「相對原值的比例」更新: new = 原值 × g(新) / g(原)。未擾動時完全等於原值。
        # 直接代公式會出錯: 輸入是 float32 的標準化值，極短的 Flow Duration (約 1 µs，尺度約 3e7)
        # 無法精確還原，速率類特徵會偏移。g(原) 為 0 時才直接代公式。
        p0 = {n: col(r0, n) for n in ("Total Fwd Packets", "Total Length of Fwd Packets", "Flow Duration",
                                      "Fwd IAT Total", "Fwd PSH Flags", "Fwd URG Flags")}
        nf0, bf0, dur0, fit0 = (p0["Total Fwd Packets"], p0["Total Length of Fwd Packets"],
                                p0["Flow Duration"], p0["Fwd IAT Total"])
        div = lambda a, b: a / b.clamp_min(1e-9)
        d1, d01 = dur.clamp_min(1.0), dur0.clamp_min(1.0)

        def upd(n, g_new, g_old):
            return torch.where(g_old > 0, col(r0, n) * div(g_new, g_old), g_new)

        out = r.clone()
        for n, v in (("Total Fwd Packets", nf), ("Total Length of Fwd Packets", bf), ("Fwd IAT Total", fit),
                     ("Fwd IAT Max", fimax), ("act_data_pkt_fwd", act), ("Fwd PSH Flags", psh),
                     ("Fwd URG Flags", urg),
                     ("Fwd Packet Length Mean", upd("Fwd Packet Length Mean", div(bf, nf), div(bf0, nf0))),
                     ("Avg Fwd Segment Size", upd("Avg Fwd Segment Size", div(bf, nf), div(bf0, nf0))),
                     ("Packet Length Mean", upd("Packet Length Mean", div(bf + bb, nf + nb), div(bf0 + bb, nf0 + nb))),
                     ("Average Packet Size", upd("Average Packet Size", div(bf + bb, nf + nb), div(bf0 + bb, nf0 + nb))),
                     ("Flow Bytes/s", upd("Flow Bytes/s", div(bf + bb, d1), div(bf0 + bb, d01))),
                     ("Flow Packets/s", upd("Flow Packets/s", div(nf + nb, d1), div(nf0 + nb, d01))),
                     ("Fwd Packets/s", upd("Fwd Packets/s", div(nf, d1), div(nf0, d01))),
                     ("Bwd Packets/s", upd("Bwd Packets/s", div(nb, d1), div(nb, d01))),
                     ("Down/Up Ratio", upd("Down/Up Ratio", div(nb, nf), div(nb, nf0))),
                     ("Fwd IAT Mean", upd("Fwd IAT Mean", div(fit, (nf - 1).clamp_min(1)), div(fit0, (nf0 - 1).clamp_min(1)))),
                     ("Flow IAT Mean", upd("Flow IAT Mean", div(dur, (nf + nb - 1).clamp_min(1)),
                                           div(dur0, (nf0 + nb - 1).clamp_min(1)))),
                     ("Max Packet Length", torch.maximum(col(r0, "Max Packet Length"), fmax)),
                     ("PSH Flag Count", col(r0, "PSH Flag Count") + psh - p0["Fwd PSH Flags"]),
                     ("URG Flag Count", col(r0, "URG Flag Count") + urg - p0["Fwd URG Flags"])):
            out[:, i[n]] = v
        # 結果等於原值的欄位 (包括被可行性限制拉回原值的主要特徵) 直接用原本的 float32 值，避免換算誤差。
        # 所有可修改的特徵都在 CONSISTENT_PRIMARY 中，因此其餘欄位一律維持原值。
        touched = torch.zeros(r.shape[1], dtype=torch.bool, device=r.device)
        for n in CONSISTENT_PRIMARY + CONSISTENT_DERIVED:
            touched[i[n]] = True
        res = ((out - mean) / scale).float()
        same = (out - r0).abs() <= 1e-9 * (r0.abs() + 1)
        return torch.where(touched & ~same, res, x0)


def p_normal(model, name, x):
    if name in nr.SEQUENCE_MODELS:
        x = x.unsqueeze(-1)
    return model(x, training=False).reshape(-1)


def default_chunk(name, other):
    """循環模型一次只送 1000 筆，避免 6 GB 顯示卡記憶體不足 (結果與 chunk 大小無關)。"""
    return 1000 if name in nr.MICRO_BATCH_MODELS else other


def predict_np(model, name, X, chunk=None):
    chunk = chunk or default_chunk(name, 20000)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), chunk):
            x = torch.as_tensor(np.asarray(X[i:i + chunk], "float32"), device=dev())
            out.append(p_normal(model, name, x).cpu().numpy())
    return np.concatenate(out)


def pgd(model, name, X0, cons, eps, steps=PGD_STEPS, chunk=1000, random_start=True):
    """讓 P(normal) 變大的 L∞ PGD，回傳 numpy 對抗樣本。"""
    out = []
    with no_cudnn():
        for i in range(0, len(X0), chunk):
            x0 = torch.as_tensor(X0[i:i + chunk], device=dev())
            x = cons.random_start(x0, eps) if random_start else x0.clone()
            for _ in range(steps):
                x.requires_grad_(True)
                loss = torch.log(p_normal(model, name, cons.complete(x, x0)).clamp_min(1e-12)).sum()
                g, = torch.autograd.grad(loss, x)
                x = cons.project(x.detach() + (2.5 * eps / steps) * g.sign(), x0, eps)
            out.append(cons.complete(cons.finalize(x.detach(), x0), x0).cpu().numpy())
    return np.concatenate(out)


def random_search(model, name, X0, cons, eps, queries=RS_QUERIES, chunk=None, seed=0):
    """不依賴梯度的攻擊: 每次隨機挑約 30% 的可修改特徵往 ±ε/2 方向試，P(normal) 變大就保留。"""
    chunk = chunk or default_chunk(name, 2000)
    gen = torch.Generator(device=dev()).manual_seed(seed)
    out = []
    with torch.no_grad():
        for i in range(0, len(X0), chunk):
            x0 = torch.as_tensor(X0[i:i + chunk], device=dev())
            best = cons.finalize(cons.random_start(x0, eps), x0)        # 只含主要特徵的擾動
            best_p = p_normal(model, name, cons.complete(best, x0))
            for _ in range(queries):
                todo = best_p <= 0.5
                if not bool(todo.any()):
                    break
                pick = (torch.rand(best.shape, generator=gen, device=dev()) < 0.3) & cons.allowed
                step = torch.where(torch.rand(best.shape, generator=gen, device=dev()) < 0.5, -1.0, 1.0)
                cand = cons.project(best + pick * step * eps / 2, x0, eps)
                cand = cons.finalize(cand, x0)
                p = p_normal(model, name, cons.complete(cand, x0))
                better = (p > best_p) & todo
                best = torch.where(better[:, None], cand, best)
                best_p = torch.where(better, p, best_p)
            out.append(cons.complete(best, x0).cpu().numpy())
    return np.concatenate(out)


def grad_norm(model, name, X0, cons, chunk=1000):
    """攻擊樣本上 log P(normal) 對可修改特徵的平均梯度範數 (梯度遮蔽的診斷)。"""
    norms = []
    with no_cudnn():
        for i in range(0, len(X0), chunk):
            x = torch.as_tensor(X0[i:i + chunk], device=dev()).requires_grad_(True)
            loss = torch.log(p_normal(model, name, x).clamp_min(1e-12)).sum()
            g, = torch.autograd.grad(loss, x)
            norms.append((g * cons.allowed).norm(dim=1).cpu().numpy())
    return float(np.concatenate(norms).mean())


def make_pool(model, name, bundle, seed, n=POOL):
    """測試集 (隨機切分) 中、模型正確偵測到的攻擊，固定抽 n 筆。"""
    t = bundle["tests"]["random"]
    idx = np.flatnonzero(t["binary"] == 0)
    detected = idx[predict_np(model, name, t["X"][idx]) <= 0.5]
    pick = np.random.default_rng(seed).choice(detected, min(n, len(detected)), replace=False)
    return t["X"][pick], len(detected) / len(idx)


def robustness(model, name, bundle, seed, threats=THREATS, eps_list=EPS):
    pool, clean_detect = make_pool(model, name, bundle, seed)
    out = dict(pool=len(pool), clean_detection=clean_detect, eps=eps_list, pgd={}, rs={}, grad_norm={})
    if len(pool) == 0:                       # 模型完全偵測不到攻擊 (例如訓練失敗)，沒有可攻擊的樣本
        print("    模型沒有偵測到任何攻擊，無法進行攻擊實驗", flush=True)
        return out
    for threat in threats:
        cons = Constraints(bundle, threat)
        out["grad_norm"][threat] = grad_norm(model, name, pool, cons)
        for attack, fn in (("pgd", pgd), ("rs", random_search)):
            rates = []
            for eps in eps_list:
                adv = fn(model, name, pool, cons, eps)
                rates.append(float((predict_np(model, name, adv) > 0.5).mean()))
            out[attack][threat] = rates
        print(f"    {threat:13s} ({cons.n_allowed} 個特徵)  " + "  ".join(
            f"ε={e:g}: pgd {p * 100:5.1f}% / rs {r * 100:5.1f}%"
            for e, p, r in zip(eps_list, out["pgd"][threat], out["rs"][threat])), flush=True)
    return out


# ---------------------------------------------------------------------------- 階段
def job(dataset, name, seed):
    return dict(group="main", dataset=dataset, condition=nids_data.MAIN_CONDITION[dataset],
                hpset="tuned", task="binary", model=name, seed=seed, perm=None)


def stage_attack(bundle, models, seeds, n_jobs):
    for name in models:
        for seed in seeds:
            path = DIR / "attack" / f"{bundle['name']}_{name}_s{seed}.json"
            if not (run_v2.WEIGHTS / f"{run_v2.job_id(job(bundle['name'], name, seed))}.weights.h5").exists():
                print(f"略過 {path.stem}: 尚未有 run_v2.py 的模型權重", flush=True)
                continue
            lock = core.claim(path)
            if lock is None:
                continue
            try:
                done = len(list((DIR / "attack").glob("*.json")))
                print(f"{core.progress(done, n_jobs)} 攻擊 {path.stem} ({time.strftime('%H:%M')})", flush=True)
                model, _ = run_v2.load_model(job(bundle["name"], name, seed), bundle)
                t0 = time.time()
                result = dict(dataset=bundle["name"], model=name, seed=seed,
                              **robustness(model, name, bundle, seed))
                result["seconds"] = round(time.time() - t0)
                core.write_json(path, result)
            finally:
                lock.unlink(missing_ok=True)
                core.cleanup()


def adversarial_training(bundle, name, seed, hp, epochs, title, threat="realistic"):
    """攻擊類樣本換成指定威脅模型下的 PGD-3 對抗樣本後訓練 (正常流量不變)。"""
    keras.utils.set_random_seed(seed)
    model = core.build(bundle, name, "binary", hp)
    batch, _ = core.batching(name)
    X, y = bundle["train"]["X"], bundle["train"]["binary"]
    n_val = int(len(X) * 0.2)                         # 與 validation_split=0.2 相同: 最後 20%
    X_tr, y_tr, X_val, y_val = X[:-n_val], y[:-n_val], X[-n_val:], y[-n_val:]
    cons = Constraints(bundle, threat)
    rng = np.random.default_rng(seed)
    val_acc, t0 = [], time.time()
    print(f"{title} 開始對抗訓練 ({time.strftime('%H:%M')}, {core.device()})", flush=True)
    for epoch in range(epochs):
        order = rng.permutation(len(X_tr))
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            xb, yb = X_tr[idx].copy(), y_tr[idx]
            attack = yb == 0
            if attack.any():
                xb[attack] = pgd(model, name, xb[attack], cons, AT_EPS, steps=AT_STEPS, chunk=batch)
            model.train_on_batch(core.shape_input(name, xb), yb)
        val_acc.append(float(((predict_np(model, name, X_val) > 0.5) == y_val).mean()))
        used = time.time() - t0
        print(f"\r  epoch {epoch + 1:3d}/{epochs}  val_acc={val_acc[-1] * 100:6.2f}%  "
              f"已用 {int(used) // 60}:{int(used) % 60:02d}  "
              f"剩餘約 {int(used / (epoch + 1) * (epochs - epoch - 1)) // 60} 分   ", end="", flush=True)
    print()
    return model, dict(val_accuracy=val_acc, collapses=core.collapses(val_acc),
                       seconds=round(time.time() - t0))


def stage_advtrain(bundle, models, seeds, n_jobs, epochs=None):
    epochs = epochs or nids_data.EPOCHS[bundle["name"]]
    for name in models:
        for seed in seeds:
            path = DIR / "advtrain" / f"{bundle['name']}_{name}_s{seed}.json"
            try:
                hp = tune.best_hp(bundle["name"], "binary", name)
            except (FileNotFoundError, KeyError):
                print(f"略過 {path.stem}: 調參尚未完成", flush=True)
                continue
            lock = core.claim(path)
            if lock is None:
                continue
            try:
                done = len(list((DIR / "advtrain").glob("*.json")))
                title = f"{core.progress(done, n_jobs)} 對抗訓練 {path.stem}"
                model, info = adversarial_training(bundle, name, seed, hp, epochs, title)
                result = dict(dataset=bundle["name"], model=name, seed=seed, hp=hp, eps=AT_EPS,
                              steps=AT_STEPS, epochs=epochs, **info,
                              tests=core.evaluate(bundle, model, name, "binary"),
                              robustness=robustness(model, name, bundle, seed, threats=["realistic"]))
                model.save_weights(path.with_suffix(".weights.h5"))
                core.write_json(path, result)
                print(f"  完成: 乾淨準確率 " + "  ".join(
                    f"{k}={m['accuracy'] * 100:.2f}" for k, m in result["tests"].items()), flush=True)
            finally:
                lock.unlink(missing_ok=True)
                core.cleanup()


def stage_advtrain_consistent(bundle, models, seeds, epochs=None):
    """以 consistent 威脅模型 (ε=1、PGD-3) 做對抗訓練，再用三種威脅模型評估。只用於 CIC 資料集。"""
    out_dir = DIR / "advtrain_consistent"
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = epochs or nids_data.EPOCHS[bundle["name"]]
    for name in models:
        for seed in seeds:
            path = out_dir / f"{bundle['name']}_{name}_s{seed}.json"
            if path.exists():
                continue
            try:
                hp = tune.best_hp(bundle["name"], "binary", name)
            except (FileNotFoundError, KeyError):
                print(f"略過 {path.stem}: 找不到調參結果", flush=True)
                continue
            lock = core.claim(path)
            if lock is None:
                continue
            try:
                title = f"一致性對抗訓練 {path.stem}"
                model, info = adversarial_training(bundle, name, seed, hp, epochs, title, threat="consistent")
                result = dict(dataset=bundle["name"], model=name, seed=seed, hp=hp, eps=AT_EPS, steps=AT_STEPS,
                              epochs=epochs, threat="consistent", **info,
                              tests=core.evaluate(bundle, model, name, "binary"),
                              robustness=robustness(model, name, bundle, seed,
                                                    threats=["realistic", "consistent", "consistent_notime"]))
                model.save_weights(path.with_suffix(".weights.h5"))
                core.write_json(path, result)
                print(f"  完成: 乾淨準確率 " + "  ".join(
                    f"{k}={m['accuracy'] * 100:.2f}" for k, m in result["tests"].items()), flush=True)
            finally:
                lock.unlink(missing_ok=True)
                core.cleanup()


def stage_consistent(bundle, models, seeds):
    """consistent 威脅模型 (衍生特徵重新計算) 攻擊標準訓練與對抗訓練後的模型。"""
    out_dir = DIR / "consistent"
    out_dir.mkdir(parents=True, exist_ok=True)
    cons = Constraints(bundle, "consistent")
    for name in models:
        for seed in seeds:
            path = out_dir / f"{bundle['name']}_{name}_s{seed}.json"
            if path.exists():
                continue
            lock = core.claim(path)
            if lock is None:
                continue
            try:
                result = dict(dataset=bundle["name"], model=name, seed=seed, eps=EPS,
                              n_primary=cons.n_allowed, derived=CONSISTENT_DERIVED)
                std, _ = run_v2.load_model(job(bundle["name"], name, seed), bundle)
                pool, _ = make_pool(std, name, bundle, seed)
                x0 = torch.as_tensor(pool, device=dev())
                # 自我檢查: 未擾動的紀錄重新計算衍生特徵後應與原值相同
                result["self_check_max_abs"] = float((cons.complete(x0, x0) - x0).abs().max())
                for label, weights in (("standard", None),
                                       ("advtrained", DIR / "advtrain" / f"{bundle['name']}_{name}_s{seed}.weights.h5")):
                    if weights is None:
                        model = std
                    elif weights.exists():
                        hp = json.loads(weights.with_suffix("").with_suffix(".json").read_text(encoding="utf-8"))["hp"]
                        model = core.build(bundle, name, "binary", hp)
                        model.load_weights(weights)
                    else:
                        continue
                    r = robustness(model, name, bundle, seed, threats=["consistent", "consistent_notime"])
                    result[label] = dict(pool=r["pool"], pgd=r["pgd"].get("consistent"),
                                         rs=r["rs"].get("consistent"),
                                         pgd_notime=r["pgd"].get("consistent_notime"),
                                         rs_notime=r["rs"].get("consistent_notime"))
                core.write_json(path, result)
                print(f"  {path.stem}: 自我檢查 {result['self_check_max_abs']:.2e}", flush=True)
            finally:
                lock.unlink(missing_ok=True)
                core.cleanup()


def stage_transfer(bundle, seeds):
    """以 DNN 產生的對抗樣本攻擊其他五個模型 (同一個種子)。"""
    for seed in seeds:
        path = DIR / "transfer" / f"{bundle['name']}_s{seed}.json"
        lock = core.claim(path)
        if lock is None:
            continue
        try:
            models = {}
            for name in core.MODELS:
                try:
                    models[name] = run_v2.load_model(job(bundle["name"], name, seed), bundle)[0]
                except FileNotFoundError:
                    print(f"  缺少 {name} 種子 {seed} 的模型，略過", flush=True)
            if "DNN" not in models:
                continue
            pool, _ = make_pool(models["DNN"], "DNN", bundle, seed, n=POOL * 2)
            result = dict(dataset=bundle["name"], seed=seed, eps=EPS, rates={})
            for threat in THREATS:
                cons = Constraints(bundle, threat)
                advs = [pgd(models["DNN"], "DNN", pool, cons, eps) for eps in EPS]
                for name, model in models.items():
                    detected = predict_np(model, name, pool) <= 0.5     # 目標模型原本就偵測到的
                    rates = [float((predict_np(model, name, a[detected]) > 0.5).mean()) for a in advs]
                    result["rates"].setdefault(threat, {})[name] = rates
                print(f"  [s{seed}] {threat}: " + "  ".join(
                    f"{n} {r[EPS.index(1.0)] * 100:.1f}%" for n, r in result["rates"][threat].items())
                    + "  (ε=1)", flush=True)
            core.write_json(path, result)
        finally:
            lock.unlink(missing_ok=True)
            core.cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=["attack", "advtrain", "transfer", "consistent", "advtrain_consistent"])
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--models", nargs="+", default=core.MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--epochs", type=int, default=None, help="只用於測試")
    args = parser.parse_args()
    tune.summarize(verbose=False)
    for sub in ("attack", "advtrain", "transfer"):
        (DIR / sub).mkdir(parents=True, exist_ok=True)
    n_jobs = len(args.datasets) * len(args.models) * len(args.seeds)
    for dataset in args.datasets:
        bundle = nids_data.load(dataset)
        print(nids_data.describe(bundle), flush=True)
        if args.stage == "attack":
            stage_attack(bundle, args.models, args.seeds, n_jobs)
        elif args.stage == "advtrain":
            stage_advtrain(bundle, args.models, args.seeds, n_jobs, args.epochs)
        elif args.stage == "consistent":
            if bundle["name"] in ("cic", "cic18"):
                stage_consistent(bundle, args.models, args.seeds)
        elif args.stage == "advtrain_consistent":
            if bundle["name"] in ("cic", "cic18"):
                stage_advtrain_consistent(bundle, args.models, args.seeds, args.epochs)
        else:
            stage_transfer(bundle, args.seeds)
        del bundle
        core.cleanup()
    print("完成。")


if __name__ == "__main__":
    main()
