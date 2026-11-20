"""重現 "A Comparative Study of Using Deep Learning Algorithms in Network
Intrusion Detection" (Elsayed et al., IEEE Access 2024)。

六個模型 (DNN / RNN / CNN / LSTM / GRU / Hybrid CNN-LSTM) x 兩種任務
(binary / multi)，超參數採用論文 Section VI 報告的 KerasTuner 最佳值。

用法:
    .venv/Scripts/python.exe nids_repro.py                  # 全部 12 個模型
    .venv/Scripts/python.exe nids_repro.py --task binary --models GRU DNN
    .venv/Scripts/python.exe nids_repro.py --epochs 5       # 快速測試
"""
import os

os.environ.setdefault("KERAS_BACKEND", "torch")

import argparse
import json
import time
import warnings
from pathlib import Path

# PyTorch cuDNN 對 Keras RNN 權重的提示，不影響結果
warnings.filterwarnings("ignore", message="RNN module weights are not part")

import numpy as np
import pandas as pd
import keras
from keras import layers
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "KDDTrain+.txt"
OUT = ROOT / "results"

SEED = 42
EPOCHS = 100        # 論文: 100 epochs
BATCH_SIZE = 5000   # 論文: batch size 5000
# 循環模型在 6GB 顯示卡上放不下 batch 5000，改成每次 1000 筆、累積 5 次梯度
# 才更新一次權重，等效於 batch 5000。
MICRO_BATCH = 1000
RECURRENT_MODELS = {"RNN", "LSTM", "GRU", "CNN-LSTM"}

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]
CATEGORICAL = ["protocol_type", "service", "flag"]

ATTACK_CATEGORY = {
    "normal": "Normal",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS", "worm": "DoS",
    # Probe
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "satan": "Probe", "mscan": "Probe", "saint": "Probe",
    # R2L
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L",
    "multihop": "R2L", "phf": "R2L", "spy": "R2L", "warezclient": "R2L",
    "warezmaster": "R2L", "sendmail": "R2L", "named": "R2L",
    "snmpgetattack": "R2L", "snmpguess": "R2L", "xlock": "R2L",
    "xsnoop": "R2L", "httptunnel": "R2L",
    # U2R
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R",
    "rootkit": "U2R", "ps": "U2R", "sqlattack": "U2R", "xterm": "U2R",
}
MULTI_CLASSES = ["DoS", "Normal", "Probe", "R2L", "U2R"]
# 論文的混淆矩陣顯示 positive 類別是 normal (LabelEncoder: abnormal=0, normal=1)
BINARY_CLASSES = ["abnormal", "normal"]

# 論文報告的數值 (%): accuracy, precision, recall, f1
PAPER = {
    "binary": {
        "DNN": (99.39, 99.45, 99.40, 99.43),
        "RNN": (99.095, 99.09, 99.21, 99.15),
        "CNN": (99.376, 99.34, 99.49, 99.42),
        "LSTM": (99.4086, 99.46, 99.43, 99.44),
        "GRU": (99.54, 99.46, 99.67, 99.56),
        "CNN-LSTM": (98.944, 98.94, 99.08, 99.01),
    },
    "multi": {
        "DNN": (99.34, 99.36, 99.33, 99.34),
        "RNN": (99.253, 99.25, 99.24, 99.25),
        "CNN": (99.385, 99.40, 99.37, 99.39),
        "LSTM": (99.39, 99.41, 99.39, 99.40),
        "GRU": (99.38, 99.39, 99.37, 99.38),
        "CNN-LSTM": (97.96, 98.01, 97.79, 97.90),
    },
}


# --------------------------------------------------------------------------
# 資料前處理 (論文 Section IV-B)
# --------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(DATA, header=None, names=COLUMNS).drop(columns="difficulty")
    numeric = [c for c in df.columns if c not in CATEGORICAL + ["label"]]

    # 1) 標準化 38 個數值特徵 (論文是先前處理、再切分)
    df[numeric] = StandardScaler().fit_transform(df[numeric].astype("float64"))
    # 2) One-hot: protocol_type(3) + service(70) + flag(11) = 84 -> 共 122 維
    X = pd.get_dummies(df.drop(columns="label"), columns=CATEGORICAL)
    X = X.to_numpy(dtype="float32")
    assert X.shape[1] == 122, X.shape

    category = df["label"].map(ATTACK_CATEGORY)
    assert category.notna().all(), df.loc[category.isna(), "label"].unique()
    y_bin = (category == "Normal").to_numpy().astype("int64")
    y_multi = category.map({c: i for i, c in enumerate(MULTI_CLASSES)}).to_numpy()
    return X, y_bin, y_multi


# --------------------------------------------------------------------------
# 模型 (論文 Section VI-B 的最佳超參數)
# --------------------------------------------------------------------------
HP = {
    "binary": {
        "DNN": dict(units_1=32, units_2=32),
        "RNN": dict(units_1=50, dropout_1=0.3, units_2=32, dropout_2=0.1,
                    units_3=8, dense_units=16),
        "CNN": dict(filters_1=64, kernel_size_1=7, dropout_1=0.1, filters_2=32,
                    kernel_size_2=7, dropout_2=0.1, dense_units=50),
        "LSTM": dict(units_1=50, dropout_1=0.2, units_2=16, dropout_2=0.1,
                     units_3=16, dense_units=32),
        "GRU": dict(units_1=128, dropout_1=0.2, units_2=128, dropout_2=0.4,
                    units_3=16, dense_units=25),
        "CNN-LSTM": dict(filters=64, kernel_size=7, units=128, dropout=0.2),
    },
    "multi": {
        "DNN": dict(units_1=64, units_2=8),
        "RNN": dict(units_1=64, dropout_1=0.1, units_2=32, dropout_2=0.2,
                    units_3=64, dense_units=25),
        "CNN": dict(filters_1=64, kernel_size_1=7, dropout_1=0.1, filters_2=32,
                    kernel_size_2=7, dropout_2=0.1, dense_units=50),
        "LSTM": dict(units_1=64, dropout_1=0.1, units_2=32, dropout_2=0.5,
                     units_3=50, dense_units=64),
        "GRU": dict(units_1=128, dropout_1=0.2, units_2=50, dropout_2=0.1,
                    units_3=25, dense_units=16),
        "CNN-LSTM": dict(filters=8, kernel_size=5, units=64, dropout=0.2),
    },
}
SEQUENCE_MODELS = {"RNN", "CNN", "LSTM", "GRU", "CNN-LSTM"}

# FT-Transformer (Gorishniy et al., NeurIPS 2021) 的預設設定，作為調參的第 0 次試驗。
# 來源: 原論文的 default configuration (3 層、token 維度 192、8 個 head、attention dropout 0.2、
# FFN dropout 0.1、FFN 倍數 4/3、AdamW lr 1e-4、weight decay 1e-5)。
FTT_DEFAULT = dict(n_blocks=3, d_token=192, attention_dropout=0.2, ffn_dropout=0.1, residual_dropout=0.0,
                   ffn_factor=4 / 3, lr=1e-4, weight_decay=1e-5, clipnorm=None)
HP["binary"]["FTT"] = dict(FTT_DEFAULT)
HP["multi"]["FTT"] = dict(FTT_DEFAULT)
# 一次放不下 batch 5000 的模型改用 1000 筆 x 累積 5 次梯度 (數學上等效)
MICRO_BATCH_MODELS = RECURRENT_MODELS | {"FTT"}


@keras.saving.register_keras_serializable("nids")
class FeatureTokenizer(layers.Layer):
    """每個 (已標準化的) 欄位 x_j 變成一個 token: x_j * W_j + b_j，最後接一個可學習的 [CLS] token。"""

    def __init__(self, d_token, **kw):
        super().__init__(**kw)
        self.d_token = d_token

    def build(self, input_shape):
        n = input_shape[-1]
        init = keras.initializers.RandomUniform(-1 / self.d_token ** 0.5, 1 / self.d_token ** 0.5)
        self.w = self.add_weight(shape=(n, self.d_token), initializer=init, name="weight")
        self.b = self.add_weight(shape=(n, self.d_token), initializer=init, name="bias")
        self.cls = self.add_weight(shape=(1, 1, self.d_token), initializer=init, name="cls")

    def call(self, x):
        tokens = keras.ops.expand_dims(x, -1) * self.w + self.b
        cls = keras.ops.broadcast_to(self.cls, (keras.ops.shape(x)[0], 1, self.d_token))
        return keras.ops.concatenate([tokens, cls], axis=1)

    def get_config(self):
        return {**super().get_config(), "d_token": self.d_token}


@keras.saving.register_keras_serializable("nids")
class ReGLU(layers.Layer):
    def call(self, x):
        a, b = keras.ops.split(x, 2, axis=-1)
        return a * keras.ops.relu(b)


def _ft_transformer(inp, hp):
    """Pre-norm Transformer；第一層不在 attention 前做 LayerNorm (與原論文相同)。輸出 [CLS] token。"""
    d = hp["d_token"]
    d_hidden = int(d * hp["ffn_factor"])
    x = FeatureTokenizer(d)(inp)
    for i in range(hp["n_blocks"]):
        h = x if i == 0 else layers.LayerNormalization()(x)
        h = layers.MultiHeadAttention(8, d // 8, dropout=hp["attention_dropout"])(h, h)
        x = x + layers.Dropout(hp["residual_dropout"])(h)
        h = layers.LayerNormalization()(x)
        h = ReGLU()(layers.Dense(2 * d_hidden)(h))
        h = layers.Dense(d)(layers.Dropout(hp["ffn_dropout"])(h))
        x = x + layers.Dropout(hp["residual_dropout"])(h)
    x = layers.Lambda(lambda t: t[:, -1])(x)                       # [CLS]
    return layers.Activation("relu")(layers.LayerNormalization()(x))


def _recurrent(cell, hp):
    """三層循環層 (前兩層 return_sequences) + 兩個 dropout + Flatten + Dense。"""
    return [
        cell(hp["units_1"], return_sequences=True),
        layers.Dropout(hp["dropout_1"]),
        cell(hp["units_2"], return_sequences=True),
        layers.Dropout(hp["dropout_2"]),
        cell(hp["units_3"]),
        layers.Flatten(),
        layers.Dense(hp["dense_units"], activation="relu"),
    ]


def build_model(name, task, n_features=122, accumulation_steps=None, n_classes=None, hp=None):
    """hp 省略時使用論文的最佳超參數；hp 可另含 lr 與 clipnorm。"""
    hp = hp or HP[task][name]
    if name == "FTT":
        if task == "binary":
            head, loss = layers.Dense(1, activation="sigmoid"), "binary_crossentropy"
        else:
            head = layers.Dense(n_classes or len(MULTI_CLASSES), activation="softmax")
            loss = "sparse_categorical_crossentropy"
        inp = keras.Input((n_features,))
        model = keras.Model(inp, head(_ft_transformer(inp, hp)), name=f"FTT_{task}")
        optimizer = keras.optimizers.AdamW(learning_rate=hp["lr"], weight_decay=hp["weight_decay"],
                                           clipnorm=hp.get("clipnorm"),
                                           gradient_accumulation_steps=accumulation_steps)
        # 與原論文相同: tokenizer、LayerNorm 與偏差項不做 weight decay
        optimizer.exclude_from_weight_decay(var_names=["bias", "beta", "gamma", "cls", "weight"])
        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        return model
    if name == "DNN":
        body = [
            keras.Input((n_features,)),
            layers.Dense(hp["units_1"], activation="relu"),
            layers.Dense(hp["units_2"], activation="relu"),
        ]
    elif name == "RNN":
        body = [keras.Input((n_features, 1))] + _recurrent(layers.SimpleRNN, hp)
    elif name == "LSTM":
        body = [keras.Input((n_features, 1))] + _recurrent(layers.LSTM, hp)
    elif name == "GRU":
        body = [keras.Input((n_features, 1))] + _recurrent(layers.GRU, hp)
    elif name == "CNN":
        body = [
            keras.Input((n_features, 1)),
            layers.Conv1D(hp["filters_1"], hp["kernel_size_1"], activation="relu"),
            layers.MaxPooling1D(4),
            layers.Dropout(hp["dropout_1"]),
            layers.Conv1D(hp["filters_2"], hp["kernel_size_2"], activation="relu"),
            layers.MaxPooling1D(2),
            layers.Dropout(hp["dropout_2"]),
            layers.Flatten(),
            layers.Dense(hp["dense_units"], activation="relu"),
        ]
    elif name == "CNN-LSTM":
        body = [
            keras.Input((n_features, 1)),
            layers.Conv1D(hp["filters"], hp["kernel_size"], activation="relu"),
            layers.MaxPooling1D(2),
            layers.LSTM(hp["units"], return_sequences=False),
            layers.Dropout(hp["dropout"]),
        ]
    else:
        raise ValueError(name)

    if task == "binary":
        head, loss = layers.Dense(1, activation="sigmoid"), "binary_crossentropy"
    else:
        head = layers.Dense(n_classes or len(MULTI_CLASSES), activation="softmax")
        loss = "sparse_categorical_crossentropy"
    model = keras.Sequential(body + [head], name=f"{name}_{task}".replace("-", "_"))
    optimizer = keras.optimizers.Adam(learning_rate=hp.get("lr", 1e-3), clipnorm=hp.get("clipnorm"),
                                      gradient_accumulation_steps=accumulation_steps)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------
# 訓練與評估
# --------------------------------------------------------------------------
def evaluate(task, y_true, prob):
    if task == "binary":
        pred = (prob.ravel() > 0.5).astype(int)
        metrics = dict(
            accuracy=accuracy_score(y_true, pred),
            precision=precision_score(y_true, pred),
            recall=recall_score(y_true, pred),
            f1=f1_score(y_true, pred),
        )
        names = BINARY_CLASSES
    else:
        pred = prob.argmax(axis=1)
        # 論文的 recall < accuracy，代表它用的是 Keras 的 Precision/Recall
        # (one-hot、門檻 0.5、micro 平均)，這裡用同樣算法以便對照。
        onehot = np.eye(prob.shape[1], dtype=bool)[y_true]
        hit = prob > 0.5
        tp = (hit & onehot).sum()
        precision, recall = tp / max(hit.sum(), 1), tp / onehot.sum()
        metrics = dict(
            accuracy=accuracy_score(y_true, pred),
            precision=precision,
            recall=recall,
            f1=2 * precision * recall / (precision + recall),
        )
        names = MULTI_CLASSES
    labels = list(range(len(names)))
    cm = confusion_matrix(y_true, pred, labels=labels)
    report = classification_report(y_true, pred, labels=labels, target_names=names,
                                   digits=4, zero_division=0, output_dict=True)
    return {k: float(v) for k, v in metrics.items()}, cm, report


def run(name, task, splits, epochs):
    X_train, X_test, y_train, y_test = splits[task]
    keras.utils.set_random_seed(SEED)
    if name in RECURRENT_MODELS:
        batch, steps = MICRO_BATCH, BATCH_SIZE // MICRO_BATCH
    else:
        batch, steps = BATCH_SIZE, None
    print(f"[{task:6s}] {name:8s} 開始訓練，epochs={epochs}，batch={batch}", flush=True)
    model = build_model(name, task, accumulation_steps=steps)
    if name in SEQUENCE_MODELS:
        X_train, X_test = X_train[..., None], X_test[..., None]

    start = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                        validation_split=0.2, verbose=1)
    seconds = time.time() - start
    print(f"[{task:6s}] {name:8s} 開始預測測試集", flush=True)
    prob = model.predict(X_test, batch_size=batch, verbose=0)
    metrics, cm, report = evaluate(task, y_test, prob)

    result = dict(model=name, task=task, epochs=epochs, params=model.count_params(),
                  train_seconds=round(seconds, 1), **metrics,
                  confusion_matrix=cm.tolist(), report=report,
                  history={k: [float(v) for v in vs]
                           for k, vs in history.history.items()})
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    (OUT / "runs" / f"{task}_{name}.json").write_text(json.dumps(result, indent=1))
    print(f"[{task:6s}] {name:8s} acc={metrics['accuracy']*100:.3f}  "
          f"prec={metrics['precision']*100:.3f}  rec={metrics['recall']*100:.3f}  "
          f"f1={metrics['f1']*100:.3f}  ({seconds:.0f}s)", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["binary", "multi", "both"], default="both")
    parser.add_argument("--models", nargs="+", default=list(HP["binary"]),
                        choices=list(HP["binary"]))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()

    import torch
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Keras {keras.__version__} / backend={keras.backend.backend()} / {device}")

    X, y_bin, y_multi = load_data()
    print(f"資料: {X.shape[0]} 筆, {X.shape[1]} 維特徵")
    # 3) 80/20 隨機切分；訓練時再用 validation_split=0.2 取驗證集
    splits = {
        "binary": train_test_split(X, y_bin, test_size=0.2, random_state=SEED),
        "multi": train_test_split(X, y_multi, test_size=0.2, random_state=SEED),
    }
    print(f"train={len(splits['binary'][0])}  test={len(splits['binary'][1])}")

    tasks = ["binary", "multi"] if args.task == "both" else [args.task]
    for task in tasks:
        for name in args.models:
            run(name, task, splits, args.epochs)


if __name__ == "__main__":
    main()
