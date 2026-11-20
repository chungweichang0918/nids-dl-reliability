"""用論文的框架 (TensorFlow / Keras 2.14) 訓練同一個模型，檢驗框架差異假設。

在 TensorFlow 環境 (.venv-tf, CPU) 執行，batch size 5000 直接訓練、不做梯度累積:
    .venv-tf/Scripts/python.exe tf_check.py --task binary --model LSTM
"""
import argparse
import json
import time

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

import nids_repro as nr

# Keras 2 沒有 gradient_accumulation_steps 參數；論文設定本來就不需要，忽略即可。
_Adam = keras.optimizers.Adam
nr.keras.optimizers.Adam = lambda gradient_accumulation_steps=None, **kw: _Adam(**kw)


class KeepBest(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best, self.epoch, self.weights = -1.0, 0, None

    def on_epoch_end(self, epoch, logs=None):
        if logs["val_accuracy"] > self.best:
            self.best, self.epoch = logs["val_accuracy"], epoch + 1
            self.weights = self.model.get_weights()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="binary")
    p.add_argument("--model", default="LSTM")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    X, y_bin, y_multi = nr.load_data()
    y = y_bin if a.task == "binary" else y_multi
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=nr.SEED)
    keras.utils.set_random_seed(a.seed)
    model = nr.build_model(a.model, a.task)
    if a.model in nr.SEQUENCE_MODELS:
        X_train, X_test = X_train[..., None], X_test[..., None]

    keep = KeepBest()
    t0 = time.time()
    hist = model.fit(X_train, y_train, epochs=a.epochs, batch_size=nr.BATCH_SIZE,
                     validation_split=0.2, verbose=2, callbacks=[keep])
    seconds = time.time() - t0
    last, _, _ = nr.evaluate(a.task, y_test, model.predict(X_test, batch_size=5000, verbose=0))
    model.set_weights(keep.weights)
    best, _, _ = nr.evaluate(a.task, y_test, model.predict(X_test, batch_size=5000, verbose=0))
    val = hist.history["val_accuracy"]
    out = dict(task=a.task, model=a.model, variant=f"tf{tf.__version__}", seed=a.seed,
               seconds=round(seconds), last=last, best=best, best_epoch=keep.epoch,
               val_accuracy=[float(v) for v in val],
               val_loss=[float(v) for v in hist.history["val_loss"]],
               collapses=int(sum(1 for i in range(1, len(val))
                                 if val[i] < max(val[:i]) - 0.05)))
    outdir = nr.OUT / "ablation"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{a.task}_{a.model}_tf.json").write_text(json.dumps(out, indent=1))
    print(f"[TF {tf.__version__}] {a.task} {a.model} last={last['accuracy']*100:.2f} "
          f"best={best['accuracy']*100:.2f}@{keep.epoch} collapses={out['collapses']} "
          f"({seconds:.0f}s)")


if __name__ == "__main__":
    main()
