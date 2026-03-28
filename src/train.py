import os
import csv
import gc
import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.config import (
    SAVE_PATH,
    SAMPLES_PER_WF,
    BATCH_SIZE,
    EPOCHS,
    ALGO_NAME,
)
from src.data_loader import load_dataset, split_dataset
from src.augmentations import augment_waveform
from src.model import create_hybrid_xtr
from src.metrics import mse_np, rmse_np, mae_np, r2_np
from src.xai import export_xai_examples


def main():

    data_wave, lbl_REF = load_dataset()
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_dataset(data_wave, lbl_REF)

    gc.collect()

    # Model
  
    model = create_hybrid_xtr(
        input_size=(SAMPLES_PER_WF, 1),
        d_model=128,
        num_layers=6,
        num_heads=8,
        ff_mult=4,
        dropout_rate=0.10,
    )
    model.summary()
    1/0   #
    # ============================================================
 
    AUTO = tf.data.AUTOTUNE

    ds_tr = (
        tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
        .shuffle(50_000)
        .map(augment_waveform, num_parallel_calls=AUTO)
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    ds_va = (
        tf.data.Dataset.from_tensor_slices((X_va, y_va))
        .batch(BATCH_SIZE)
        .prefetch(AUTO)
    )

    ckpt_path = os.path.join(SAVE_PATH, f"{ALGO_NAME}_best.keras")

    cbs = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_mse",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mse",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_mse",
            patience=20,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    print(f"Starting training: {ALGO_NAME}")
    hist = model.fit(
        ds_tr,
        validation_data=ds_va,
        epochs=EPOCHS,
        callbacks=cbs,
        verbose=1,
    )

    # ============================================================

    np.save(os.path.join(SAVE_PATH, f"history_{ALGO_NAME}.npy"), hist.history)
    model.save(
        os.path.join(SAVE_PATH, f"{ALGO_NAME}_final.keras"),
        include_optimizer=False,
        overwrite=True,
    )
    model.save(
        os.path.join(SAVE_PATH, f"{ALGO_NAME}_final.h5"),
        include_optimizer=False,
        overwrite=True,
    )

    # ============================================================

    yhat_tr = model.predict(X_tr, verbose=0).reshape(-1)
    yhat_va = model.predict(X_va, verbose=0).reshape(-1)
    yhat_te = model.predict(X_te, verbose=0).reshape(-1)

    mtr, rtr, atr, r2tr = mse_np(y_tr, yhat_tr), rmse_np(y_tr, yhat_tr), mae_np(y_tr, yhat_tr), r2_np(y_tr, yhat_tr)
    mva, rva, ava, r2va = mse_np(y_va, yhat_va), rmse_np(y_va, yhat_va), mae_np(y_va, yhat_va), r2_np(y_va, yhat_va)
    mte, rte, ate, r2te = mse_np(y_te, yhat_te), rmse_np(y_te, yhat_te), mae_np(y_te, yhat_te), r2_np(y_te, yhat_te)

    print("_" * 100)
    print(f"{ALGO_NAME} | TRAIN -> MSE:{mtr:.6f} RMSE:{rtr:.6f} MAE:{atr:.6f} R2:{r2tr:.6f}")
    print(f"{ALGO_NAME} | VAL   -> MSE:{mva:.6f} RMSE:{rva:.6f} MAE:{ava:.6f} R2:{r2va:.6f}")
    print(f"{ALGO_NAME} | TEST  -> MSE:{mte:.6f} RMSE:{rte:.6f} MAE:{ate:.6f} R2:{r2te:.6f}")
    print("_" * 100)

    with open(os.path.join(SAVE_PATH, f"{ALGO_NAME}_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "MSE", "RMSE", "MAE", "R2"])
        w.writerow(["train", mtr, rtr, atr, r2tr])
        w.writerow(["val", mva, rva, ava, r2va])
        w.writerow(["test", mte, rte, ate, r2te])

    # ============================================================

    export_xai_examples(
        model=model,
        X_te=X_te,
        y_te=y_te,
        save_path=SAVE_PATH,
        num_samples=5,
        last_block=6,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(
            "\nExecution disabled: this repository accompanies a research under review.\n"
            "Full code will be released after publication.\n"
            "Contact: a.r.dehghanpour@gmail.com\n"
        )