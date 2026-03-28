import os
import gc
import numpy as np
from sklearn.model_selection import train_test_split

from .config import DATA_ROOT, SAMPLES_PER_WF


def load_file_float64(path):
    with open(path, "rb") as f:
        return np.fromfile(f, np.float64)


def load_file_float32(path):
    with open(path, "rb") as f:
        return np.fromfile(f, np.float32)


def resolve_data_dirs():
    att_dir = os.path.join(DATA_ROOT, "ATT")
    ref_dir = os.path.join(DATA_ROOT, "REF")
    wf_dir = os.path.join(DATA_ROOT, "SIM_WF")

    if not (os.path.isdir(att_dir) and os.path.isdir(ref_dir) and os.path.isdir(wf_dir)):
        att_dir = ref_dir = wf_dir = DATA_ROOT

    print("_" * 100)
    print("Using folders:")
    print(" ATT dir :", att_dir)
    print(" REF dir :", ref_dir)
    print(" WF dir :", wf_dir)
    print("_" * 100)

    return att_dir, ref_dir, wf_dir


def load_dataset():
    att_dir, ref_dir, wf_dir = resolve_data_dirs()

    files = [f for f in os.listdir(att_dir) if f.startswith("ATT_ARR_") and f.endswith(".bin")]
    if len(files) == 0:
        raise RuntimeError(f"No ATT_ARR_*.bin files found in {att_dir}. Check DATA_ROOT/folders.")

    data_wave_chunks = []
    lbl_REF_chunks = []

    print("Loading data ...")
    for fn in files:
        idx = fn.split("_")[-1].split(".")[0]

        att = load_file_float64(os.path.join(att_dir, f"ATT_ARR_{idx}.bin"))
        ref = load_file_float64(os.path.join(ref_dir, f"REF_ARR_{idx}.bin"))
        wf_raw = load_file_float32(os.path.join(wf_dir, f"SIM_WF_NOISE_{idx}.bin"))

        n_att = att.shape[0]

        if ref.shape[0] != n_att:
            raise ValueError(f"[{idx}] REF size ({ref.shape[0]}) != ATT size ({n_att})")

        total = wf_raw.size
        if total % (n_att * SAMPLES_PER_WF) != 0:
            raise ValueError(
                f"[{idx}] WF size mismatch: size={total}, n_att={n_att}, samples={SAMPLES_PER_WF}"
            )

        n_noise = total // (n_att * SAMPLES_PER_WF)
        wf = wf_raw.reshape(n_att, n_noise, SAMPLES_PER_WF)
        mx = np.max(wf, axis=-1, keepdims=True)
        mx[mx == 0.0] = 1.0
        wf = wf / mx

        data_wave_chunks.append(wf.reshape(-1, SAMPLES_PER_WF))
        lbl_REF_chunks.append(np.repeat(ref, n_noise))

    data_wave = np.concatenate(data_wave_chunks, axis=0).astype(np.float32)
    lbl_REF = np.concatenate(lbl_REF_chunks, axis=0).astype(np.float32)

    del data_wave_chunks, lbl_REF_chunks
    gc.collect()

    print("_" * 100)
    print("Final shapes:")
    print(" data_wave:", data_wave.shape)
    print(" lbl_REF :", lbl_REF.shape)
    print("_" * 100)

    return data_wave, lbl_REF


def split_dataset(data_wave, lbl_REF):
    x_train, x_test, y_train, y_test = train_test_split(
        data_wave, lbl_REF, test_size=0.20, random_state=0
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.15, random_state=0
    )

    del data_wave, lbl_REF
    gc.collect()

    X_tr = x_train[..., np.newaxis]
    X_va = x_val[..., np.newaxis]
    X_te = x_test[..., np.newaxis]

    y_tr = y_train
    y_va = y_val
    y_te = y_test

    return X_tr, X_va, X_te, y_tr, y_va, y_te