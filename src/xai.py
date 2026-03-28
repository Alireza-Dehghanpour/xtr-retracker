import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

from .config import SAMPLES_PER_WF


def extract_surface_query_attention(model_base, x1, block_idx, window_bins=2):
    pred = float(model_base.predict(x1, verbose=0).reshape(-1)[0])
    q0 = int(np.clip(round(pred), 0, 127))
    q_idx = np.arange(max(0, q0 - window_bins), min(128, q0 + window_bins + 1))

    ln1 = model_base.get_layer(f"blk{block_idx}_ln1")
    inter = keras.Model(model_base.input, ln1.output)
    attn_in = inter.predict(x1, verbose=0)

    mha = model_base.get_layer(f"blk{block_idx}_mha")
    _, scores = mha(attn_in, attn_in, return_attention_scores=True, training=False)
    scores = scores.numpy()[0]  # (H,T,T)

    A_mean = scores.mean(axis=0)  # (T,T)
    imp = A_mean[q_idx, :].mean(axis=0)
    imp = imp / (imp.max() + 1e-8)

    return imp.astype(np.float32), A_mean.astype(np.float32), pred


def compute_saliency_map(model_base, input_waveform):
    x = tf.convert_to_tensor(input_waveform, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y_pred = model_base(x, training=False)
    grads = tape.gradient(y_pred, x)
    sal = tf.abs(grads).numpy().reshape(-1)
    return (sal / (sal.max() + 1e-9)).astype(np.float32)


def compute_smoothgrad_saliency(model_base, input_waveform, n_samples=16, noise_std=0.01):
    acc = np.zeros((SAMPLES_PER_WF,), dtype=np.float32)
    for _ in range(n_samples):
        noise = np.random.normal(0.0, noise_std, size=input_waveform.shape).astype(np.float32)
        acc += compute_saliency_map(model_base, input_waveform + noise)
    acc /= n_samples
    return (acc / (acc.max() + 1e-9)).astype(np.float32)


def export_xai_examples(model, X_te, y_te, save_path, num_samples=5, last_block=6):
    print("\nXAI export.............")
    inds = np.random.choice(len(X_te), num_samples, replace=False)

    for idx in inds:
        x1 = X_te[idx:idx + 1]
        true_val = y_te[idx]
        pred_val = float(model.predict(x1, verbose=0)[0, 0])

        imp, Amean, _ = extract_surface_query_attention(
            model, x1, block_idx=last_block, window_bins=2
        )
        sal = compute_smoothgrad_saliency(model, x1, n_samples=16, noise_std=0.01)

        wf = x1[0, :, 0]

        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axs[0].plot(wf)
        axs[0].set_title(f"idx={idx} | True={true_val:.3f} Pred={pred_val:.3f}")
        axs[0].set_ylabel("Amp")

        axs[1].plot(sal)
        axs[1].fill_between(np.arange(SAMPLES_PER_WF), 0, sal, alpha=0.2)
        axs[1].set_ylabel("SmoothGrad")

        axs[2].plot(imp)
        axs[2].fill_between(np.arange(SAMPLES_PER_WF), 0, imp, alpha=0.2)
        axs[2].set_ylabel("Surface-query attn")
        axs[2].set_xlabel("Bin")

        fig.tight_layout()
        plt.savefig(os.path.join(save_path, f"xai_idx{idx}.png"), dpi=150)
        plt.close()

        plt.figure(figsize=(6, 5))
        plt.imshow(Amean, aspect="auto", origin="lower")
        plt.colorbar(label="Attention")
        plt.title(f"Attention mean (blk {last_block}) idx={idx}")
        plt.xlabel("Key bin")
        plt.ylabel("Query bin")
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_path, f"xai_heatmap_blk{last_block}_idx{idx}.png"),
            dpi=150,
        )
        plt.close()

    print("Done.")