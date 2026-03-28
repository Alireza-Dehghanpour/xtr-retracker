import os
import numpy as np
import tensorflow as tf

# ============================================================
DATA_ROOT = "/home/XXXXX"   #
SAVE_PATH = "/home/XXXXXXXXXx"

os.makedirs(SAVE_PATH, exist_ok=True)

SAMPLES_PER_WF = 128
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-4
CLIPNORM = 1.0
ALGO_NAME = "XTR_XXX"

AUG_PROB = 0.95
AUG_NOISE_STD = 0.015
AUG_SCALE_JITTER = 0.08
AUG_SHIFT_MAX = 2
AUG_SMOOTH_PROB = 0.35
AUG_SMOOTH_KERNEL = 3

tf.random.set_seed(42)
np.random.seed(42)