import os
import numpy as np
import tensorflow as tf

# ============================================================
DATA_ROOT = "/home/XXXXX"   #
SAVE_PATH = "/home/XXXXXXXXXx"

os.makedirs(SAVE_PATH, exist_ok=True)

SAMPLES_PER_WF = XX
BATCH_SIZE = XX
EPOCHS = XX
LR = XX
CLIPNORM =XX
ALGO_NAME = "XTR_XXX"

AUG_PROB = XX
AUG_NOISE_STD = XX
AUG_SCALE_JITTER = XX
AUG_SHIFT_MAX = XX
AUG_SMOOTH_PROB = XX
AUG_SMOOTH_KERNEL = X

tf.random.set_seed(42)
np.random.seed(42)
