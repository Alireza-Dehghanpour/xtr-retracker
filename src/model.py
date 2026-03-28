import tensorflow as tf
from tensorflow import keras

from .config import LR, CLIPNORM


def create_hybrid_xtr(
    input_size=(128, 1),
    cnn_filters=(64, 64, 96, 96, 128, 128),
    cnn_kernel=3,
    cnn_dropout=0.05,
    d_model=128,
    num_layers=6,
    num_heads=8,
    ff_mult=4,
    dropout_rate=0.10,
    head_units=(256, 128),
):
    L = keras.layers

    In = L.Input(shape=input_size, name="input_layer")

    # --stem
    x = In
    for i, f in enumerate(cnn_filters):
        x = L.Conv1D(f, cnn_kernel, strides=1, padding="same", name=f"stem_conv{i+1}")(x)
        x = L.BatchNormalization(name=f"stem_bn{i+1}")(x)
        x = L.Activation("relu", name=f"stem_relu{i+1}")(x)
        x = L.Dropout(cnn_dropout, name=f"stem_drop{i+1}")(x)

    x = L.Dense(d_model, name="proj_dense")(x)

    T_len = input_size[0]
    pos = tf.range(0, T_len)
    pos_emb = L.Embedding(T_len, d_model, name="pos_emb")(pos)
    pos_emb = tf.expand_dims(pos_emb, axis=0)
    x = L.Add(name="add_pos")([x, pos_emb])

    # ---------------------------------transformer
    for i in range(num_layers):
        attn_in = L.LayerNormalization(epsilon=1e-6, name=f"blk{i+1}_ln1")(x)
        attn_out = L.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=max(8, d_model // num_heads),
            dropout=dropout_rate,
            name=f"blk{i+1}_mha",
        )(attn_in, attn_in)
        attn_out = L.Dropout(dropout_rate, name=f"blk{i+1}_attn_drop")(attn_out)
        x = L.Add(name=f"blk{i+1}_attn_add")([x, attn_out])

        ffn_in = L.LayerNormalization(epsilon=1e-6, name=f"blk{i+1}_ln2")(x)
        ffn = L.Dense(d_model * ff_mult, activation="swish", name=f"blk{i+1}_ffn1")(ffn_in)
        ffn = L.Dropout(dropout_rate, name=f"blk{i+1}_ffn_drop1")(ffn)
        ffn = L.Dense(d_model, name=f"blk{i+1}_ffn2")(ffn)
        ffn = L.Dropout(dropout_rate, name=f"blk{i+1}_ffn_drop2")(ffn)
        x = L.Add(name=f"blk{i+1}_ffn_add")([x, ffn])

    
    x = L.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
    x = L.GlobalAveragePooling1D(name="gap")(x)
    x = L.Dense(head_units[0], activation="relu", name="head_dense1")(x)
    x = L.Dropout(0.10, name="head_drop1")(x)
    x = L.Dense(head_units[1], activation="relu", name="head_dense2")(x)
    x = L.Dropout(0.05, name="head_drop2")(x)
    out = L.Dense(1, activation="linear", name="output")(x)

    model = keras.Model(In, out, name="HybridXTR_AUG")

    opt = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=CLIPNORM)
    model.compile(
        optimizer=opt,
        loss="mse",
        metrics=[tf.keras.metrics.MeanSquaredError(name="mse")],
    )
    return model