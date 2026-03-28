import tensorflow as tf

from .config import (
    AUG_PROB,
    AUG_NOISE_STD,
    AUG_SCALE_JITTER,
    AUG_SHIFT_MAX,
    AUG_SMOOTH_PROB,
    AUG_SMOOTH_KERNEL,
)


def smooth_1d(x, k=3):
    #
    kernel = tf.ones((k, 1, 1), dtype=x.dtype) / float(k)
    x2 = tf.expand_dims(x, axis=0)  # (1,T,1)
    x2 = tf.nn.conv1d(x2, filters=kernel, stride=1, padding="SAME")
    return tf.squeeze(x2, axis=0)


@tf.function
def augment_waveform(x, y):

    u = tf.random.uniform(())

    def do_aug():

        scale = 1.0 + tf.random.uniform((), -AUG_SCALE_JITTER, AUG_SCALE_JITTER)
        x1 = x * scale

        n = tf.random.normal(tf.shape(x1), stddev=AUG_NOISE_STD)
        x1 = x1 + n

        shift = tf.random.uniform((), -AUG_SHIFT_MAX, AUG_SHIFT_MAX + 1, dtype=tf.int32)
        x1 = tf.roll(x1, shift=shift, axis=0)

        u2 = tf.random.uniform(())
        x1 = tf.cond(
            u2 < AUG_SMOOTH_PROB,
            lambda: smooth_1d(x1, k=AUG_SMOOTH_KERNEL),
            lambda: x1,
        )

        x1 = tf.clip_by_value(x1, 0.0, 1.2)
        return x1, y

    return tf.cond(u < AUG_PROB, do_aug, lambda: (x, y))