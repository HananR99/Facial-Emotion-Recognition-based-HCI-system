import os
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split

# === CONFIG ===
IMG_SIZE    = 160
BATCH_SIZE  = 32
VAL_SPLIT   = 0.2
TEST_SPLIT  = 0.1
SHUFFLE_BUF = 2048

ROOT     = os.path.dirname(__file__)
PNG_ROOT = os.path.join(ROOT, '..', 'data', 'raw', 'finegrained')

EMOTIONS = [
    'anger','contempt','disgust','fear',
    'happiness','neutrality','sadness','surprise'
]

def get_datasets():
    # Collect image paths and labels
    paths, labels = [], []
    for idx, label in enumerate(EMOTIONS):
        folder = os.path.join(PNG_ROOT, label)
        for p in glob(os.path.join(folder, '*.png')):
            paths.append(p)
            labels.append(idx)
    paths = np.array(paths)
    labels = np.array(labels)

    # Stratified split: train / (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        paths, labels,
        test_size=VAL_SPLIT + TEST_SPLIT,
        stratify=labels,
        random_state=42
    )
    # split temp into val / test
    rel_test = TEST_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=rel_test,
        stratify=y_temp,
        random_state=42
    )

    def make_ds(paths_arr, labels_arr, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((paths_arr, labels_arr))
        if shuffle:
            ds = ds.shuffle(buffer_size=SHUFFLE_BUF, seed=42)
        def _load(path, label):
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE]) / 255.0
            return img, label
        ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return (
        make_ds(X_train, y_train, shuffle=True),
        make_ds(X_val,   y_val),
        make_ds(X_test,  y_test)
    )

# Expose for imports
get_datasets = get_datasets
EMOTION_LABELS = EMOTIONS