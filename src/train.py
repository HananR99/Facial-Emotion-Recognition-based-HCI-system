import os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from dataset_loader import get_datasets, EMOTION_LABELS as EMOTIONS

# Ensure output dir
os.makedirs('models', exist_ok=True)

# 1. load data
train_ds, val_ds, test_ds = get_datasets()

# 2. build model
base = MobileNetV2(input_shape=(160,160,3), include_top=False, weights='imagenet')
base.trainable = False
model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.5),
    layers.Dense(len(EMOTIONS), activation='softmax')
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

# 3. callbacks
cbs = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3),
    ModelCheckpoint('models/best_model.h5', monitor='val_accuracy', save_best_only=True)
]

# 4. train head
history = model.fit(
    train_ds,
    epochs=32,
    validation_data=val_ds,
    callbacks=cbs
)

# 5. fine-tune last layers
base.trainable = True
for layer in base.layers[:-50]:
    layer.trainable = False
model.compile(
    optimizer=optimizers.Adam(1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
history2 = model.fit(
    train_ds,
    epochs=42,            # total = 32+10
    initial_epoch=32,
    validation_data=val_ds,
    callbacks=cbs
)

# 6. evaluate & save
print("\nTest set evaluation:")
model.evaluate(test_ds)
model.save('models/final_model.h5')
print("Models saved → models/best_model.h5 & models/final_model.h5")