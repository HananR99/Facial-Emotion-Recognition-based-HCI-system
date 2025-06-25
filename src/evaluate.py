import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from dataset_loader import get_datasets, EMOTION_LABELS as EMOTIONS

def main():
    _, _, test_ds = get_datasets()
    model = tf.keras.models.load_model('models/final_model.h5')

    loss, acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {acc:.4f}\n")

    # reload test for reporting
    _, _, test_ds = get_datasets()
    y_true, y_pred = [], []
    for X, y in test_ds:
        preds = model.predict(X)
        y_true.extend(y.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    labels = list(range(len(EMOTIONS)))
    print("Classification Report:")
    print(classification_report(
        y_true, y_pred,
        labels=labels,
        target_names=EMOTIONS,
        digits=4,
        zero_division=0
    ))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8,8))
    disp = ConfusionMatrixDisplay(cm, display_labels=EMOTIONS)
    disp.plot(ax=ax, cmap='Blues', xticks_rotation='vertical', values_format='d')
    ax.set_xticks(np.arange(len(EMOTIONS)))
    ax.set_yticks(np.arange(len(EMOTIONS)))
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
