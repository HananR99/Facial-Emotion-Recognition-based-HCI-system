import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from dataset import load_data


def parse_args():
    default_model = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../models/final_model.h5')
    )
    parser = argparse.ArgumentParser(
        description="Evaluate a trained CNN on the test dataset."
    )
    parser.add_argument(
        "--model_path", type=str, default=default_model,
        help="Path to the .h5 model file to evaluate."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Check model file exists
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")

    # Load test dataset (assumes load_data returns (train_ds, val_ds, test_ds))
    _, _, test_ds = load_data()

    # Load the trained model
    model = tf.keras.models.load_model(args.model_path)
    print(f"Loaded model from {args.model_path}\n")

    # Predict and collect labels
    y_true, y_pred = [], []
    for X_batch, Y_batch in test_ds:
        preds = model.predict(X_batch)
        y_true.extend(np.argmax(Y_batch, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    # Define class names in correct order
    labels = [
        'Anger', 'Contempt', 'Disgust', 'Fear',
        'Happiness', 'Neutrality', 'Sadness', 'Surprise'
    ]

    # Print detailed classification report
    report = classification_report(
        y_true, y_pred,
        target_names=labels,
        digits=4
    )
    print("Classification Report on Test Set:\n")
    print(report)


if __name__ == "__main__":
    main()
