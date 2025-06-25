import cv2
import numpy as np
import pyautogui
import tensorflow as tf
from dataset_loader import EMOTION_LABELS as EMOTIONS

model = tf.keras.models.load_model('models/final_model.h5')

action_map = {
    'anger':     lambda: pyautogui.press('left'),
    'contempt':  lambda: pyautogui.press('right'),
    'disgust':   lambda: pyautogui.press('up'),
    'fear':      lambda: pyautogui.press('down'),
    'happiness': lambda: pyautogui.press('space'),
    'neutrality':lambda: None,
    'sadness':   lambda: pyautogui.press('volumeup'),
    'surprise':  lambda: pyautogui.press('volumedown')
}

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret: break
    img = cv2.resize(frame, (160,160)).astype('float32')/255.0
    preds = model.predict(np.expand_dims(img, 0))
    label = EMOTIONS[np.argmax(preds)]
    action_map[label]()
    cv2.putText(frame, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow('Emotion HCI', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
