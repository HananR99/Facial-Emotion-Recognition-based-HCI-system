from pathlib import Path
import cv2
import os

# Define input and output directories
data_dir = Path('data/raw/finegrained')
output_dir = Path('data/cropped_faces')
output_dir.mkdir(parents=True, exist_ok=True)

# Load OpenCV's pre-trained Haar cascade for frontal face detection
if hasattr(cv2, 'data'):
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
else:
    # Fallback: try to find the file in common locations or raise an error
    cascade_path = os.path.join(
        os.path.dirname(cv2.__file__),
        'data', 'haarcascade_frontalface_default.xml'
    )
    if not os.path.exists(cascade_path):
        raise FileNotFoundError("Could not find haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(cascade_path)

# Iterate through all image files in subdirectories
for img_path in data_dir.rglob('*.*'):
    if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
        continue

    # Read the image
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    # Convert to grayscale for detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Save each detected face
    for i, (x, y, w, h) in enumerate(faces):
        face_img = img[y:y+h, x:x+w]
        # Preserve subfolder structure in output
        rel_subpath = img_path.relative_to(data_dir).parent
        filename = f"{img_path.stem}_face{i}{img_path.suffix}"
        save_path = output_dir / rel_subpath / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_path), face_img)

print("Face cropping complete. Cropped images saved to:", output_dir)
