import cv2

def detect_eyes(img_array, min_neighbors=10):
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=min_neighbors
    )

    for (x, y, w, h) in eyes:
        cv2.rectangle(img_array, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return img_array, len(eyes)