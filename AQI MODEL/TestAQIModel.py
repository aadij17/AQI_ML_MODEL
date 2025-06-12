import cv2
import numpy as np
from keras.models import model_from_json


emotion_dict = {0: "POOR", 1: "VERY POOR", 2: "SEVERE"}

# load json and create model
json_file = open('D:\\Sayyam AQI Model\\air_quality_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
aqi_model = model_from_json(loaded_model_json)

# load weights into new modelpyt
aqi_model.load_weights("D:\\Sayyam AQI Model\\air_quality_model.h5")
print("Loaded model from disk")

# start the webcam feed
# cap = cv2.VideoCapture(0)

cap = cv2.imread("D:\\Sayyam AQI Model\\Dataset\\TEST\\POOR\\test_image.jpeg", 1)

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('Emotion_detection_with_CNN-main\\haarcascades\\haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = aqi_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
