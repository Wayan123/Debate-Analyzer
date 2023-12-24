import cv2
import numpy as np
from deepface import DeepFace
import imutils
import argparse
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", required=True, help="video path")
args = vars(ap.parse_args())


prototxt_path = 'deploy.prototxt.txt'
caffemodel_path = 'res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
window_name = "Debate Analyzer"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
output_file = "emotion_data.txt"
output_video_file = "output_video.mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter(output_video_file, fourcc, 30, (1000, 563))


video_path = args["video"]
vs = cv2.VideoCapture(video_path)


def detect_faces(frame, net):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    detected_faces = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            detected_faces.append((startX, startY, endX, endY))

    return detected_faces


def analyze_emotion(face_roi):
    result = DeepFace.analyze(
        face_roi, actions=['emotion'], enforce_detection=False)
    detected_emotions = result[0]['emotion']
    dominant_emotion = result[0]['dominant_emotion']
    return detected_emotions, dominant_emotion


detected_emotions = []

roi_width_percent = 45
roi_height_percent = 60

emotion_data = {
    'happy': 0,
    'sad': 0,
    'angry': 0,
    'neutral': 0,
    'surprise': 0,
    'fear': 0,
    'disgust': 0
}
frame_count = 0
start_time = time.time()
interval = 1


while True:
    ret, frame = vs.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    frame = imutils.resize(frame, width=1000)
    (frame_height, frame_width) = frame.shape[:2]

    roi_x = int(frame_width * (100 - roi_width_percent) // 200)
    roi_y = int(frame_height * (100 - roi_height_percent) // 200)
    roi_w = int(frame_width * roi_width_percent // 100)
    roi_h = int(frame_height * roi_height_percent // 100)

    # Draw the ROI box on the frame
    cv2.rectangle(frame, (roi_x, roi_y),
                  (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

    # Update the detect_faces function to consider only faces within the ROI
    detected_faces = detect_faces(
        frame[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w], net)

    for (startX, startY, endX, endY) in detected_faces:
        # Offset the detected face coordinates by the ROI position
        startX += roi_x
        startY += roi_y
        endX += roi_x
        endY += roi_y

        face_roi = frame[startY:endY, startX:endX]
        detected_emotions, dominant_emotion = analyze_emotion(face_roi)

        for emotion, value in detected_emotions.items():
            emotion_data[emotion] += value

        frame_count += 1

        # Log emotions every 'interval' seconds
        if time.time() - start_time >= interval:
            if dominant_emotion:
                with open(output_file, 'a') as f:
                    f.write(f"{dominant_emotion}\n")
            start_time = time.time()

        # Draw a black box for the dominant emotion in the top left corner of the window
        cv2.rectangle(frame, (0, 0), (480, 40), (0, 0, 0), -1)
        cv2.putText(frame, f'Dominant Facial Expression: {dominant_emotion}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)

        # Draw rectangles around the detected faces
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Display all detected emotions to the right of the bounding box
        bg_height = 22 * len(detected_emotions)  # Adjust the height as needed
        cv2.rectangle(frame, (endX + 10, startY), (endX + 200,
                      startY + bg_height), (0, 0, 0), -1, cv2.LINE_AA)
        y_text = startY + 20
        for emotion, value in detected_emotions.items():
            cv2.putText(frame, f'{emotion}: {round(value, 5)}', (endX + 10, y_text),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            y_text += 20

    output_video.write(frame)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cv2.destroyAllWindows()
vs.release()
