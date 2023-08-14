#https://hackaday.io/project/27552-blinktotext/log/68360-eye-blink-detection-algorithms
#https://thesai.org/Downloads/Volume13No6/Paper_93-Deep_Learning_Approach_for_Efficient_Eye_blink_Detection.pdf

import argparse
import cv2
import dlib
import numpy as np
from imutils import face_utils
from scipy.spatial import distance as dist

#passing command-line arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor", required=True, help="path to financial landmark  predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

def calculate_eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A+B)/(2*C)
    return ear

# reads in the video
print("[INFO] Loading Video")
cap = cv2.VideoCapture(args["video"])

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (
    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
)
# constant variables
EYE_BLINK_THRESH = 0.25
SUCC_FRAME = fps/10 #(3)
COUNTER = 0
TOTAL = 0

# Load the pre-trained facial landmark detector
print("[INFO] Loading facial landmark predictor..")
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor(args["shape_predictor"])

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


while True:

    success, frame = cap.read()

    if not success: #accomodates when the last frame is reached
        break

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(frame_gray)
    
    if faces:
        face = faces[0]
       
        shape = landmark_predict(frame_gray, face)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[lStart:lEnd]
        right_eye = shape[rStart:rEnd]

        left_EyeHull = cv2.convexHull(left_eye)
        right_EyeHull = cv2.convexHull(right_eye)

        cv2.drawContours(frame, [left_EyeHull], -1, (0,255,0), 1)
        cv2.drawContours(frame, [right_EyeHull], -1, (0,255,0), 1)

        left_EAR = calculate_eye_aspect_ratio(left_eye)
        right_EAR = calculate_eye_aspect_ratio(right_eye)

        avg_EAR = (left_EAR+right_EAR)/2
        #print(avg_EAR) 

        if avg_EAR < EYE_BLINK_THRESH:
            COUNTER += 1
        else:
            # if COUNTER >= SUCC_FRAME and COUNTER > CLOSE_THRESH:
            #     #print("Eyes Closed.")
            #     pass
            if COUNTER >= SUCC_FRAME: #eye blink
                TOTAL += 1

            COUNTER = 0

        cv2.putText(frame, "Blink Counter: {}".format(TOTAL), (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)
        cv2.putText(frame, "EAR: {:.2f}".format(avg_EAR), (300,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255),2)

    cv2.imshow("Image", frame)
    
    key = cv2.waitKey(1)
    # Break the loop if q is pressed (optional)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
