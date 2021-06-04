import cv2 as cv
import warnings
import joblib
import time
import numpy as np
import argparse
import sys
warnings.simplefilter('ignore')


parser = argparse.ArgumentParser(description='Run right sitting check')
parser.add_argument("--device", default="cpu", help="Device to inference on")
parser.add_argument("--input_source", default="test/video/IMG_9655.MOV", help="Input Video")
parser.add_argument("--output_file", default="test/video/output_video/output.avi",
                    help="Output video path")
parser.add_argument("--model", default="SVM", help="Used classification model")

args = parser.parse_args()


# Function that get points from model's output
def points_find(out_put, npoints, h, w, frame_width, frame_height, th):
    points = []

    for i in range(npoints):
        # Confidence map of corresponding body's part.
        prob_map = out_put[0, i, :, :]

        # Find global maxima of the probMap.
        min_val, prob, min_loc, point = cv.minMaxLoc(prob_map)

        # Scale the point to fit on the original image
        x = (frame_width * point[0]) / w
        y = (frame_height * point[1]) / h

        if prob > th:
            points.append(int(x))
            points.append(int(y))
        else:
            points.append(None)
            points.append(None)

    points = np.array(points)
    points = np.delete(points, [20, 21, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35])

    return points


# Load pre-trained models from sklearn
if args.model == "SVM":
    # Support vector machine classifier
    class_model = joblib.load('models/classification/svm_model.pkl')
elif args.model == "CatBoost":
    # CatBoost classifier
    class_model = joblib.load('models/classification/cat_boost_model.pkl')
else:
    print("Wrong model!")
    sys.exit()

# Load Standard Scaler
scaler_model = joblib.load('models/classification/scaler.pkl')
# Load KNN Impute
knn_impute = joblib.load('models/classification/knn_missing.pkl')


# Load OpenPose model pre-trained on COCO dataset
protoFile = "models/OpenPose/coco/pose_deploy_linevec.prototxt"
weightsFile = "models/OpenPose/coco/pose_iter_440000.caffemodel"
# Using number of points in model
nPoints = 18
POSE_PAIRS = [[1, 0], [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
              [11, 12], [12, 13], [0, 14], [0, 15], [14, 16], [15, 17]]
# Load neural network weights
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)
# Check what device to use for calculation
device = args.device
if device == 'cpu':
    net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif device == 'gpu':
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")
else:
    print("Wrong device")
    sys.exit()

# Load video from project source
input_s = args.input_source

if input_s == '0' or input_s == '1':
    input_s = int(input_s)

cap = cv.VideoCapture(input_s)
hasFrame, frame = cap.read()
fps = cap.get(cv.CAP_PROP_FPS)
print(f"FPS = {int(fps)}")
# Save processed video
output_source = args.output_file
vid_writer = cv.VideoWriter(output_source, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                            fps,
                            (frame.shape[1], frame.shape[0]))

# Set input parameters for image
inWidth = 225
inHeight = 225
threshold = 0.1

# Frames counter
# This frame counter will be used for take frame for processed with classification model every 1 second
count = 0
# Variable for siting conditions
how = ''

# Loop for video processed, this loop takes every frame from video
while cv.waitKey(1) < 0:
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    if count % int(fps) == 0:

        inpBlob = cv.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                       (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
        output = net.forward()

        H = output.shape[2]
        W = output.shape[3]

        key_points = points_find(output, nPoints, H, W, frameWidth, frameHeight, threshold)
        key_points = np.reshape(key_points, (1, 24))
        key_points = knn_impute.transform(key_points)
        key_points = scaler_model.transform(key_points)
        label = class_model.predict(key_points)[0]

        if label == 0:
            how = 'Good'

        elif label == 1:
            how = 'Bad'

    cv.putText(frame, how, (50, 150), cv.FONT_HERSHEY_COMPLEX, 1.5, (255, 50, 0), 3, lineType=cv.LINE_AA)
    count += 1

    cv.putText(frame, "time taken = {:.2f} sec".format(time.time() - t), (50, 50), cv.FONT_HERSHEY_COMPLEX,
               .8, (255, 50, 0), 2, lineType=cv.LINE_AA)

    # Output frame number and time that spend on processed this frame
    if input_s == 0 or input_s == 1:
        cv.imshow("Output", frame)
        if cv.waitKey(10) == 27:  # Клавиша Esc
            break
    else:
        print(count, time.time() - t)
        vid_writer.write(frame)


if input_s == 0 or input_s == 1:
    cap.release()
    cv.destroyAllWindows()
else:
    vid_writer.release()
