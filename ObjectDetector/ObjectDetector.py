import joblib
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from FeatureExtraction.feature_extraction import extract_features
from RegionBasedDetection.selective_search import selective_search

# Function to calculate Intersection over Union (IoU)
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Function to detect objects in an image using the trained classifier and regressor
def detect_objects(image, classifier, regressor):
    _, regions = selective_search(image, min_size=1000)
    detected_boxes = []
    print("len(regions): ", len(regions))
    for region in regions:
        x, y, w, h = region
        proposal_img = image[y:y+h, x:x+w]
        proposal_img = cv2.resize(proposal_img, (224, 224))  # Resize to VGG16 input size
        feature_vector = extract_features(proposal_img, 'hog')
        prediction = classifier.predict([feature_vector])
        if prediction == 1:  # Assuming 1 is the positive class
            bbox_reg = regressor.predict([feature_vector])[0]
            dx, dy, dw, dh = bbox_reg
            x = int(x + dx * w)
            y = int(y + dy * h)
            w = int(w * np.exp(dw))
            h = int(h * np.exp(dh))
            detected_boxes.append((x, y, w, h))
    return detected_boxes

# Load the dataset information

data_info = pd.read_csv("E:/Graduation Project/Graduation-Project-ML/datasets/train.csv", header=None)
data_info = data_info.iloc[1:]  # Assuming your CSV has headers


# load the trained classifier and regressor
svm_classifier = joblib.load("svm_classifier.pkl")
bbox_regressor = joblib.load("bbox_regressor.pkl")

# Example usage
print('Testing the model.............')
image = cv2.imread("E:/Graduation Project/Graduation-Project-ML/datasets/mimic-cxr-jpg/files/p11/p11002268/s58301648/a57c42a3-a519a3eb-50a43237-c6d2eacb-fbae58b3.jpg", cv2.IMREAD_UNCHANGED)
original_height, original_width = image.shape[:2]
image = cv2.resize(image, (512, 512))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
detected_boxes = detect_objects(image, svm_classifier, bbox_regressor)

# Adjust detected boxes for visualization
scale_x = 512 / original_width
scale_y = 512 / original_height
detected_boxes_label = data_info.iloc[0, 4]
detected_boxes_label = eval(detected_boxes_label)
detected_boxes_label = [(int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)) for (x, y, w, h) in detected_boxes_label]

# calculate iou for each detected box
print("Calculating IOU.............")
iou = []
for box in detected_boxes:
    for label_box in detected_boxes_label:
        if calculate_iou(box, label_box) > 0.4:
            iou.append(calculate_iou(box, label_box))
            break
print("IOU: ", iou)
# Draw detected boxes on the image
print("len(detected_boxes): ", len(detected_boxes))
print("len(detected_boxes_label): ", len(detected_boxes_label))
for (x, y, w, h) in detected_boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
# for (x, y, w, h) in detected_boxes_label:
#     cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
plt.imshow(image)
plt.axis('off')
plt.show()
# python -m ObjectDetector.ObjectDetector