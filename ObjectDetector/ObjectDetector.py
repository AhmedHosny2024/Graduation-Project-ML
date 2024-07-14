import joblib
import cv2
import matplotlib.pyplot as plt
import pandas as pd
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
    _, regions = selective_search(image)
    detected_boxes = []
    print("len(regions): ", len(regions))
    for region in regions:
        x, y, w, h = region
        proposal_img = image[y:y+h, x:x+w]
        proposal_img = cv2.resize(proposal_img, (224, 224))  # Resize to VGG16 input size
        feature_vector = extract_features(proposal_img, 'hog')
        #concatenate the feature vector to be size 26244
        feature_vector = np.concatenate((feature_vector, np.zeros(26244 - len(feature_vector))))
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

data_info = pd.read_csv("datasets/train.csv", header=None)
data_info = data_info.iloc[1:]  # Assuming your CSV has headers


# load the trained classifier and regressor
svm_classifier = joblib.load("svm_classifier.pkl")
bbox_regressor = joblib.load("bbox_regressor.pkl")

# Example usage
print('Testing the model.............')
image = cv2.imread("datasets/mimic-cxr-jpg/files/p11/p11002268/s57561051/4a62e451-665fb8d2-9d037176-e88bb926-9b6beed0.jpg", cv2.IMREAD_UNCHANGED)
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
    iou_best = 0
    for label_box in detected_boxes_label:
        iou_box=calculate_iou(box, label_box)
        if iou_box > 0.4:
            if iou_box > iou_best:
                iou_best = iou_box
    iou.append(iou_best)
print("IOU: ", iou)
# Draw detected boxes on the image
print("len(detected_boxes): ", len(detected_boxes))
print("len(detected_boxes_label): ", len(detected_boxes_label))
for (x, y, w, h) in detected_boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
plt.imshow(image)
plt.axis('off')
plt.show()
# python -m ObjectDetector.ObjectDetector