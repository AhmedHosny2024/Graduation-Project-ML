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
data_info = pd.read_csv("/content/Graduation-Project-ML/datasets/train-10000.csv", header=None)
data_info = data_info.iloc[1:]  # Assuming your CSV has headers

# Initialize lists to store features and labels
features = []
labels = []
bbox_targets = []

# Load images and extract region proposals
count_not_found=0
for idx in range(len(data_info)):
    print("Processing image {}/{}".format(idx+1, len(data_info)))
    img_path = data_info.iloc[idx, 3]
    img_path = os.path.join(os.getcwd(), img_path.replace("\\", "/"))
    image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if image is None:
      count_not_found+=1
      print("Image not found ", count_not_found)
      continue
    
    original_height, original_width = image.shape[:2]
    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, regions = selective_search(image, min_size=1000)
    
    ground_truth_boxes = data_info.iloc[idx, 4]
    ground_truth_boxes = eval(ground_truth_boxes)
    
    # Scale ground truth boxes according to resized image dimensions
    scale_x = 512 / original_width
    scale_y = 512 / original_height
    scaled_gt_boxes = [(int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)) for (x, y, w, h) in ground_truth_boxes]
    
    for region in regions:
        x, y, w, h = region
        proposal_img = image[y:y+h, x:x+w]
        proposal_img = cv2.resize(proposal_img, (224, 224))  
        feature_vector = extract_features(proposal_img, 'hog')
        features.append(feature_vector)
        # Determine if the proposal is a positive or negative example
        label = 0  # Default to negative example
        for gt_box in scaled_gt_boxes:
            iou = calculate_iou((x, y, w, h), gt_box)
            if iou > 0.1:  # Consider a proposal as positive if IoU > 0.1
                label = 1
                dx = (gt_box[0] - x) / w
                dy = (gt_box[1] - y) / h
                dw = np.log(gt_box[2] / w)
                dh = np.log(gt_box[3] / h)
                bbox_targets.append((dx, dy, dw, dh))
                break
        if label == 0:
            bbox_targets.append((0, 0, 0, 0))
        labels.append(label)

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)
bbox_targets = np.array(bbox_targets)
# Split data into training and testing sets
X_train, X_test, y_train, y_test, bbox_train, bbox_test = train_test_split(features, labels, bbox_targets, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
print("start Training SVM.............")
svm_classifier = SVC(kernel='linear', class_weight='balanced')
svm_classifier.fit(X_train, y_train)
print("Start Training Bounding Box Regressor.............")
# Train the bounding box regressor
bbox_regressor = LinearRegression()
bbox_regressor.fit(X_train[y_train == 1], bbox_train[y_train == 1])
# Evaluate the classifier
y_pred = svm_classifier.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))

print("Saving the trained model.............")
joblib.dump(svm_classifier, "svm_classifier.pkl")
joblib.dump(bbox_regressor, "bbox_regressor.pkl")

# Example usage
print('Testing the model.............')
image = cv2.imread("/content/Graduation-Project-ML/datasets/mimic-cxr-jpg/files/p11/p11001469/s54076811/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg", cv2.IMREAD_UNCHANGED)
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
        if calculate_iou(box, label_box) > 0.1:
            iou.append(calculate_iou(box, label_box))
print("IOU: ", iou)
# Draw detected boxes on the image
for (x, y, w, h) in detected_boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
for (x, y, w, h) in detected_boxes_label:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
plt.imshow(image)
plt.axis('off')
plt.show()
# python -m ObjectDetector.full_object_detector