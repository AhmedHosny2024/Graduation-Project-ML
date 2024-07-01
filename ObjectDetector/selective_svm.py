import cv2
import selectivesearch
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image as keras_image
import numpy as np

# Load the pre-trained VGG16 model
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Function to extract features using VGG16
def extract_features(img):
    img = keras_image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = vgg_model.predict(img)
    return features.flatten()

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

# Function to detect objects in an image using the trained classifier
def detect_objects(image, classifier):
    _, regions = selectivesearch.selective_search(image, scale=500, sigma=0.9, min_size=100)
    detected_boxes = []
    for region in regions:
        x, y, w, h = region['rect']
        proposal_img = image[y:y+h, x:x+w]
        proposal_img = cv2.resize(proposal_img, (224, 224))  # Resize to VGG16 input size
        feature_vector = extract_features(proposal_img)
        prediction = classifier.predict([feature_vector])
        if prediction == 1:  # Assuming 1 is the positive class
            detected_boxes.append((x, y, w, h))
    return detected_boxes

# Load the dataset information
print("Loading dataset information...")
data_info = pd.read_csv("E:/Graduation Project/Graduation-Project-ML/datasets/train.csv", header=None)
data_info = data_info.iloc[1:]  # Assuming your CSV has headers

# Initialize lists to store features and labels
features = []
labels = []

print("Extracting features and labels...")
# Load images and extract region proposals
for idx in range(len(data_info)):
    img_path = data_info.iloc[idx, 3]
    img_path = os.path.join(os.getcwd(), img_path.replace("\\", "/"))
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image_label = data_info.iloc[idx, 4]  # Assuming the label is in the 5th column
    if image is None:
        continue

    image = cv2.resize(image, (512, 512))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, regions = selectivesearch.selective_search(image,min_size=1000)
    
    ground_truth_boxes = data_info.iloc[idx, 4]

    # convert the string representation of bounding boxes into list of list
    ground_truth_boxes = eval(ground_truth_boxes)
    ground_truth_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                           21, 22, 23, 24, 25, 26, 27, 28, 29]  # Replace with actual ground truth labels

    for region in regions:
        x, y, w, h = region['rect']
        proposal_img = image[y:y+h, x:x+w]
        proposal_img = cv2.resize(proposal_img, (224, 224))  # Resize to VGG16 input size
        feature_vector = extract_features(proposal_img)
        features.append(feature_vector)

        # Determine if the proposal is a positive or negative example
        label = 0  # Default to negative example
        for gt_box in ground_truth_boxes:
            if calculate_iou((x, y, w, h), gt_box) > 0.5:  # Consider a proposal as positive if IoU > 0.5
                label = 1
                break
        labels.append(label)

# Convert to numpy arrays
features = np.array(features)
labels = np.array(labels)

print("Dataset loaded. Training SVM classifier...")
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Initialize and train the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Evaluate the classifier
y_pred = svm_classifier.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Training complete.")

print("Classifier ready for object detection.")
# Example usage
image = cv2.imread("datasets/mimic-cxr-jpg/files/p11/p11001469/s54076811/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg")
image = cv2.resize(image, (512, 512))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
detected_boxes = detect_objects(image, svm_classifier)

# Draw detected boxes on the image
for (x, y, w, h) in detected_boxes:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

plt.imshow(image)
plt.axis('off')
plt.show()
