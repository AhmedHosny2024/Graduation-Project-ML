import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the X-ray image
image_path = 'datasets/mimic-cxr-jpg/files/p11/p11001469/s54076811/d0d2bd0c-8bc50aa2-a9ab3ca1-cf9c9404-543a10b7.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError(f"Image at path {image_path} could not be loaded.")

# Initialize SIFT detector
sift = cv2.SIFT_create()

# Detect SIFT keypoints and descriptors
keypoints, descriptors = sift.detectAndCompute(image, None)

# Draw keypoints on the image
image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,color=(0, 255, 0))

# Display the image with keypoints
plt.imshow(image_with_keypoints, cmap='gray')
plt.title('SIFT Keypoints')
plt.show()

# Define Regions of Interest (ROIs) - Example: Split image into 4 quadrants
height, width = image.shape
rois = [
    (0, 0, width // 2, height // 2),
    (width // 2, 0, width // 2, height // 2),
    (0, height // 2, width // 2, height // 2),
    (width // 2, height // 2, width // 2, height // 2),
]

# Extract SIFT features within each ROI
for (x, y, w, h) in rois:
    roi = image[y:y+h, x:x+w]
    roi_keypoints, roi_descriptors = sift.detectAndCompute(roi, None)
    image_with_roi_keypoints = cv2.drawKeypoints(roi, roi_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    plt.imshow(image_with_roi_keypoints, cmap='gray')
    plt.title(f'SIFT Keypoints in ROI ({x}, {y}, {w}, {h})')
    plt.show()


# Match features between template and each ROI
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Compare keypoints and descriptors between different ROIs
for i, (x1, y1, w1, h1) in enumerate(rois):
    for j, (x2, y2, w2, h2) in enumerate(rois):
        if i < j:
            roi1 = image[y1:y1+h1, x1:x1+w1]
            roi2 = image[y2:y2+h2, x2:x2+w2]
            
            roi1_keypoints, roi1_descriptors = sift.detectAndCompute(roi1, None)
            roi2_keypoints, roi2_descriptors = sift.detectAndCompute(roi2, None)
            
            matches = bf.match(roi1_descriptors, roi2_descriptors)
            
            matched_image = cv2.drawMatches(roi1, roi1_keypoints, roi2, roi2_keypoints, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            plt.imshow(matched_image, cmap='gray')
            plt.title(f'Matched Features between ROI ({x1}, {y1}, {w1}, {h1}) and ROI ({x2}, {y2}, {w2}, {h2})')
            plt.show()
