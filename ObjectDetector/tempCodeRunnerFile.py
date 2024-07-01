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